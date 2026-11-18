"""Daily updates, diary retrieval and the "what if" answer."""

import json
import re
from datetime import date, datetime

import numpy as np

from . import config
from .embed import embed
from .rag import RAGIndex

rag_index = RAGIndex()

# ============================================================
# BASIC HELPERS
# ============================================================

_STOP = set(
    """
    a an the and or but if of to in on for with at by from
    is am are was were be been i me my we you your it this
    that what how do does did can will would should could
    have has had not no so as about
    """.split()
)


def _words(text):
    return {
        w[:5]
        for w in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(w) > 2 and w not in _STOP
    }


def plain(text):
    """Remove markdown-like formatting from model output."""
    return re.sub(r"[*#`]", "", str(text)).strip()


# ============================================================
# DAILY DIARY PROMPT
# ============================================================

SYSTEM_DIARY = (
    "You analyse a private diary. Be concrete and honest. "
    "Use only facts in the diary. "
    "Plain text only: no markdown, no asterisks, no heading symbols."
)


# ============================================================
# WHAT-IF PROMPT
# ============================================================

SYSTEM_WHATIF = """
You help a person think through a life choice using their diary evidence and life profile.

Reply with ONE valid JSON object only.

Use exactly these keys:

{
  "short_answer": "one or two sentences that directly answer the question",

  "diary_says": [
    "up to 3 short points about the person that matter here, each ending with a diary date like (2026-11-12)"
  ],

  "options": [
    {
      "name": "short option name",
      "what_happens": "short explanation",
      "chance_good": 40,
      "main_risk": "main risk"
    }
  ],

  "blind_spots": [
    "up to 3 hidden risks"
  ],

  "actions": [
    "exactly 3 concrete actions for the next 30 days"
  ],

  "confidence": {
    "level": "Low",
    "why": "short explanation"
  }
}

IMPORTANT RULES:

1. Never invent diary facts.

2. Use ONLY facts actually present in DIARY ENTRIES.

3. The LIFE PROFILE is background context only.
   Never claim that something came from the diary if it only came from the profile.

4. diary_says must contain only facts supported by DIARY ENTRIES.

5. Every diary_says item MUST end with the exact diary date.

6. If there are no relevant diary entries, diary_says must be an empty list.

7. Give 3 to 4 realistic options.

8. Include waiting or delaying when it genuinely makes sense.

9. Option names must be different.

10. chance_good is NOT a scientific probability.
    It is only a rough subjective estimate for this person.

11. Keep chance_good between 15 and 85.

12. Do not pretend to know the future.

13. Every action must be concrete and useful.

14. Give exactly 3 actions.

15. Spread the actions across the next 30 days:
    - one action for week 1
    - one action for weeks 2-3
    - one action for week 4

16. If there are fewer than 5 diary entries OR there are no relevant entries,
    confidence must be Low.

17. Every text value should be short, preferably under 25 words.

18. No markdown.

19. No asterisks.

20. No introductory explanation outside the JSON.

21. The answer should be practical, balanced and specific to the person's diary.

22. Do not automatically recommend the most conservative option.
    Let the diary evidence determine the recommendation.
"""


# ============================================================
# SAFETY
# ============================================================

_RISKY = re.compile(
    r"pregnan|deliver(y|ies)|delivary|dalivary|"
    r"labou?r pain|contraction|newborn|baby|"
    r"medicin|medication|overdose|suicid|self.?harm|"
    r"kill myself|chest pain|bleeding|ambulance|"
    r"emergency(?! fund)",
    re.I,
)

SAFETY_TEXT = (
    "This question is about health or safety, and a diary-based "
    "simulation cannot judge it. Please speak to your doctor or "
    "hospital now. In an emergency call 112."
)


# ============================================================
# SAVE DIARY ENTRY
# ============================================================

def save_entry(store, day, text, mood=None, tags=None):

    store.save_entry(
        day,
        text,
        mood,
        tags or [],
        # Keep the existing database schema compatible.
        # RAG embeddings are rebuilt from the encrypted diary.
        embed(text, is_query=False),
    )

    # Refresh the private in-memory RAG index.
    rebuild_rag_index(
        store
    )


# ============================================================
# DAILY INSIGHTS
# ============================================================

def daily_update(store, llm, day):
    """Summarise one day, then fold it into the long-term life profile."""

    e = store.get_entry(day)

    if not e:
        return None

    summary = plain(
        llm.generate(
            SYSTEM_DIARY,
            (
                f"Diary entry for {day} "
                f"(mood: {e.get('mood')}):\n"
                f"{e['text']}\n\n"
                "Write five short lines:\n"
                "Themes\n"
                "Emotions\n"
                "Decisions or worries\n"
                "One lesson\n"
                "One suggestion for tomorrow\n"
                "Maximum 100 words."
            ),
            max_tokens=300,
        )
    )

    profile = plain(
        llm.generate(
            SYSTEM_DIARY,
            (
                f"Current life profile (may be empty):\n"
                f"{store.get_profile()}\n\n"
                f"New day {day}:\n"
                f"{summary}\n\n"
                "Rewrite the profile in under 200 words covering:\n"
                "values, habits, fears, goals, decision style, "
                "relationships by role, strengths and risks.\n"
                "Keep what is still true."
            ),
            max_tokens=500,
        )
    )

    store.save_insight(day, {"summary": summary})
    store.save_profile(profile)

    return {"summary": summary}


def catch_up(store, llm):
    """Run daily updates for every day that has no insight."""

    done = []

    for d in store.days_missing_insight():
        daily_update(store, llm, d)
        done.append(d)

    return done


# ============================================================
# KEYWORD RETRIEVAL
# ============================================================

# ============================================================
# RAG INDEX
# ============================================================

def rebuild_rag_index(store):
    """
    Rebuild the private in-memory RAG index.

    SQLite remains the source of truth.
    """

    entries = list(
        store.all_entries()
    )

    chunk_count = rag_index.build(
        entries
    )

    print(
        f"[RAG] Indexed {len(entries)} diary entries "
        f"into {chunk_count} chunks."
    )

    return chunk_count


def retrieve(store, question, k=6):
    """
    Hybrid RAG retrieval.

    Semantic similarity is primary.
    Keyword overlap provides a secondary signal.

    No hard semantic threshold is applied here.
    We rank candidates and return the best evidence.
    """

    # Build the index automatically if necessary.
    if not rag_index.chunks:

        rebuild_rag_index(
            store
        )

    results = rag_index.search(
        question,
        k=k
    )

    if config.RETRIEVAL_DEBUG:

        rag_index.debug_search(
            question,
            k=10
        )

    # --------------------------------------------------------
    # Convert RAG chunks back into the structure expected by
    # your existing what_if() function.
    # --------------------------------------------------------

    output = []

    seen_days = set()

    for result in results:

        chunk = result["chunk"]

        # Prevent the same diary day appearing multiple times.
        if chunk.day in seen_days:
            continue

        seen_days.add(
            chunk.day
        )

        entry = {
            "text": chunk.text,
            "rag_score": result["score"],
            "semantic_score": result["semantic_score"],
            "keyword_score": result["keyword_score"],
        }

        output.append(
            (
                chunk.day,
                entry
            )
        )

    return output
# ============================================================
# REINDEX
# ============================================================

def reindex_entries(store):
    """
    Recreate existing SQLite embeddings and rebuild
    the chunk-based in-memory RAG index.
    """

    count = 0

    for day, entry, _old_vector in store.all_entries():

        text = entry.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        new_vector = embed(
            text,
            is_query=False
        )

        store.update_embedding(
            day,
            new_vector
        )

        count += 1

    # Build the actual chunk-based RAG index.
    chunk_count = rebuild_rag_index(
        store
    )

    print(
        f"[RAG] Reindexed {count} entries "
        f"into {chunk_count} chunks."
    )

    return count

# ============================================================
# PARSE MODEL ANSWER
# ============================================================

def _chance(value):
    """Convert model chance value to display-friendly percentage."""

    match = re.search(
        r"\d+",
        str(value),
    )

    if not match:
        return "?"

    number = int(match.group())

    number = max(
        0,
        min(number, 100),
    )

    return f"{number}%"


def parse_answer(raw):
    """
    Convert model JSON into the structure expected by the UI.

    IMPORTANT:
    Uses the ORIGINAL schema:
        short_answer
        diary_says
        options
        blind_spots
        actions
        confidence
    """

    try:

        # Find the JSON object even if the model accidentally
        # places a small amount of text before/after it.
        start = raw.index("{")
        end = raw.rindex("}") + 1

        data = json.loads(
            raw[start:end]
        )

    except (
        ValueError,
        AttributeError,
        json.JSONDecodeError,
    ):
        return {
            "raw": plain(raw)
        }

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    options = []

    for option in data.get("options", []):

        if not isinstance(option, dict):
            continue

        options.append(
            {
                "name": plain(
                    option.get("name", "")
                ),
                "what_happens": plain(
                    option.get("what_happens", "")
                ),
                "chance_good": _chance(
                    option.get("chance_good")
                ),
                "main_risk": plain(
                    option.get("main_risk", "")
                ),
            }
        )

    options = options[:4]

    # If the model failed to provide options,
    # don't pretend the answer is valid.
    if not options:
        return {
            "raw": plain(raw)
        }

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = data.get(
        "confidence"
    )

    if not isinstance(
        confidence,
        dict,
    ):
        confidence = {}

    # --------------------------------------------------------
    # LIST HELPER
    # --------------------------------------------------------

    def items(key, limit):

        values = data.get(
            key,
            [],
        )

        if not isinstance(
            values,
            list,
        ):
            return []

        return [
            plain(value)
            for value in values
            if isinstance(value, str)
        ][:limit]

    # --------------------------------------------------------
    # FINAL STRUCTURE
    # --------------------------------------------------------

    return {
        "short_answer": plain(
            data.get(
                "short_answer",
                "",
            )
        ),

        "diary_says": items(
            "diary_says",
            3,
        ),

        "options": options,

        "blind_spots": items(
            "blind_spots",
            3,
        ),

        "actions": items(
            "actions",
            3,
        ),

        "confidence": {
            "level": plain(
                confidence.get(
                    "level",
                    "Low",
                )
            ),
            "why": plain(
                confidence.get(
                    "why",
                    "",
                )
            ),
        },
    }


# ============================================================
# WHAT-IF
# ============================================================

def what_if(store, llm, question):

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if _RISKY.search(question):

        result = {
            "raw": SAFETY_TEXT
        }

        store.save_chat(
            {
                "question": question,
                "result": result,
                "evidence": [],
            }
        )

        return {
            "result": result,
            "evidence": [],
            "entries_total": len(
                store.days()
            ),
        }

    # --------------------------------------------------------
    # RETRIEVE RELEVANT DIARY ENTRIES
    # --------------------------------------------------------

    hits = retrieve(
        store,
        question,
        k=6,
    )

    days = [
        day
        for day, _entry in hits
    ]

    # --------------------------------------------------------
    # BUILD EVIDENCE
    # --------------------------------------------------------

    evidence_parts = []

    for day, entry in sorted(
        hits,
        key=lambda item: item[0],
    ):

        text = entry.get(
            "text",
            "",
        )

        # Send only the configured amount.
        text = text[
            :config.SNIPPET_CHARS
        ]

        evidence_parts.append(
            f"[{day}] {text}"
        )

    evidence = "\n".join(
        evidence_parts
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = (
        f"Today: {date.today().isoformat()}\n\n"

        "BACKGROUND PROFILE "
        "(context only, never cite it as written by the person):\n"

        f"{store.get_profile() or '(not built yet)'}\n\n"

        f"DIARY ENTRIES ({len(hits)} relevant entries found):\n"

        f"{evidence or '(no relevant diary entries found)'}\n\n"

        f"THE CHOICE:\n{question}"
    )

    # --------------------------------------------------------
    # GROQ / OLLAMA
    # --------------------------------------------------------

    raw = llm.generate(
        SYSTEM_WHATIF,
        prompt,
        max_tokens=1200,
        as_json=True,
    )

    result = parse_answer(
        raw
    )

    # --------------------------------------------------------
    # VALIDATE DIARY DATES
    # --------------------------------------------------------

    if "raw" not in result:

        result["diary_says"] = [
            item
            for item in result["diary_says"]
            if any(
                day in item
                for day in days
            )
        ]

        # If retrieval found nothing,
        # confidence must be Low.
        if not days:

            result["confidence"] = {
                "level": "Low",
                "why": (
                    "No relevant diary entries were found, "
                    "so the answer has limited personal evidence."
                ),
            }

    # --------------------------------------------------------
    # SAVE CHAT
    # --------------------------------------------------------

    store.save_chat(
        {
            "question": question,
            "result": result,
            "evidence": days,
        }
    )

    return {
        "result": result,
        "evidence": days,
        "entries_total": len(
            store.days()
        ),
    }


# ============================================================
# ON THIS DAY
# ============================================================

def on_this_day(
    store,
    today=None,
):

    t = today or date.today()

    suffix = t.strftime(
        "-%m-%d"
    )

    return [
        {
            "day": day,
            "text": store.get_entry(day)["text"],
        }
        for day in store.days()
        if day.endswith(suffix)
        and day != t.isoformat()
    ]