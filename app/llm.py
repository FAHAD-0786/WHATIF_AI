"""One interface for Ollama, Groq, Anthropic Claude, or mock."""

import json

import httpx

from . import config
from .privacy import scrub


GROQ_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)

ANTHROPIC_URL = (
    "https://api.anthropic.com/v1/messages"
)


# ============================================================
# MOCK RESPONSE
# ============================================================

_MOCK_ANSWER = {
    "short_answer": "Mock answer for testing.",

    "diary_says": [
        "Mock evidence (2026-01-01)"
    ],

    "options": [
        {
            "name": "Do it now",
            "what_happens": "Mock outcome",
            "chance_good": 40,
            "main_risk": "Mock risk",
        },
        {
            "name": "Wait",
            "what_happens": "Mock outcome",
            "chance_good": 60,
            "main_risk": "Mock risk",
        },
        {
            "name": "Ask family first",
            "what_happens": "Mock outcome",
            "chance_good": 55,
            "main_risk": "Mock risk",
        },
    ],

    "blind_spots": [
        "Mock blind spot"
    ],

    "actions": [
        "Mock action 1",
        "Mock action 2",
        "Mock action 3",
    ],

    "confidence": {
        "level": "Low",
        "why": "Mock mode.",
    },
}


# ============================================================
# LLM
# ============================================================

class LLM:

    def __init__(
        self,
        mode=None,
    ):
        self.mode = (
            mode or config.LLM_MODE
        ).lower()

    @property
    def is_cloud(self):

        return self.mode in (
            "groq",
            "cloud",
        )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    def generate(
        self,
        system,
        user,
        max_tokens=1200,
        as_json=False,
    ):

        # Mock
        if self.mode == "mock":

            if as_json:
                return json.dumps(
                    _MOCK_ANSWER
                )

            return (
                "[mock] "
                + user[:200]
            )

        # Cloud privacy protection
        if self.is_cloud:

            user = scrub(
                user,
                config.PRIVATE_NAMES,
            )

        try:

            # Groq
            if self.mode == "groq":

                return self._groq(
                    system,
                    user,
                    max_tokens,
                    as_json,
                )

            # Claude
            if self.mode == "cloud":

                return self._claude(
                    system,
                    user,
                    max_tokens,
                )

            # Ollama
            return self._ollama(
                system,
                user,
                as_json,
            )

        except httpx.ConnectError:

            if self.is_cloud:

                raise RuntimeError(
                    "Cannot reach the internet."
                )

            raise RuntimeError(
                "Ollama is not running. "
                f"Start it, then run: "
                f"ollama pull {config.OLLAMA_MODEL}"
            )

        except httpx.TimeoutException:

            raise RuntimeError(
                "The model took too long to answer."
            )

        except httpx.HTTPStatusError as e:

            # IMPORTANT:
            # Show Groq's real error instead of
            # hiding it behind "401".
            try:
                detail = e.response.text[:1000]
            except Exception:
                detail = ""

            raise RuntimeError(
                f"{self.mode} returned error "
                f"{e.response.status_code}: "
                f"{detail}"
            )

    # --------------------------------------------------------
    # OLLAMA
    # --------------------------------------------------------

    def _ollama(
        self,
        system,
        user,
        as_json,
    ):

        body = {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "options": {
                "temperature": 0.3
            },
            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
        }

        if as_json:

            body["format"] = "json"

        r = httpx.post(
            f"{config.OLLAMA_URL}/api/chat",
            json=body,
            timeout=300,
        )

        r.raise_for_status()

        return r.json()[
            "message"
        ][
            "content"
        ]

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    def _groq(
        self,
        system,
        user,
        max_tokens,
        as_json,
    ):

        if not config.GROQ_API_KEY:

            raise RuntimeError(
                "GROQ_API_KEY is empty. "
                "Add it to .env."
            )

        body = {
            "model": config.GROQ_MODEL,
            "temperature": 0.3,

            # Current Groq parameter.
            "max_completion_tokens": max_tokens,

            "messages": [
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
        }

        headers = {
            "Authorization":
                f"Bearer {config.GROQ_API_KEY}",

            "Content-Type":
                "application/json",
        }

        if as_json:

            body[
                "response_format"
            ] = {
                "type": "json_object"
            }

        # ----------------------------------------------------
        # FIRST REQUEST
        # ----------------------------------------------------

        r = httpx.post(
            GROQ_URL,
            json=body,
            headers=headers,
            timeout=120,
        )

        print(
            "[GROQ] Status:",
            r.status_code,
        )

        if r.status_code >= 400:

            print(
                "[GROQ] Error:",
                r.text[:2000],
            )

        # ----------------------------------------------------
        # JSON MODE FALLBACK
        # ----------------------------------------------------

        if (
            r.status_code == 400
            and as_json
        ):

            print(
                "[GROQ] JSON response mode "
                "was rejected."
            )

            body.pop(
                "response_format",
                None,
            )

            r = httpx.post(
                GROQ_URL,
                json=body,
                headers=headers,
                timeout=120,
            )

            print(
                "[GROQ] Retry status:",
                r.status_code,
            )

            if r.status_code >= 400:

                print(
                    "[GROQ] Retry error:",
                    r.text[:2000],
                )

        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        r.raise_for_status()

        data = r.json()

        return data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

    # --------------------------------------------------------
    # CLAUDE
    # --------------------------------------------------------

    def _claude(
        self,
        system,
        user,
        max_tokens,
    ):

        if not config.ANTHROPIC_API_KEY:

            raise RuntimeError(
                "ANTHROPIC_API_KEY is empty."
            )

        r = httpx.post(
            ANTHROPIC_URL,
            timeout=120,

            headers={
                "x-api-key":
                    config.ANTHROPIC_API_KEY,

                "anthropic-version":
                    "2023-06-01",
            },

            json={
                "model":
                    config.CLOUD_MODEL,

                "max_tokens":
                    max_tokens,

                "system":
                    system,

                "messages": [
                    {
                        "role": "user",
                        "content": user,
                    }
                ],
            },
        )

        r.raise_for_status()

        return "".join(
            block.get(
                "text",
                "",
            )
            for block in r.json()[
                "content"
            ]
        )