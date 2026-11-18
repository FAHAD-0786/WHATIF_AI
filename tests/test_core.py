import json
from datetime import date

import pytest

from app import config, engine, llm as llm_module
from app.crypto import Vault, WrongPassphrase
from app.llm import LLM
from app.privacy import scrub
from app.store import Store


@pytest.fixture()
def env(tmp_path):
    v = Vault(tmp_path)
    v.unlock("correct horse battery")
    return tmp_path, v, Store(tmp_path / "t.db", v)


def test_wrong_passphrase(env):
    folder, v, _ = env
    v2 = Vault(folder)
    with pytest.raises(WrongPassphrase):
        v2.unlock("not the passphrase")
    v2.unlock("correct horse battery")
    assert v2.unlocked


def test_nothing_readable_on_disk(env):
    folder, v, s = env
    engine.save_entry(s, "2026-01-01", "SECRETWORD my salary talk", "good", ["work"])
    assert b"SECRETWORD" not in (folder / "t.db").read_bytes()
    assert s.get_entry("2026-01-01")["text"].startswith("SECRETWORD")


def test_retrieval_finds_related_entry(env):
    _, _, s = env
    engine.save_entry(s, "2026-01-01", "I felt stressed about my job and my manager at work")
    engine.save_entry(s, "2026-01-02", "Cooked pasta and walked in the park with friends")
    assert engine.retrieve(s, "should I quit my job", k=1)[0][0] == "2026-01-01"


def test_daily_update_and_whatif(env):
    _, _, s = env
    mock = LLM("mock")
    engine.save_entry(s, "2026-01-01", "Thinking about starting a business", "okay")
    assert engine.catch_up(s, mock) == ["2026-01-01"]
    assert s.get_profile()
    out = engine.what_if(s, mock, "What if I start a business?")
    assert out["evidence"] == ["2026-01-01"] and s.chats()
    assert 3 <= len(out["result"]["options"]) <= 4 and len(out["result"]["actions"]) == 3


def test_parse_answer_cleans_and_limits():
    raw = json.dumps({"short_answer": "**Wait**", "diary_says": ["a", "b", "c", "d"], "actions": ["1", "2", "3", "4"],
                      "options": [{"name": f"# o{i}", "what_happens": "x", "chance_good": "about 70 %",
                                   "main_risk": "y"} for i in range(6)],
                      "confidence": {"level": "Low", "why": "few entries"}})
    r = engine.parse_answer("Sure! " + raw)
    assert r["short_answer"] == "Wait" and len(r["options"]) == 4 and r["options"][0]["name"] == "o0"
    assert r["options"][0]["chance_good"] == "70%" and len(r["diary_says"]) == 3 and len(r["actions"]) == 3


def test_parse_answer_falls_back_to_text():
    assert engine.parse_answer("just *plain* words") == {"raw": "just plain words"}


def test_scrub():
    t = scrub("Mail ravi@x.com or call +91 98765 43210, Priya agreed", ["Priya"])
    assert "ravi@x.com" not in t and "98765" not in t and "Priya" not in t


def test_groq_request_is_anonymised(monkeypatch):
    sent = {}

    class Reply:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return Reply()

    monkeypatch.setattr(llm_module.httpx, "post", fake_post)
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(config, "PRIVATE_NAMES", ["Priya"])
    assert LLM("groq").generate("sys", "Priya wrote to ravi@x.com", as_json=True) == "{}"
    user_msg = sent["body"]["messages"][1]["content"]
    assert "Priya" not in user_msg and "ravi@x.com" not in user_msg
    assert sent["headers"]["Authorization"] == "Bearer gsk_test"
    assert sent["body"]["response_format"] == {"type": "json_object"}


def test_on_this_day(env):
    _, _, s = env
    engine.save_entry(s, "2025-11-15", "last year")
    assert engine.on_this_day(s, date(2026, 11, 15))[0]["day"] == "2025-11-15"
