"""Storage: one SQLite file. Only the date is readable; all content is encrypted."""
import json
import sqlite3
import time

import numpy as np


class Store:
    def __init__(self, path, vault):
        self.v = vault
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS entries (day TEXT PRIMARY KEY, body BLOB, emb BLOB, updated REAL);
            CREATE TABLE IF NOT EXISTS insights(day TEXT PRIMARY KEY, body BLOB, updated REAL);
            CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY CHECK(id=1), body BLOB, updated REAL);
            CREATE TABLE IF NOT EXISTS chats   (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, body BLOB);
        """)

    def _enc(self, obj):
        return self.v.encrypt(json.dumps(obj, ensure_ascii=False).encode())

    def _dec(self, blob):
        return json.loads(self.v.decrypt(blob))

    # diary entries
    def save_entry(self, day, text, mood, tags, emb):
        self.db.execute("REPLACE INTO entries VALUES (?,?,?,?)", (
            day, self._enc({"text": text, "mood": mood, "tags": tags}),
            self.v.encrypt(emb.astype(np.float32).tobytes()), time.time()))
        self.db.commit()

    def get_entry(self, day):
        r = self.db.execute("SELECT body FROM entries WHERE day=?", (day,)).fetchone()
        return self._dec(r[0]) if r else None

    def days(self):
        return [r[0] for r in self.db.execute("SELECT day FROM entries ORDER BY day")]

    def all_entries(self):
        """Returns [(day, entry_dict, embedding)] oldest first."""
        out = []
        for day, body, emb in self.db.execute("SELECT day, body, emb FROM entries ORDER BY day").fetchall():
            out.append((day, self._dec(body), np.frombuffer(self.v.decrypt(emb), dtype=np.float32)))
        return out

    # daily insights
    def save_insight(self, day, data):
        self.db.execute("REPLACE INTO insights VALUES (?,?,?)", (day, self._enc(data), time.time()))
        self.db.commit()

    def get_insight(self, day):
        r = self.db.execute("SELECT body FROM insights WHERE day=?", (day,)).fetchone()
        return self._dec(r[0]) if r else None

    def days_missing_insight(self):
        q = "SELECT day FROM entries WHERE day NOT IN (SELECT day FROM insights) ORDER BY day"
        return [r[0] for r in self.db.execute(q)]

    # rolling life profile
    def get_profile(self):
        r = self.db.execute("SELECT body FROM profile WHERE id=1").fetchone()
        return self._dec(r[0])["text"] if r else ""

    def save_profile(self, text):
        self.db.execute("REPLACE INTO profile VALUES (1,?,?)", (self._enc({"text": text}), time.time()))
        self.db.commit()

    # chat history
    def save_chat(self, data):
        self.db.execute("INSERT INTO chats(ts, body) VALUES (?,?)", (time.time(), self._enc(data)))
        self.db.commit()

    def chats(self, limit=20):
        rows = self.db.execute("SELECT ts, body FROM chats ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": ts, **self._dec(b)} for ts, b in rows]

    # export / wipe
    def export_all(self):
        return {"entries": {d: e for d, e, _ in self.all_entries()},
                "profile": self.get_profile(), "chats": self.chats(10_000)}

    def wipe(self):
        for t in ("entries", "insights", "profile", "chats"):
            self.db.execute(f"DELETE FROM {t}")
        self.db.commit()
        self.db.execute("VACUUM")
        
    def update_embedding(self, day, emb):
        self.db.execute(
        "UPDATE entries SET emb=?, updated=? WHERE day=?",
        (
            self.v.encrypt(
                emb.astype(np.float32).tobytes()
            ),
            time.time(),
            day,
        ),
    )

        self.db.commit()
