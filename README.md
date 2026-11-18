# What If - your private future simulator

Write a short diary each day. Later ask "What if I choose X?" and get a short answer, a table of 3-4 options
with chances, hidden risks and 3 actions for the next 30 days, built from your own diary with dates as evidence.

It is a reflection tool, not a crystal ball. Accuracy depends on how honest and regular your diary is.
Check the Confidence line in every answer.

## Quick start

1. Install Python 3.11+ (https://python.org).
2. Get a free Groq key at https://console.groq.com/keys.
3. Install Ollama (https://ollama.com) and run once: `ollama pull qwen2.5:7b-instruct`
   (daily insights use it, so full diary entries stay on your PC).
4. Run `run.bat` (Windows) or `./run.sh` (Mac/Linux). It creates `.env` on first run.
5. Put your key in `.env` as `GROQ_API_KEY=...`, then start the app again.
6. Open http://127.0.0.1:8765 and choose a passphrase (8+ characters). Write it down: it cannot be recovered.

No Groq and no Ollama? Set `LLM_MODE=mock` and `DAILY_MODE=mock` to try the app with fake answers.

## Where your data lives

Everything is in one file: `data/whatif.db` (plus `vault.salt` and `vault.check` in the same folder).

| What | Stored in | Readable on disk? |
|---|---|---|
| Diary text, mood, tags | `entries` table | No, AES-256-GCM encrypted |
| Search vectors (the RAG index) | `entries.emb` column | No, encrypted |
| Daily insights | `insights` table | No, encrypted |
| Life profile | `profile` table | No, encrypted |
| Past what-if answers | `chats` table | No, encrypted |
| Dates | all tables | Yes, dates only |

The key is made from your passphrase (scrypt), kept only in memory, and never written to disk.
Embeddings are created on your PC by fastembed. They are encrypted on purpose: stored as plain vectors they could
be turned back into text. At each question the app decrypts them in memory, finds the most related diary days,
and passes only those to the model. That is the retrieval step of a RAG system, without a separate vector database
that would hold your data unencrypted.

## What leaves your computer

| Mode | What is sent | Where |
|---|---|---|
| `local` | Nothing | Stays on your PC |
| `groq` / `cloud` | Your question, your life profile and up to 6 diary excerpts (first `SNIPPET_CHARS` characters each), with emails, links, phone numbers and `PRIVATE_NAMES` replaced by placeholders | Groq / Anthropic servers, for that one request |

Never sent in any mode: the database, the passphrase, the full diary, the search vectors.

Recommended: `LLM_MODE=groq` and `DAILY_MODE=local`. Daily insights read whole diary entries, so keeping them local
means the cloud only ever sees short excerpts for questions you ask.
Only the names you list in `PRIVATE_NAMES` are hidden, so list every person and company you write about.
Leave out anything you never want to leave your PC, or use `LLM_MODE=local`.

## Daily use

1. Diary tab: pick the date, write, press Save entry. Generate daily insight also runs automatically at
   `DAILY_UPDATE_AT` while the app is unlocked, and missed days are caught up.
2. What if tab: type a decision and press Simulate.
3. My profile shows what the app has learned about you. History shows past answers. Export downloads a readable copy.
4. Lock when done. Locking removes the key from memory.

Tip: 5 minutes a day for 3-4 weeks gives much better answers than one long entry.

## Settings (`.env`)

| Setting | Meaning |
|---|---|
| `LLM_MODE` | Brain for what-if answers: `local`, `groq`, `cloud`, `mock` |
| `DAILY_MODE` | Brain for daily insights and profile (default: same as `LLM_MODE`) |
| `GROQ_API_KEY`, `GROQ_MODEL` | Groq key and model, default `llama-3.3-70b-versatile` |
| `OLLAMA_MODEL` | Any model you pulled, e.g. `qwen2.5:7b-instruct` |
| `ANTHROPIC_API_KEY`, `CLOUD_MODEL` | Only for `cloud` mode |
| `PRIVATE_NAMES` | `Priya,Ravi,Acme Ltd` are hidden before cloud calls |
| `SNIPPET_CHARS` | Max characters of each diary entry sent to a cloud model (600) |
| `DAILY_UPDATE_AT` | `22:00` |
| `PORT` | `8765` |

To try a larger Groq model, change `GROQ_MODEL` to any chat model listed in your Groq console.

## Project map

```
app/main.py      web server and API (localhost only)
app/engine.py    daily updates, retrieval, what-if prompt, answer parsing
app/crypto.py    passphrase to key, AES-GCM encrypt/decrypt
app/store.py     encrypted SQLite storage
app/embed.py     local semantic search (fastembed, with fallback)
app/llm.py       Ollama / Groq / Claude / mock
app/privacy.py   anonymiser for cloud calls
app/static/      the web page
tests/           automatic tests (run: python -m pytest)
data/            your encrypted data (never share or upload this folder)
```

## Backup and delete

- Backup: copy the whole `data/` folder. It is encrypted and useless without your passphrase.
- Delete everything: delete the `data/` folder.

## Troubleshooting

- "Ollama is not running": start Ollama and run `ollama pull <model>`.
- "GROQ_API_KEY is empty": add the key to `.env` and restart.
- "groq returned error 401 / 429": wrong key / rate limit, wait a minute.
- Wrong passphrase: there is no reset by design. Restore a backup or delete `data/` to start fresh.
  After 5 wrong tries the app waits 60 seconds.
- Port busy: change `PORT` in `.env`.
- Never expose this app to the internet (no port-forwarding, no tunnels).
