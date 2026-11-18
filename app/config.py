"""Settings from the environment or a .env file in the project root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# LLM_MODE answers "what if" questions; DAILY_MODE writes daily insights and the profile.
LLM_MODE = os.getenv("LLM_MODE", "local").lower()
DAILY_MODE = os.getenv("DAILY_MODE", LLM_MODE).lower()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "claude-sonnet-5-5")

PRIVATE_NAMES = [n.strip() for n in os.getenv("PRIVATE_NAMES", "").split(",") if n.strip()]
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "600"))  # max characters per diary entry sent to an LLM
DAILY_UPDATE_AT = os.getenv("DAILY_UPDATE_AT", "22:00")
PORT = int(os.getenv("PORT", "8765"))


RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.55"))
RETRIEVAL_DEBUG = os.getenv("RETRIEVAL_DEBUG", "0") == "1"