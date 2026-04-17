"""
MindTrack — AI Mental Wellness Backend
FastAPI + SQLite + Anthropic Claude Integration
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import sqlite3
import hashlib
import hmac
import os
import json
import time
import uuid
from datetime import datetime, timedelta
import anthropic

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MindTrack API",
    description="AI-powered Mental Wellness Backend — Powered by Anthropic Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ─── Config ───────────────────────────────────────────────────────────────────

DATABASE = "mindtrack.db"
SECRET_KEY = os.getenv("SECRET_KEY", "mindtrack-secret-change-in-production")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            email       TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            password    TEXT NOT NULL,
            roll_no     TEXT,
            reminder_time TEXT DEFAULT '09:00',
            notifications INTEGER DEFAULT 1,
            auto_analyze  INTEGER DEFAULT 1,
            topic_modeling INTEGER DEFAULT 1,
            wellness_index_enabled INTEGER DEFAULT 1,
            dark_mode     INTEGER DEFAULT 0,
            data_retention TEXT DEFAULT 'Forever',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Journal entries table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            text            TEXT NOT NULL,
            mood_label      TEXT,
            analysis        TEXT,         -- JSON blob from Claude
            wellness_index  INTEGER,
            dominant_emotion TEXT,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Mood calendar table (daily snapshot)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_calendar (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            date       TEXT NOT NULL,
            mood       TEXT NOT NULL,      -- 'good' | 'neutral' | 'bad'
            score      INTEGER,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Sessions (simple token store)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


init_db()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256((password + SECRET_KEY).encode()).hexdigest()


def create_token(user_id: str) -> str:
    token = str(uuid.uuid4())
    expires = (datetime.utcnow() + timedelta(days=7)).isoformat()
    conn = sqlite3.connect(DATABASE)
    conn.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires),
    )
    conn.commit()
    conn.close()
    return token


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db),
):
    token = credentials.credentials
    row = db.execute(
        "SELECT s.user_id, s.expires_at, u.* FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        db.execute("DELETE FROM sessions WHERE token = ?", (token,))
        db.commit()
        raise HTTPException(status_code=401, detail="Token expired")
    return dict(row)

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    roll_no: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class JournalEntryCreate(BaseModel):
    text: str
    mood_label: Optional[str] = None   # quick mood from UI


class MoodCalendarCreate(BaseModel):
    date: str          # YYYY-MM-DD
    mood: str          # good | neutral | bad
    score: Optional[int] = None


class SettingsUpdate(BaseModel):
    name: Optional[str] = None
    reminder_time: Optional[str] = None
    notifications: Optional[bool] = None
    auto_analyze: Optional[bool] = None
    topic_modeling: Optional[bool] = None
    wellness_index_enabled: Optional[bool] = None
    dark_mode: Optional[bool] = None
    data_retention: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str


# ─── Claude Analysis ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert emotion analyst for the MindTrack mental wellness app, powered by a DistilRoBERTa emotion model.
Analyze the journal entry and return ONLY valid JSON with this exact structure:
{
  "emotions": [
    {"emotion": "Joy", "percentage": 0},
    {"emotion": "Calm", "percentage": 0},
    {"emotion": "Sadness", "percentage": 0},
    {"emotion": "Stress", "percentage": 0},
    {"emotion": "Anger", "percentage": 0},
    {"emotion": "Fear", "percentage": 0}
  ],
  "dominant": "Joy",
  "wellnessIndex": 72,
  "triggers": ["word1", "word2", "word3"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "summary": "One sentence emotional summary."
}
Rules: percentages must sum to 100. wellnessIndex is 0-100. triggers are actual keywords from the text. suggestions are actionable mental wellness tips. Return ONLY JSON, no markdown."""


def run_claude_analysis(text: str) -> dict:
    """Call Claude API and return parsed JSON analysis."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured on server.")

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f'Analyze this journal entry:\n\n"{text}"'}],
    )
    raw = "".join(b.text for b in message.content if hasattr(b, "text"))
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

# ─── Routes: Auth ─────────────────────────────────────────────────────────────

@app.post("/api/auth/register", summary="Register a new user", tags=["Auth"])
def register(body: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute("SELECT id FROM users WHERE email = ?", (body.email,)).fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO users (id, email, name, password, roll_no) VALUES (?, ?, ?, ?, ?)",
        (user_id, body.email, body.name, hash_password(body.password), body.roll_no),
    )
    db.commit()
    token = create_token(user_id)
    return {"token": token, "user": {"id": user_id, "email": body.email, "name": body.name}}


@app.post("/api/auth/login", summary="Login and receive a session token", tags=["Auth"])
def login(body: LoginRequest, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE email = ?", (body.email,)).fetchone()
    if not row or row["password"] != hash_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(row["id"])
    return {
        "token": token,
        "user": {
            "id": row["id"],
            "email": row["email"],
            "name": row["name"],
            "roll_no": row["roll_no"],
        },
    }


@app.post("/api/auth/logout", summary="Invalidate session token", tags=["Auth"])
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db),
):
    db.execute("DELETE FROM sessions WHERE token = ?", (credentials.credentials,))
    db.commit()
    return {"message": "Logged out successfully"}


@app.get("/api/auth/me", summary="Get current user profile", tags=["Auth"])
def me(current_user: dict = Depends(get_current_user)):
    return {k: v for k, v in current_user.items() if k != "password"}

# ─── Routes: Journal Entries ───────────────────────────────────────────────────

@app.post("/api/entries", summary="Create a journal entry (with optional AI analysis)", tags=["Journal"])
def create_entry(
    body: JournalEntryCreate,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    entry_id = str(uuid.uuid4())
    analysis = None
    wellness_index = None
    dominant = None

    # Run Claude analysis if auto_analyze is enabled
    if current_user.get("auto_analyze", 1):
        try:
            analysis = run_claude_analysis(body.text)
            wellness_index = analysis.get("wellnessIndex")
            dominant = analysis.get("dominant")
        except Exception as e:
            # Analysis failure is non-fatal; entry still saved
            analysis = {"error": str(e)}

    db.execute(
        """INSERT INTO journal_entries
           (id, user_id, text, mood_label, analysis, wellness_index, dominant_emotion)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            entry_id,
            current_user["id"],
            body.text,
            body.mood_label,
            json.dumps(analysis) if analysis else None,
            wellness_index,
            dominant,
        ),
    )

    # Update mood calendar for today
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if wellness_index is not None:
        mood = "good" if wellness_index >= 65 else ("neutral" if wellness_index >= 40 else "bad")
        db.execute(
            """INSERT INTO mood_calendar (id, user_id, date, mood, score)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, date) DO UPDATE SET mood=excluded.mood, score=excluded.score""",
            (str(uuid.uuid4()), current_user["id"], today, mood, wellness_index),
        )

    db.commit()

    return {
        "id": entry_id,
        "text": body.text,
        "mood_label": body.mood_label,
        "analysis": analysis,
        "wellness_index": wellness_index,
        "dominant_emotion": dominant,
        "created_at": datetime.utcnow().isoformat(),
    }


@app.get("/api/entries", summary="List all journal entries for current user", tags=["Journal"])
def list_entries(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        """SELECT id, text, mood_label, analysis, wellness_index, dominant_emotion, created_at
           FROM journal_entries WHERE user_id = ?
           ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (current_user["id"], limit, offset),
    ).fetchall()

    entries = []
    for r in rows:
        entry = dict(r)
        entry["analysis"] = json.loads(entry["analysis"]) if entry["analysis"] else None
        entries.append(entry)
    return {"entries": entries, "total": len(entries)}


@app.get("/api/entries/{entry_id}", summary="Get a single journal entry", tags=["Journal"])
def get_entry(
    entry_id: str,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    row = db.execute(
        "SELECT * FROM journal_entries WHERE id = ? AND user_id = ?",
        (entry_id, current_user["id"]),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entry not found")
    entry = dict(row)
    entry["analysis"] = json.loads(entry["analysis"]) if entry["analysis"] else None
    return entry


@app.delete("/api/entries/{entry_id}", summary="Delete a journal entry", tags=["Journal"])
def delete_entry(
    entry_id: str,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    result = db.execute(
        "DELETE FROM journal_entries WHERE id = ? AND user_id = ?",
        (entry_id, current_user["id"]),
    )
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"message": "Entry deleted"}

# ─── Routes: AI Analysis (standalone) ─────────────────────────────────────────

@app.post("/api/analyze", summary="Run emotion analysis on arbitrary text", tags=["AI"])
def analyze_text(
    body: AnalyzeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Analyze text without saving it as a journal entry."""
    result = run_claude_analysis(body.text)
    return result

# ─── Routes: Analytics ────────────────────────────────────────────────────────

@app.get("/api/analytics/weekly", summary="Weekly mood & stress data (last 7 days)", tags=["Analytics"])
def weekly_analytics(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        """SELECT date(created_at) as day,
                  AVG(wellness_index) as avg_mood,
                  COUNT(*) as entries
           FROM journal_entries
           WHERE user_id = ? AND created_at >= date('now', '-7 days')
           GROUP BY date(created_at)
           ORDER BY day ASC""",
        (current_user["id"],),
    ).fetchall()

    weekly = [
        {
            "day": r["day"],
            "mood": round(r["avg_mood"] or 0),
            "stress": max(0, 100 - round(r["avg_mood"] or 0)),
            "entries": r["entries"],
        }
        for r in rows
    ]
    return {"weekly": weekly}


@app.get("/api/analytics/monthly", summary="30-day wellness trend", tags=["Analytics"])
def monthly_analytics(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        """SELECT date(created_at) as day,
                  AVG(wellness_index) as avg_mood
           FROM journal_entries
           WHERE user_id = ? AND created_at >= date('now', '-30 days')
           GROUP BY date(created_at)
           ORDER BY day ASC""",
        (current_user["id"],),
    ).fetchall()

    monthly = [
        {"day": r["day"], "mood": round(r["avg_mood"] or 0)}
        for r in rows
    ]
    return {"monthly": monthly}


@app.get("/api/analytics/emotions", summary="Aggregated emotion distribution", tags=["Analytics"])
def emotion_distribution(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        "SELECT analysis FROM journal_entries WHERE user_id = ? AND analysis IS NOT NULL ORDER BY created_at DESC LIMIT 30",
        (current_user["id"],),
    ).fetchall()

    totals: dict[str, float] = {}
    count = 0
    for r in rows:
        try:
            data = json.loads(r["analysis"])
            for em in data.get("emotions", []):
                name = em["emotion"]
                totals[name] = totals.get(name, 0) + em["percentage"]
            count += 1
        except Exception:
            pass

    if count == 0:
        return {"emotions": []}

    distribution = [
        {"emotion": k, "percentage": round(v / count, 1)}
        for k, v in totals.items()
    ]
    distribution.sort(key=lambda x: x["percentage"], reverse=True)
    return {"emotions": distribution}


@app.get("/api/analytics/triggers", summary="Top emotional trigger words", tags=["Analytics"])
def trigger_words(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        "SELECT analysis FROM journal_entries WHERE user_id = ? AND analysis IS NOT NULL ORDER BY created_at DESC LIMIT 20",
        (current_user["id"],),
    ).fetchall()

    freq: dict[str, int] = {}
    for r in rows:
        try:
            data = json.loads(r["analysis"])
            for word in data.get("triggers", []):
                freq[word] = freq.get(word, 0) + 1
        except Exception:
            pass

    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    return {"triggers": [{"word": w, "count": c} for w, c in top]}


@app.get("/api/analytics/summary", summary="High-level wellness summary stats", tags=["Analytics"])
def analytics_summary(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    row = db.execute(
        """SELECT COUNT(*) as total_entries,
                  AVG(wellness_index) as avg_wellness,
                  MAX(wellness_index) as peak_wellness,
                  MIN(wellness_index) as lowest_wellness
           FROM journal_entries WHERE user_id = ?""",
        (current_user["id"],),
    ).fetchone()

    dominant_row = db.execute(
        """SELECT dominant_emotion, COUNT(*) as cnt
           FROM journal_entries WHERE user_id = ? AND dominant_emotion IS NOT NULL
           GROUP BY dominant_emotion ORDER BY cnt DESC LIMIT 1""",
        (current_user["id"],),
    ).fetchone()

    return {
        "total_entries": row["total_entries"] or 0,
        "avg_wellness": round(row["avg_wellness"] or 0, 1),
        "peak_wellness": row["peak_wellness"] or 0,
        "lowest_wellness": row["lowest_wellness"] or 0,
        "dominant_emotion": dominant_row["dominant_emotion"] if dominant_row else None,
    }

# ─── Routes: Mood Calendar ────────────────────────────────────────────────────

@app.get("/api/mood-calendar", summary="Get mood calendar data (last 30 days)", tags=["Mood"])
def mood_calendar(
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        """SELECT date, mood, score FROM mood_calendar
           WHERE user_id = ? AND date >= date('now', '-30 days')
           ORDER BY date ASC""",
        (current_user["id"],),
    ).fetchall()
    return {"calendar": [dict(r) for r in rows]}


@app.post("/api/mood-calendar", summary="Manually log a mood for a date", tags=["Mood"])
def log_mood(
    body: MoodCalendarCreate,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    db.execute(
        """INSERT INTO mood_calendar (id, user_id, date, mood, score)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(user_id, date) DO UPDATE SET mood=excluded.mood, score=excluded.score""",
        (str(uuid.uuid4()), current_user["id"], body.date, body.mood, body.score),
    )
    db.commit()
    return {"message": "Mood logged", "date": body.date, "mood": body.mood}

# ─── Routes: Settings ─────────────────────────────────────────────────────────

@app.get("/api/settings", summary="Get user settings", tags=["Settings"])
def get_settings(current_user: dict = Depends(get_current_user)):
    keys = [
        "name", "email", "roll_no", "reminder_time", "notifications",
        "auto_analyze", "topic_modeling", "wellness_index_enabled",
        "dark_mode", "data_retention",
    ]
    return {k: current_user.get(k) for k in keys}


@app.patch("/api/settings", summary="Update user settings", tags=["Settings"])
def update_settings(
    body: SettingsUpdate,
    current_user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return {"message": "Nothing to update"}

    set_clauses = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [current_user["id"]]
    db.execute(f"UPDATE users SET {set_clauses} WHERE id = ?", values)
    db.commit()
    return {"message": "Settings updated", "updated_fields": list(updates.keys())}

# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/api/health", summary="Health check", tags=["System"])
def health():
    return {
        "status": "ok",
        "service": "MindTrack API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "claude_configured": bool(ANTHROPIC_API_KEY),
    }
