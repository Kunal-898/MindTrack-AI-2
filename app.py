"""
MindTrack — AI Mental Wellness Backend
FastAPI + Uvicorn + DistilRoBERTa + Anthropic Claude
"""

import os
import json
import uuid
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from model import EmotionClassifier
from claude_service import ClaudeEnrichmentService

# ─── App Init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MindTrack API",
    description="AI Mental Wellness — DistilRoBERTa + Anthropic Claude",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ─── Startup: Load Model & Services ───────────────────────────────────────────

classifier: EmotionClassifier = None
enricher: ClaudeEnrichmentService = None

@app.on_event("startup")
def startup_event():
    global classifier, enricher
    classifier = EmotionClassifier()
    enricher = ClaudeEnrichmentService()
    init_db()
    print("✅ MindTrack API ready — model loaded, DB initialised")

# ─── Database ─────────────────────────────────────────────────────────────────

DATABASE = "mindtrack.db"

def get_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             TEXT PRIMARY KEY,
            email          TEXT UNIQUE NOT NULL,
            name           TEXT NOT NULL,
            password       TEXT NOT NULL,
            roll_no        TEXT,
            reminder_time  TEXT DEFAULT '09:00',
            notifications  INTEGER DEFAULT 1,
            auto_analyze   INTEGER DEFAULT 1,
            dark_mode      INTEGER DEFAULT 0,
            data_retention TEXT DEFAULT 'Forever',
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS journal_entries (
            id               TEXT PRIMARY KEY,
            user_id          TEXT NOT NULL,
            text             TEXT NOT NULL,
            mood_label       TEXT,
            analysis         TEXT,
            wellness_index   INTEGER,
            dominant_emotion TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS mood_calendar (
            id      TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            date    TEXT NOT NULL,
            mood    TEXT NOT NULL,
            score   INTEGER,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("SECRET_KEY", "mindtrack-secret-key-change-in-prod")

def hash_password(password: str) -> str:
    return hashlib.sha256((password + SECRET_KEY).encode()).hexdigest()

def create_token(user_id: str, db_conn) -> str:
    token = str(uuid.uuid4())
    expires = (datetime.utcnow() + timedelta(days=7)).isoformat()
    db_conn.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)",
        (token, user_id, expires),
    )
    db_conn.commit()
    return token

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db),
):
    row = db.execute(
        """SELECT s.expires_at, u.*
           FROM sessions s JOIN users u ON s.user_id = u.id
           WHERE s.token=?""",
        (creds.credentials,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        db.execute("DELETE FROM sessions WHERE token=?", (creds.credentials,))
        db.commit()
        raise HTTPException(status_code=401, detail="Token expired — please login again")
    return dict(row)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    roll_no: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class JournalRequest(BaseModel):
    text: str
    mood_label: Optional[str] = None

class AnalyzeRequest(BaseModel):
    text: str

class MoodLogRequest(BaseModel):
    date: str
    mood: str
    score: Optional[int] = None

class SettingsUpdateRequest(BaseModel):
    name: Optional[str] = None
    reminder_time: Optional[str] = None
    notifications: Optional[bool] = None
    auto_analyze: Optional[bool] = None
    dark_mode: Optional[bool] = None
    data_retention: Optional[str] = None

# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.post("/api/auth/register", tags=["Auth"])
def register(body: RegisterRequest, db: sqlite3.Connection = Depends(get_db)):
    if db.execute("SELECT id FROM users WHERE email=?", (body.email,)).fetchone():
        raise HTTPException(400, "Email already registered")
    uid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO users (id,email,name,password,roll_no) VALUES (?,?,?,?,?)",
        (uid, body.email, body.name, hash_password(body.password), body.roll_no),
    )
    db.commit()
    token = create_token(uid, db)
    return {"token": token, "user": {"id": uid, "name": body.name, "email": body.email}}

@app.post("/api/auth/login", tags=["Auth"])
def login(body: LoginRequest, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE email=?", (body.email,)).fetchone()
    if not row or row["password"] != hash_password(body.password):
        raise HTTPException(401, "Invalid email or password")
    token = create_token(row["id"], db)
    return {
        "token": token,
        "user": {"id": row["id"], "name": row["name"], "email": row["email"], "roll_no": row["roll_no"]},
    }

@app.post("/api/auth/logout", tags=["Auth"])
def logout(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: sqlite3.Connection = Depends(get_db),
):
    db.execute("DELETE FROM sessions WHERE token=?", (creds.credentials,))
    db.commit()
    return {"message": "Logged out"}

@app.get("/api/auth/me", tags=["Auth"])
def me(user=Depends(get_current_user)):
    return {k: v for k, v in user.items() if k not in ("password", "expires_at")}

# ─── Analyze Route ────────────────────────────────────────────────────────────

@app.post("/api/analyze", tags=["AI"])
def analyze(body: AnalyzeRequest, user=Depends(get_current_user)):
    """
    Pipeline:
      1. DistilRoBERTa → emotion scores + wellness index
      2. Claude         → triggers, suggestions, summary
      3. Merge and return
    """
    model_result = classifier.predict(body.text)
    enrichment = enricher.enrich(body.text, model_result)
    return {
        "emotions": model_result["emotions"],
        "dominant": model_result["dominant"],
        "wellnessIndex": model_result["wellnessIndex"],
        "triggers": enrichment.get("triggers", []),
        "suggestions": enrichment.get("suggestions", []),
        "summary": enrichment.get("summary", ""),
    }

# ─── Journal Routes ───────────────────────────────────────────────────────────

@app.post("/api/entries", tags=["Journal"])
def create_entry(
    body: JournalRequest,
    user=Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    entry_id = str(uuid.uuid4())
    analysis = wellness_index = dominant = None

    if user.get("auto_analyze", 1):
        try:
            model_result = classifier.predict(body.text)
            enrichment = enricher.enrich(body.text, model_result)
            analysis = {
                "emotions": model_result["emotions"],
                "dominant": model_result["dominant"],
                "wellnessIndex": model_result["wellnessIndex"],
                "triggers": enrichment.get("triggers", []),
                "suggestions": enrichment.get("suggestions", []),
                "summary": enrichment.get("summary", ""),
            }
            wellness_index = analysis["wellnessIndex"]
            dominant = analysis["dominant"]
        except Exception as e:
            analysis = {"error": str(e)}

    db.execute(
        """INSERT INTO journal_entries
           (id,user_id,text,mood_label,analysis,wellness_index,dominant_emotion)
           VALUES (?,?,?,?,?,?,?)""",
        (entry_id, user["id"], body.text, body.mood_label,
         json.dumps(analysis) if analysis else None, wellness_index, dominant),
    )

    today = datetime.utcnow().strftime("%Y-%m-%d")
    if wellness_index is not None:
        mood = "good" if wellness_index >= 65 else ("neutral" if wellness_index >= 40 else "bad")
        db.execute(
            """INSERT INTO mood_calendar (id,user_id,date,mood,score) VALUES (?,?,?,?,?)
               ON CONFLICT(user_id,date) DO UPDATE SET mood=excluded.mood, score=excluded.score""",
            (str(uuid.uuid4()), user["id"], today, mood, wellness_index),
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

@app.get("/api/entries", tags=["Journal"])
def list_entries(
    limit: int = 20,
    offset: int = 0,
    user=Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    rows = db.execute(
        """SELECT id,text,mood_label,analysis,wellness_index,dominant_emotion,created_at
           FROM journal_entries WHERE user_id=?
           ORDER BY created_at DESC LIMIT ? OFFSET ?""",
        (user["id"], limit, offset),
    ).fetchall()
    entries = []
    for r in rows:
        e = dict(r)
        e["analysis"] = json.loads(e["analysis"]) if e["analysis"] else None
        entries.append(e)
    return {"entries": entries, "count": len(entries)}

@app.get("/api/entries/{entry_id}", tags=["Journal"])
def get_entry(entry_id: str, user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        "SELECT * FROM journal_entries WHERE id=? AND user_id=?",
        (entry_id, user["id"]),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Entry not found")
    e = dict(row)
    e["analysis"] = json.loads(e["analysis"]) if e["analysis"] else None
    return e

@app.delete("/api/entries/{entry_id}", tags=["Journal"])
def delete_entry(entry_id: str, user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    r = db.execute(
        "DELETE FROM journal_entries WHERE id=? AND user_id=?",
        (entry_id, user["id"]),
    )
    db.commit()
    if r.rowcount == 0:
        raise HTTPException(404, "Entry not found")
    return {"message": "Entry deleted"}

# ─── Analytics Routes ─────────────────────────────────────────────────────────

@app.get("/api/analytics/weekly", tags=["Analytics"])
def weekly(user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """SELECT date(created_at) as day,
                  ROUND(AVG(wellness_index)) as mood, COUNT(*) as entries
           FROM journal_entries
           WHERE user_id=? AND created_at >= date('now','-7 days')
           GROUP BY date(created_at) ORDER BY day ASC""",
        (user["id"],),
    ).fetchall()
    return {"weekly": [
        {"day": r["day"], "mood": int(r["mood"] or 0),
         "stress": max(0, 100 - int(r["mood"] or 0)), "entries": r["entries"]}
        for r in rows
    ]}

@app.get("/api/analytics/monthly", tags=["Analytics"])
def monthly(user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """SELECT date(created_at) as day, ROUND(AVG(wellness_index)) as mood
           FROM journal_entries
           WHERE user_id=? AND created_at >= date('now','-30 days')
           GROUP BY date(created_at) ORDER BY day ASC""",
        (user["id"],),
    ).fetchall()
    return {"monthly": [{"day": r["day"], "mood": int(r["mood"] or 0)} for r in rows]}

@app.get("/api/analytics/emotions", tags=["Analytics"])
def emotions(user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT analysis FROM journal_entries WHERE user_id=? AND analysis IS NOT NULL ORDER BY created_at DESC LIMIT 30",
        (user["id"],),
    ).fetchall()
    totals: dict = {}
    count = 0
    for r in rows:
        try:
            for em in json.loads(r["analysis"]).get("emotions", []):
                totals[em["emotion"]] = totals.get(em["emotion"], 0) + em["percentage"]
            count += 1
        except Exception:
            pass
    if not count:
        return {"emotions": []}
    return {"emotions": sorted(
        [{"emotion": k, "percentage": round(v / count, 1)} for k, v in totals.items()],
        key=lambda x: x["percentage"], reverse=True,
    )}

@app.get("/api/analytics/triggers", tags=["Analytics"])
def triggers(user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT analysis FROM journal_entries WHERE user_id=? AND analysis IS NOT NULL ORDER BY created_at DESC LIMIT 20",
        (user["id"],),
    ).fetchall()
    freq: dict = {}
    for r in rows:
        try:
            for w in json.loads(r["analysis"]).get("triggers", []):
                freq[w] = freq.get(w, 0) + 1
        except Exception:
            pass
    return {"triggers": [{"word": w, "count": c} for w, c in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]]}

@app.get("/api/analytics/summary", tags=["Analytics"])
def summary(user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        """SELECT COUNT(*) as total, ROUND(AVG(wellness_index),1) as avg,
                  MAX(wellness_index) as peak, MIN(wellness_index) as lowest
           FROM journal_entries WHERE user_id=?""",
        (user["id"],),
    ).fetchone()
    dom = db.execute(
        """SELECT dominant_emotion, COUNT(*) as cnt FROM journal_entries
           WHERE user_id=? AND dominant_emotion IS NOT NULL
           GROUP BY dominant_emotion ORDER BY cnt DESC LIMIT 1""",
        (user["id"],),
    ).fetchone()
    return {
        "total_entries": row["total"] or 0,
        "avg_wellness": row["avg"] or 0,
        "peak_wellness": row["peak"] or 0,
        "lowest_wellness": row["lowest"] or 0,
        "dominant_emotion": dom["dominant_emotion"] if dom else None,
    }

# ─── Mood Calendar ────────────────────────────────────────────────────────────

@app.get("/api/mood-calendar", tags=["Mood"])
def get_calendar(user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        "SELECT date,mood,score FROM mood_calendar WHERE user_id=? AND date >= date('now','-30 days') ORDER BY date ASC",
        (user["id"],),
    ).fetchall()
    return {"calendar": [dict(r) for r in rows]}

@app.post("/api/mood-calendar", tags=["Mood"])
def log_mood(body: MoodLogRequest, user=Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    db.execute(
        """INSERT INTO mood_calendar (id,user_id,date,mood,score) VALUES (?,?,?,?,?)
           ON CONFLICT(user_id,date) DO UPDATE SET mood=excluded.mood, score=excluded.score""",
        (str(uuid.uuid4()), user["id"], body.date, body.mood, body.score),
    )
    db.commit()
    return {"message": "Mood logged", "date": body.date, "mood": body.mood}

# ─── Settings ─────────────────────────────────────────────────────────────────

@app.get("/api/settings", tags=["Settings"])
def get_settings(user=Depends(get_current_user)):
    keys = ["name", "email", "roll_no", "reminder_time", "notifications",
            "auto_analyze", "dark_mode", "data_retention"]
    return {k: user.get(k) for k in keys}

@app.patch("/api/settings", tags=["Settings"])
def update_settings(
    body: SettingsUpdateRequest,
    user=Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return {"message": "Nothing to update"}
    clauses = ", ".join(f"{k}=?" for k in updates)
    db.execute(f"UPDATE users SET {clauses} WHERE id=?", [*updates.values(), user["id"]])
    db.commit()
    return {"message": "Settings updated", "fields": list(updates.keys())}

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": classifier.model_name if classifier else "not loaded",
        "claude_available": enricher.available if enricher else False,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
