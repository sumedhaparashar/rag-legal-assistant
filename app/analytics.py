"""
Lightweight analytics tracking backed by SQLite.

Tracks:
  - Questions asked (with user, response time, timestamp)
  - User activity (logins, queries)
  - Derived metrics: DAU, total users, popular topics, etc.

The SQLite database is stored at data/analytics.db and is
initialised automatically on first import.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.config import ANALYTICS_DB_PATH

logger = logging.getLogger("rag_legal_assistant.analytics")

# ── Server start timestamp (for uptime calculation) ──────────────
_server_start_time = time.time()


# ── Database initialisation ──────────────────────────────────────

def _init_db() -> None:
    """Create analytics tables if they don't exist."""
    ANALYTICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(str(ANALYTICS_DB_PATH)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS questions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         TEXT    NOT NULL DEFAULT 'anonymous',
                    question        TEXT    NOT NULL,
                    response_time_ms REAL   NOT NULL DEFAULT 0,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS user_activity (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL,
                    action      TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_q_created
                    ON questions(created_at);
                CREATE INDEX IF NOT EXISTS idx_q_user
                    ON questions(user_id);
                CREATE INDEX IF NOT EXISTS idx_ua_created
                    ON user_activity(created_at);
                CREATE INDEX IF NOT EXISTS idx_ua_user
                    ON user_activity(user_id);
            """)
        logger.info("Analytics DB initialised at %s", ANALYTICS_DB_PATH)
    except Exception as exc:
        logger.warning("Could not initialise analytics DB: %s", exc)


_init_db()


# ── Connection helper ────────────────────────────────────────────

@contextmanager
def _get_db():
    """Yield a SQLite connection with Row factory, auto-commit."""
    conn = sqlite3.connect(str(ANALYTICS_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Tracking functions ───────────────────────────────────────────

def track_question(
    user_id: str,
    question: str,
    response_time_ms: float,
) -> None:
    """Record a question event."""
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO questions (user_id, question, response_time_ms) "
                "VALUES (?, ?, ?)",
                (user_id, question, round(response_time_ms, 1)),
            )
    except Exception as exc:
        logger.warning("Failed to track question: %s", exc)


def track_activity(user_id: str, action: str) -> None:
    """Record a user activity event (login, query, etc.)."""
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO user_activity (user_id, action) VALUES (?, ?)",
                (user_id, action),
            )
    except Exception as exc:
        logger.warning("Failed to track activity: %s", exc)


# ── Metrics retrieval ────────────────────────────────────────────

def get_analytics() -> Dict[str, Any]:
    """Return aggregated analytics metrics."""
    try:
        with _get_db() as conn:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            week_ago = (
                datetime.now(timezone.utc) - timedelta(days=7)
            ).strftime("%Y-%m-%d")

            total_users = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM user_activity"
            ).fetchone()[0]

            dau = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM user_activity "
                "WHERE date(created_at) = ?",
                (today,),
            ).fetchone()[0]

            total_questions = conn.execute(
                "SELECT COUNT(*) FROM questions"
            ).fetchone()[0]

            weekly_questions = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE date(created_at) >= ?",
                (week_ago,),
            ).fetchone()[0]

            avg_rt = conn.execute(
                "SELECT AVG(response_time_ms) FROM questions"
            ).fetchone()[0] or 0.0

            # Daily counts for last 7 days
            daily_counts: List[Dict] = []
            for i in range(6, -1, -1):
                day = (
                    datetime.now(timezone.utc) - timedelta(days=i)
                ).strftime("%Y-%m-%d")
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM questions WHERE date(created_at) = ?",
                    (day,),
                ).fetchone()[0]
                daily_counts.append({"date": day, "count": cnt})

            topics = _extract_topics(conn)

            uptime = time.time() - _server_start_time

            return {
                "total_users": total_users,
                "daily_active_users": dau,
                "total_questions": total_questions,
                "weekly_questions": weekly_questions,
                "avg_response_time_ms": round(avg_rt, 1),
                "top_topics": topics,
                "uptime_seconds": round(uptime),
                "daily_question_counts": daily_counts,
            }
    except Exception as exc:
        logger.error("Failed to retrieve analytics: %s", exc)
        return {
            "total_users": 0,
            "daily_active_users": 0,
            "total_questions": 0,
            "weekly_questions": 0,
            "avg_response_time_ms": 0,
            "top_topics": [],
            "uptime_seconds": round(time.time() - _server_start_time),
            "daily_question_counts": [],
        }


# ── Topic extraction ─────────────────────────────────────────────

_LEGAL_TOPICS: Dict[str, List[str]] = {
    "Directors & Board": [
        "director", "board", "chairman", "managing director", "independent director",
    ],
    "Shareholders & Equity": [
        "shareholder", "share", "dividend", "voting right", "equity",
    ],
    "Company Formation": [
        "registration", "incorporation", "memorandum", "articles of association",
    ],
    "Compliance & Filing": [
        "compliance", "filing", "annual return", "audit", "statutory",
    ],
    "Contracts & Agreements": [
        "contract", "agreement", "liability", "indemnity", "obligation",
    ],
    "Intellectual Property": [
        "patent", "copyright", "trademark", "intellectual property", "ip right",
    ],
    "Banking & Finance": [
        "banking", "loan", "interest", "deposit", "rbi", "bank",
    ],
    "Securities & SEBI": [
        "sebi", "securities", "stock", "listing", "capital market",
    ],
    "Criminal & Penal": [
        "penalty", "offence", "imprisonment", "fine", "ipc", "criminal",
    ],
    "Competition Law": [
        "competition", "monopoly", "merger", "anti-competitive", "cci",
    ],
    "Foreign Investment": [
        "foreign", "fdi", "fcra", "foreign contribution", "investment",
    ],
    "Corporate Governance": [
        "governance", "transparency", "disclosure", "resolution", "csr",
    ],
}


def _extract_topics(conn) -> List[Dict[str, Any]]:
    """Extract top legal topics from stored questions using keyword matching."""
    rows = conn.execute("SELECT question FROM questions").fetchall()
    counts: Dict[str, int] = {t: 0 for t in _LEGAL_TOPICS}

    for row in rows:
        q_lower = row[0].lower()
        for topic, keywords in _LEGAL_TOPICS.items():
            if any(kw in q_lower for kw in keywords):
                counts[topic] += 1

    sorted_topics = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"topic": t, "count": c} for t, c in sorted_topics if c > 0][:8]
