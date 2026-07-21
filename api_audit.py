from fastapi import APIRouter, HTTPException, Depends
import sqlite3
import datetime
from api_auth import get_current_user

router = APIRouter()

def get_audit_db():
    conn = sqlite3.connect('audit.db')
    conn.row_factory = sqlite3.Row
    conn.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            action TEXT,
            target_id TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    return conn

def log_audit(username: str, role: str, action: str, target_id: str = ""):
    try:
        conn = get_audit_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO audit_logs (username, role, action, target_id, timestamp) VALUES (?, ?, ?, ?, ?)",
            (username, role, action, target_id, datetime.datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to write audit log: {e}")

@router.get("/api/admin/audit")
def get_audit_logs(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") not in ["Administrator", "DSP"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    conn = get_audit_db()
    c = conn.cursor()
    c.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 500")
    logs = [dict(r) for r in c.fetchall()]
    conn.close()
    return logs
