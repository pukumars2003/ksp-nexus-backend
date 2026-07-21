from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import sqlite3
import datetime
import uuid
from typing import List, Optional
from api_auth import get_current_user
import json
import logging

router = APIRouter()

def get_inv_db():
    conn = sqlite3.connect('investigations.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS investigations (
            id TEXT PRIMARY KEY,
            username TEXT,
            title TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS investigation_bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investigation_id TEXT,
            fir_id TEXT,
            text_snippet TEXT,
            added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS investigation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investigation_id TEXT,
            role TEXT,
            content TEXT,
            confidence_score TEXT,
            rationale TEXT,
            citations TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS investigation_cdr_networks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investigation_id TEXT,
            nodes_json TEXT,
            links_json TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    return conn

class InvestigationCreate(BaseModel):
    title: str

class BookmarkAdd(BaseModel):
    fir_id: str
    text_snippet: Optional[str] = None

class MessageAdd(BaseModel):
    role: str
    content: str
    confidence_score: Optional[str] = None
    rationale: Optional[str] = None

class InvestigationChatRequest(BaseModel):
    message: str
    investigation_id: str

@router.post("/api/investigation/create")
def create_investigation(req: InvestigationCreate, current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    inv_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO investigations (id, username, title, created_at) VALUES (?, ?, ?, ?)",
              (inv_id, current_user["username"], req.title, now))
    conn.commit()
    conn.close()
    
    from api_audit import log_audit
    log_audit(current_user["username"], current_user.get("role", "User"), "CREATE_INVESTIGATION", inv_id)
    
    return {"id": inv_id, "title": req.title, "created_at": now}

@router.get("/api/investigation/list")
def list_investigations(current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT * FROM investigations WHERE username = ? ORDER BY created_at DESC", (current_user["username"],))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/api/investigation/{inv_id}")
def get_investigation(inv_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    
    c.execute("SELECT * FROM investigations WHERE id = ? AND username = ?", (inv_id, current_user["username"]))
    inv = c.fetchone()
    if not inv:
        conn.close()
        raise HTTPException(status_code=404, detail="Investigation not found")
        
    c.execute("SELECT * FROM investigation_bookmarks WHERE investigation_id = ?", (inv_id,))
    bookmarks = [dict(r) for r in c.fetchall()]
    
    import json
    c.execute("SELECT * FROM investigation_messages WHERE investigation_id = ? ORDER BY id ASC", (inv_id,))
    messages = []
    for r in c.fetchall():
        msg = dict(r)
        if msg.get('citations'):
            try:
                msg['citations'] = json.loads(msg['citations'])
            except:
                msg['citations'] = []
        messages.append(msg)
    
    conn.close()
    return {
        "investigation": dict(inv),
        "bookmarks": bookmarks,
        "messages": messages
    }

@router.post("/api/investigation/{inv_id}/bookmark")
def add_bookmark(inv_id: str, req: BookmarkAdd, current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT id FROM investigations WHERE id = ? AND username = ?", (inv_id, current_user["username"]))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Investigation not found")
        
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO investigation_bookmarks (investigation_id, fir_id, text_snippet, added_at) VALUES (?, ?, ?, ?)",
              (inv_id, req.fir_id, req.text_snippet, now))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@router.post("/api/investigation/{inv_id}/chat")
def chat_investigation(inv_id: str, req: InvestigationChatRequest, current_user: dict = Depends(get_current_user)):
    from main import app_state
    llm = app_state.get("chat_llm")
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not loaded")
        
    conn = get_inv_db()
    c = conn.cursor()
    
    # Verify owner
    c.execute("SELECT id FROM investigations WHERE id = ? AND username = ?", (inv_id, current_user["username"]))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Investigation not found")
        
    # Get bookmarks
    c.execute("SELECT fir_id, text_snippet FROM investigation_bookmarks WHERE investigation_id = ?", (inv_id,))
    bookmarks = c.fetchall()
    bookmark_text = "\n".join([f"FIR {b['fir_id']}: {b['text_snippet']}" for b in bookmarks]) if bookmarks else "No bookmarked evidence."
    
    # Get history
    c.execute("SELECT role, content FROM investigation_messages WHERE investigation_id = ? ORDER BY id ASC LIMIT 10", (inv_id,))
    history = c.fetchall()
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])
    
    # Build prompt
    sys_prompt = f"""You are a senior KSP Investigation Analyst assisting an officer with an active case.
You are investigating based on the following BOOKMARKED EVIDENCE:
{bookmark_text}

PAST CHAT HISTORY:
{history_text}

USER QUERY: {req.message}

Respond strictly with a JSON object (no markdown code blocks, just raw JSON). Use this format:
{{
  "response": "Your detailed professional response formatted in Markdown (use tables if comparing).",
  "confidence_score": "e.g., 92%",
  "confidence_rationale": "Brief 1-sentence reasoning for why you are confident or not confident based on the evidence.",
  "citations": ["FIR 1234", "Phone Number 9876543210", "Common Vehicle"]
}}
"""
    try:
        res = llm.invoke(sys_prompt)
        content = res.content if hasattr(res, "content") else str(res)
        
        # Clean potential markdown JSON wrapping
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        import json
        data = json.loads(content.strip())
        
        # Save to DB
        now = datetime.datetime.utcnow().isoformat()
        # Save user message
        c.execute("INSERT INTO investigation_messages (investigation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (inv_id, "User", req.message, now))
        # Save AI message
        import json
        c.execute("INSERT INTO investigation_messages (investigation_id, role, content, confidence_score, rationale, citations, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (inv_id, "AI", data.get("response", ""), data.get("confidence_score", ""), data.get("confidence_rationale", ""), json.dumps(data.get("citations", [])), now))
        
        conn.commit()
        conn.close()
        
        return data
    except Exception as e:
        conn.close()
        logging.error(f"Investigation chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/investigation/{inv_id}/network")
def get_network(inv_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT id FROM investigations WHERE id = ? AND username = ?", (inv_id, current_user["username"]))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Investigation not found")
        
    c.execute("SELECT fir_id, text_snippet FROM investigation_bookmarks WHERE investigation_id = ?", (inv_id,))
    bookmarks = c.fetchall()
    conn.close()
    
    if not bookmarks:
        return {"nodes": [], "links": []}
        
    bookmark_text = "\n\n".join([f"FIR {b['fir_id']}:\n{b['text_snippet']}" for b in bookmarks])
    
    from main import app_state
    llm = app_state.get("chat_llm")
    if not llm:
        import os
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return {"nodes": [], "links": [], "error": "LLM not loaded and GROQ_API_KEY not found"}
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)
    
    sys_prompt = f"""You are a Police Intelligence System.
Extract entities (Person, Phone, Location, Vehicle, Organization) from the following evidence and their relationships.
Output STRICTLY raw JSON (no markdown formatting, no ` ```json ` blocks).
Format:
{{
  "nodes": [
    {{"id": "Name or Number", "group": "Person"}},
    {{"id": "9876543210", "group": "Phone"}}
  ],
  "links": [
    {{"source": "Name", "target": "9876543210", "value": 1, "label": "Owns/Calls"}}
  ]
}}

EVIDENCE:
{bookmark_text}
"""
    try:
        res = llm.invoke(sys_prompt)
        content = res.content if hasattr(res, "content") else str(res)
        if content.startswith("```json"): content = content[7:-3]
        elif content.startswith("```"): content = content[3:-3]
        import json
        fir_network = json.loads(content.strip())
    except Exception as e:
        logging.error(f"Network error: {e}")
        fir_network = {"nodes": [], "links": []}
        
    # Merge with CDR Networks
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT nodes_json, links_json FROM investigation_cdr_networks WHERE investigation_id = ?", (inv_id,))
    cdr_rows = c.fetchall()
    conn.close()
    
    merged_nodes = {n.get("id"): n for n in fir_network.get("nodes", []) if n.get("id")}
    merged_links = []
    merged_links.extend(fir_network.get("links", []))
    
    for row in cdr_rows:
        try:
            import json
            c_nodes = json.loads(row["nodes_json"])
            c_links = json.loads(row["links_json"])
            for n in c_nodes:
                if n.get("id") and n.get("id") not in merged_nodes:
                    merged_nodes[n["id"]] = n
            merged_links.extend(c_links)
        except:
            pass
            
    return {
        "nodes": list(merged_nodes.values()),
        "links": merged_links
    }

from fastapi import UploadFile, File

@router.post("/api/investigation/{inv_id}/cdr-analyze")
async def analyze_cdr(inv_id: str, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT id FROM investigations WHERE id = ? AND username = ?", (inv_id, current_user["username"]))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Investigation not found")
        
    try:
        import pandas as pd
        import io
        
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Dynamic Column Identification (Robust)
        cols_lower = [c.lower() for c in df.columns]
        
        caller_name_col = next((c for c in df.columns if 'caller' in c.lower() and 'name' in c.lower()), None)
        caller_num_col = next((c for c in df.columns if 'caller' in c.lower() and 'num' in c.lower()), None)
        recv_name_col = next((c for c in df.columns if 'receiv' in c.lower() and 'name' in c.lower()), None)
        recv_num_col = next((c for c in df.columns if 'receiv' in c.lower() and 'num' in c.lower()), None)
        
        # Fallbacks if columns are just "Phone1", "Phone2"
        if not caller_num_col:
            phone_cols = [c for c in df.columns if 'phone' in c.lower() or 'number' in c.lower() or 'contact' in c.lower()]
            if len(phone_cols) >= 2:
                caller_num_col, recv_num_col = phone_cols[0], phone_cols[1]
                
        dur_col = next((c for c in df.columns if 'dur' in c.lower()), None)
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        time_col = next((c for c in df.columns if 'time' in c.lower() and 'stamp' not in c.lower()), None)
        
        # 1. Build Entity Mapping dictionary (Number -> Name)
        num_to_name = {}
        if caller_name_col and caller_num_col:
            for _, r in df.dropna(subset=[caller_name_col, caller_num_col]).iterrows():
                num_to_name[str(r[caller_num_col])] = str(r[caller_name_col])
        if recv_name_col and recv_num_col:
            for _, r in df.dropna(subset=[recv_name_col, recv_num_col]).iterrows():
                num_to_name[str(r[recv_num_col])] = str(r[recv_name_col])

        # 2. Centrality Analysis (Find the hub)
        stats_summary = ""
        avg_dur = 0
        cdr_nodes = []
        cdr_links = []
        
        if caller_num_col and recv_num_col:
            all_numbers = pd.concat([df[caller_num_col].astype(str), df[recv_num_col].astype(str)])
            hub = all_numbers.value_counts().idxmax()
            hub_count = all_numbers.value_counts().max()
            hub_name = num_to_name.get(str(hub), "Unknown")
            stats_summary += f"Most Connected Person: {hub_name} ({hub}) - {hub_count} total connections.\n"
            
            top_callers = df[caller_num_col].astype(str).value_counts().head(3)
            stats_summary += "Frequent Contacts:\n"
            for n, count in top_callers.items():
                stats_summary += f"- {num_to_name.get(n, n)} ({n}): {count} calls\n"
                
            # Build Graph Data
            unique_nums = all_numbers.unique()
            for num in unique_nums:
                name = num_to_name.get(num, "Unknown")
                group = "Person" if name != "Unknown" else "Phone"
                cdr_nodes.append({"id": name if name != "Unknown" else num, "group": group})
                
            for _, r in df.dropna(subset=[caller_num_col, recv_num_col]).iterrows():
                c_num = str(r[caller_num_col])
                r_num = str(r[recv_num_col])
                c_name = num_to_name.get(c_num, c_num)
                r_name = num_to_name.get(r_num, r_num)
                cdr_links.append({"source": c_name, "target": r_name, "value": 1, "label": "Call"})
                
            # Unknown Number Analysis
            unknown_calls = df[(df[caller_num_col].astype(str).map(lambda x: num_to_name.get(x) is None)) | (df[recv_num_col].astype(str).map(lambda x: num_to_name.get(x) is None))]
            if not unknown_calls.empty:
                stats_summary += "\nUnknown Contact Analysis:\n"
                stats_summary += "Unknown numbers communicated with:\n"
                unknown_connected_names = set()
                for _, r in unknown_calls.iterrows():
                    c_n = str(r[caller_num_col])
                    r_n = str(r[recv_num_col])
                    if num_to_name.get(c_n): unknown_connected_names.add(num_to_name.get(c_n))
                    if num_to_name.get(r_n): unknown_connected_names.add(num_to_name.get(r_n))
                for name in unknown_connected_names:
                    stats_summary += f"- {name}\n"

        if dur_col:
            df[dur_col] = pd.to_numeric(df[dur_col], errors='coerce')
            avg_dur = int(df[dur_col].mean()) if not pd.isna(df[dur_col].mean()) else 0
            stats_summary += f"\nAverage Call Duration: {avg_dur} seconds\n"
            
            max_dur = df[dur_col].max()
            if not pd.isna(max_dur):
                longest = df.loc[df[dur_col].idxmax()]
                c_n = num_to_name.get(str(longest.get(caller_num_col, "")), str(longest.get(caller_num_col, "")))
                r_n = num_to_name.get(str(longest.get(recv_num_col, "")), str(longest.get(recv_num_col, "")))
                stats_summary += f"Longest Communication: {c_n} <-> {r_n} ({max_dur} seconds)\n"

        # Save CDR Graph to DB
        import json
        c.execute("INSERT INTO investigation_cdr_networks (investigation_id, nodes_json, links_json, created_at) VALUES (?, ?, ?, ?)",
                  (inv_id, json.dumps(cdr_nodes), json.dumps(cdr_links), datetime.datetime.utcnow().isoformat()))
        conn.commit()

        # 3. LLM Cross-Reference & Formatting
        c.execute("SELECT text_snippet FROM investigation_bookmarks WHERE investigation_id = ?", (inv_id,))
        bookmarks = "\n".join([b['text_snippet'] for b in c.fetchall()])
        
        from main import app_state
        llm = app_state.get("chat_llm")
        
        if llm:
            cross_prompt = f"""You are a master criminal investigator analyzing a CDR (Call Detail Record).

Here are the extracted statistics from the raw CSV data:
{stats_summary}

Here is the existing bookmarked case evidence (which contains suspect names and MO):
{bookmarks}

Your task is to produce a highly professional, formatted Intelligence Briefing. 
CRITICAL INSTRUCTIONS:
1. DO NOT make definitive legal conclusions. Use cautious phrases like "possible coordination related to activities under investigation" rather than "execution of the fraud".
2. Follow the exact Markdown template below.

Format your EXACT output exactly like this (use Markdown):
### Communication Analysis

**Network Statistics**
Most Connected Person: [Name]
Direct Connections: [X]
Average Call Duration: [X] seconds
Longest Communication: [Name] <-> [Name] ([X] seconds)

**Key Connected Accused**
- [Name 1]
- [Name 2]

**Unknown Contact Analysis**
Unknown numbers communicated with:
1. [Name 1]
2. [Name 2]
Recommendation: Perform subscriber lookup and historical CDR analysis.

**Observation**
[Your 3-4 sentence legally-cautious analytical observation explaining the network, the central hub, and matching it to the pattern found in the evidence.]

**Confidence Ratings**
- CDR Parsing Confidence: High
- Entity Matching Confidence: High
- Investigative Inference Confidence: Medium
"""
            try:
                res = llm.invoke(cross_prompt)
                final_report = res.content if hasattr(res, "content") else str(res)
            except Exception as e:
                final_report = f"### Communication Analysis\n\n{stats_summary}\n\n*Error generating LLM intelligence: {e}*"
        else:
            final_report = f"### Communication Analysis\n\n{stats_summary}"
                
        now = datetime.datetime.utcnow().isoformat()
        c.execute("INSERT INTO investigation_messages (investigation_id, role, content, confidence_score, rationale, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                  (inv_id, "AI", final_report, "High", "Based on deterministic Pandas parsing and LLM Entity Resolution.", now))
        from api_audit import log_audit
        log_audit(current_user["username"], current_user.get("role", "User"), "CDR_ANALYZE", inv_id)
        conn.commit()
        conn.close()
        
        return {"status": "ok"}
    except Exception as e:
        import traceback
        try:
            with open('C:/Users/AjayKumar/.gemini/antigravity-ide/brain/6cf8124b-8bc1-4d15-a2a0-aafb6a166efc/scratch/err.log', 'w') as f:
                f.write(traceback.format_exc())
        except: pass
        if 'conn' in locals(): conn.close()
        raise HTTPException(status_code=500, detail=str(e))
