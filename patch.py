import os

with open('api_investigation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add DELETE endpoint
if '@router.delete("/api/investigation/{inv_id}")' not in content:
    delete_endpoint = '''
@router.delete("/api/investigation/{inv_id}")
def delete_investigation(inv_id: str, current_user: dict = Depends(get_current_user)):
    role = current_user.get("role", "")
    if role in ["Field Officer", "Crime Analyst"]:
        raise HTTPException(status_code=403, detail="Your role does not have permission to delete cases.")
        
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT id FROM investigations WHERE id = ?", (inv_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Investigation not found")
    
    conn.execute("DELETE FROM investigations WHERE id = ?", (inv_id,))
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Investigation deleted"}
'''
    content += delete_endpoint

# 2. Add PDF export endpoints
if '@router.get("/api/investigation/{inv_id}/draft-diary")' not in content:
    pdf_endpoints = '''
class DraftData(BaseModel):
    part_a: dict
    part_b: list

@router.get("/api/investigation/{inv_id}/draft-diary")
def generate_draft_diary(inv_id: str, current_user: dict = Depends(get_current_user)):
    conn = get_inv_db()
    c = conn.cursor()
    c.execute("SELECT notes FROM investigations WHERE id = ?", (inv_id,))
    row = c.fetchone()
    notes = row["notes"] if row and row["notes"] else ""
    conn.close()
    
    import sqlite3
    try:
        c_conn = sqlite3.connect('chat.db')
        c_conn.row_factory = sqlite3.Row
        cur = c_conn.cursor()
        cur.execute("SELECT memory_json FROM chat_sessions WHERE session_id = ?", (inv_id,))
        chat_row = cur.fetchone()
        c_conn.close()
    except:
        chat_row = None
        
    chat_history = ""
    if chat_row:
        import json
        try:
            mem = json.loads(chat_row["memory_json"])
            chat_history = str(mem.get("history", []))
        except:
            pass

    from state import app_state
    llm = app_state.get("chat_llm")
    if not llm:
        from langchain_groq import ChatGroq
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)
        
    if not llm:
        return {
            "part_a": {
                "Officer Name": current_user["username"],
                "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "Case Summary": notes or "No notes provided."
            },
            "part_b": [{"time_place": "Pending", "details": "Pending investigation details"}]
        }
        
    prompt = f"""
    You are an expert police officer. Create a Case Diary Draft based on the following notes and chat history.
    Notes: {notes}
    Chat History: {chat_history}
    
    Output strictly valid JSON with this format:
    {{
        "part_a": {{
            "Officer Name": "{current_user['username']}",
            "Date": "{datetime.datetime.now().strftime('%Y-%m-%d')}",
            "Case Summary": "Brief summary here"
        }},
        "part_b": [
            {{"time_place": "10:00 AM, Scene", "details": "Found evidence..."}}
        ]
    }}
    """
    try:
        response = llm.invoke(prompt)
        import json
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        return json.loads(content.strip())
    except Exception as e:
        return {
            "part_a": {
                "Officer Name": current_user["username"],
                "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "Case Summary": notes or "No notes provided."
            },
            "part_b": [
                {"time_place": "Pending", "details": "Pending investigation details"}
            ]
        }

from fastapi.responses import Response

@router.post("/api/investigation/{inv_id}/export-diary-pdf")
def export_diary_pdf(inv_id: str, data: DraftData, current_user: dict = Depends(get_current_user)):
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CASE DIARY (Draft)", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="PART A: Administrative Details", ln=1)
    pdf.set_font("Arial", size=12)
    for k, v in data.part_a.items():
        pdf.multi_cell(0, 10, txt=f"{k}: {v}")
        
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="PART B: Case Timeline & Findings", ln=1)
    pdf.set_font("Arial", size=12)
    
    for item in data.part_b:
        time_place = item.get("time_place", "")
        details = item.get("details", "")
        pdf.set_font("Arial", 'B', 11)
        pdf.multi_cell(0, 8, txt=f"Time/Place: {time_place}")
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, txt=f"Details: {details}")
        pdf.ln(2)
        
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    
    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=CaseDiary_{inv_id[:8]}.pdf"
    })
'''
    content += pdf_endpoints

# 3. Restrict create_investigation
content = content.replace(
    "def create_investigation(req: InvestigationCreate, current_user: dict = Depends(get_current_user)):\n    conn = get_inv_db()",
    "def create_investigation(req: InvestigationCreate, current_user: dict = Depends(get_current_user)):\n    if current_user.get('role') == 'Crime Analyst':\n        raise HTTPException(status_code=403, detail='Crime Analysts cannot create investigations.')\n    conn = get_inv_db()"
)

# 4. Restrict chat_investigation
content = content.replace(
    "def chat_investigation(inv_id: str, req: InvestigationChatRequest, current_user: dict = Depends(get_current_user)):\n    from state import app_state",
    "def chat_investigation(inv_id: str, req: InvestigationChatRequest, current_user: dict = Depends(get_current_user)):\n    if current_user.get('role') == 'Crime Analyst':\n        raise HTTPException(status_code=403, detail='Crime Analysts do not have access to the Investigation Chat AI.')\n    from state import app_state"
)

with open('api_investigation.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched api_investigation.py")
