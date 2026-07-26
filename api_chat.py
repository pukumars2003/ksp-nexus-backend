import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any, Optional
from functools import lru_cache
from fastapi.responses import FileResponse
from api_auth import get_current_user

router = APIRouter()

@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@lru_cache(maxsize=1)
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8")

class ChatRequest(BaseModel):
    text: str = ""
    language: str = "en"
    use_semantic: bool = True
    session_id: str = ""


@router.get("/api/chat/sessions")
async def get_chat_sessions(user: dict = Depends(get_current_user)):
    import sqlite3, os, json
    db_path = os.path.join(os.path.dirname(__file__), 'chat.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    title TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    memory_json TEXT
                  )''')
    username = user.get("username", "Officer")
    cursor = conn.execute("SELECT session_id, title, updated_at FROM chat_sessions WHERE username = ? ORDER BY updated_at DESC", (username,))
    sessions = [{"id": row["session_id"], "title": row["title"], "updated_at": row["updated_at"]} for row in cursor]
    conn.close()
    return {"sessions": sessions}

@router.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, user: dict = Depends(get_current_user)):
    import sqlite3, os
    db_path = os.path.join(os.path.dirname(__file__), 'chat.db')
    conn = sqlite3.connect(db_path)
    username = user.get("username", "Officer")
    conn.execute("DELETE FROM chat_sessions WHERE session_id = ? AND username = ?", (session_id, username))
    conn.commit()
    conn.close()
    from state import app_state
    if "chat_sessions" in app_state and session_id in app_state["chat_sessions"]:
        del app_state["chat_sessions"][session_id]
    return {"status": "success"}

@router.put("/api/chat/sessions/{session_id}")
async def rename_chat_session(session_id: str, request: dict, user: dict = Depends(get_current_user)):
    import sqlite3, os
    db_path = os.path.join(os.path.dirname(__file__), 'chat.db')
    conn = sqlite3.connect(db_path)
    username = user.get("username", "Officer")
    new_title = request.get("title", "Renamed Chat")
    conn.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ? AND username = ?", (new_title, session_id, username))
    conn.commit()
    conn.close()
    return {"status": "success"}

@router.get("/api/chat/history")

async def get_chat_history(session_id: str = None, user: dict = Depends(get_current_user)):
    if not session_id:
        return {"history": []}
        
    from state import app_state
    import sqlite3, json, os
    
    db_path = os.path.join(os.path.dirname(__file__), 'chat.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    title TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    memory_json TEXT
                  )''')
                  
    chat_sessions = app_state.get("chat_sessions", {})
    if session_id not in chat_sessions:
        cursor = conn.execute("SELECT memory_json FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            chat_sessions[session_id] = json.loads(row['memory_json'])
        else:
            chat_sessions[session_id] = {"history": [], "last_cases": ""}
    
    app_state["chat_sessions"] = chat_sessions
    conn.close()
    
    return {"history": chat_sessions[session_id].get("history", [])}


@router.post("/api/chat")
async def chat_interaction(
    request: Request,
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    language: str = Form("en"),
    session_id: Optional[str] = Form(None),
    use_semantic: Optional[bool] = Form(True),
    tts_provider: str = Form("Edge"),
    district: str = Form("All"),
    crime_type: str = Form("All"),
    current_user: dict = Depends(get_current_user)
):
    if current_user.get("role") == "Crime Analyst":
        raise HTTPException(status_code=403, detail="Crime Analysts do not have access to the Investigation Chat AI.")
        
    username = current_user["username"]
    from state import app_state
    
    llm = app_state.get("chat_llm")
    if not llm:
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            llm = ChatOpenAI(model="openrouter/free", temperature=0.0, api_key=api_key, base_url="https://openrouter.ai/api/v1")

    df = app_state.get("df")
    glossary = app_state.get("crime_glossary")
    
    if df is None or df.empty:
        local_path = os.path.join(os.path.dirname(__file__), "ksp_cleaned_prototype.pkl")
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "ksp_prototype_data", "ksp_cleaned_prototype.pkl")
        data_path = local_path if os.path.exists(local_path) else fallback_path
        try:
            df = pd.read_pickle(data_path)
            app_state["df"] = df
        except:
            pass
            
    if not glossary:
        glossary_path = os.path.join(os.path.dirname(__file__), "crime_glossary.json")
        try:
            import json
            with open(glossary_path, "r", encoding="utf-8") as f:
                glossary = json.load(f)
            app_state["crime_glossary"] = glossary
        except Exception:
            glossary = {}
    
    if df is not None and current_user["jurisdiction"] != "All":
        df = df[df["District_Name"] == current_user["jurisdiction"]]
    
    if not llm or df is None:
        raise HTTPException(status_code=500, detail="Models or database not loaded")

    user_query = text
    detected_lang = language
    temp_wav = None

    # 1. Handle Audio Input (STT)
    if audio:
        try:
            temp_wav = f"temp_{uuid.uuid4()}.webm"
            with open(temp_wav, "wb") as f:
                f.write(await audio.read())
            
            api_key = os.environ.get("GROQ_API_KEY", "")
            if api_key:
                import requests
                headers = {"Authorization": f"Bearer {api_key}"}
                with open(temp_wav, "rb") as f:
                    files = {"file": (temp_wav, f)}
                    data = {"model": "whisper-large-v3", "response_format": "verbose_json"}
                    res = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, files=files, data=data)
                    
                if res.status_code == 200:
                    res_json = res.json()
                    transcription = res_json.get("text", "").strip()
                    whisper_lang = res_json.get("language", "")
                    if whisper_lang == "kannada": detected_lang = "kn"
                    elif whisper_lang == "hindi": detected_lang = "hi"
                    elif whisper_lang == "english": detected_lang = "en"
                else:
                    transcription = ""
            else:

                whisper_model = get_whisper()
                segments, info = whisper_model.transcribe(
                    temp_wav, 
                    language=None, 
                    beam_size=1,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6
                )
                transcription = " ".join([s.text for s in segments]).strip()
            
            if text and "Analyze cases from" in text:
                user_query = f"{text} {transcription}"
            else:
                user_query = transcription
                
            os.remove(temp_wav)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")


    if not user_query:
        raise HTTPException(status_code=400, detail="No text or audio provided")

    # Trust frontend language preference over Whisper detection
    if language and language != "en":
        detected_lang = language
    elif not detected_lang:
        detected_lang = "en"

    # 1b. Translation (Semantic Search must run in English)
    import re
    if re.search(r'[\u0C80-\u0CFF]', user_query):
        detected_lang = "kn"
    elif re.search(r'[\u0900-\u097F]', user_query):
        detected_lang = "hi"
        
    final_query_en = user_query
    if detected_lang != "en":
        try:
            from deep_translator import GoogleTranslator
            import concurrent.futures
            
            def do_translation():
                return GoogleTranslator(source='auto', target="en").translate(user_query)
                
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(do_translation)
            # GoogleTranslator can get rate-limited and hang, so enforce a strict 3-second timeout
            final_query_en = future.result(timeout=3.0)
            # Do NOT shutdown(wait=True) because it will block if the thread is hanging
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            print(f"Translation failed or timed out: {e}")
            final_query_en = user_query
            # If it timed out, executor is still running the thread in the background, which is fine
    
    # --- 1.5 MEMORY & INTENT ---
    username = current_user.get("username", "Officer")
    
    import sqlite3, json
    
    if not session_id:
        session_id = str(uuid.uuid4())
        
    db_path = os.path.join(os.path.dirname(__file__), 'chat.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    title TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    memory_json TEXT
                  )''')
                  
    chat_sessions = app_state.get("chat_sessions", {})
    if session_id not in chat_sessions:
        cursor = conn.execute("SELECT memory_json FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            chat_sessions[session_id] = json.loads(row['memory_json'])
        else:
            chat_sessions[session_id] = {"history": [], "last_cases": ""}
            
    app_state["chat_sessions"] = chat_sessions
    mem = chat_sessions[session_id]
    
    def save_memory(new_title=None):
        try:
            curr_title = new_title or "New Chat"
            if not new_title:
                c = conn.execute("SELECT title FROM chat_sessions WHERE session_id = ?", (session_id,))
                r = c.fetchone()
                if r and r['title']: curr_title = r['title']
            
            conn.execute('''INSERT INTO chat_sessions (session_id, username, title, updated_at, memory_json) 
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                            ON CONFLICT(session_id) DO UPDATE SET 
                            updated_at=CURRENT_TIMESTAMP, memory_json=excluded.memory_json, title=excluded.title''', 
                         (session_id, username, curr_title, json.dumps(chat_sessions[session_id], ensure_ascii=False)))
            conn.commit()
        except Exception as e:
            print(f"Error saving chat DB: {e}")


    intent = "SEARCH"
    if mem["history"]:
        intent_prompt = f"Given this query: '{final_query_en}', classify if the user wants to SEARCH for new cases/data, or INVESTIGATE/explain cases already discussed in memory. Reply ONLY with SEARCH or INVESTIGATE."
        try:
            intent_res = llm.invoke(intent_prompt)
            intent = str(intent_res.content).strip().upper()
            if "INVESTIGATE" in intent: intent = "INVESTIGATE"
            else: intent = "SEARCH"
        except:
            pass

    intent = "SEARCH" # Forced for reliability
    
    # 2. Semantic Search Context Retrieval
    context_str = "No specific historical cases retrieved."
    
    if use_semantic and not df.empty:

        try:
            context_cases = []
            sources_list = []
            
            # 1. LLM Entity Extraction for Natural Language Filtering
            import json
            
            # Dynamically fetch valid districts to prevent hardcoding
            valid_districts = df['District_Name'].dropna().unique().tolist() if 'District_Name' in df.columns else []
            valid_districts_str = ", ".join(valid_districts) if valid_districts else "any valid district"
            
            unique_crimes = df['CrimeHead_Name'].dropna().unique().tolist() if 'CrimeHead_Name' in df.columns else []
            unique_crimes_str = ", ".join(unique_crimes)
            
            extraction_prompt = f"""Extract filter parameters from this query if present. Return ONLY a valid JSON object with keys: 'district', 'unit', 'year', 'crime_types'. Use null if not present.
Note: If a district is mentioned, map it to the closest match from this official list: {valid_districts_str}.
Note: For 'crime_types', map the requested crime to a JSON array of the most relevant EXACT strings from this official list: [{unique_crimes_str}]. Pick only the ones that strongly match the user's intent.
Query: {final_query_en}"""
            
            extraction_res = llm.invoke(extraction_prompt)
            extraction_text = extraction_res.content if hasattr(extraction_res, "content") else str(extraction_res)
            
            import re
            json_match = re.search(r'\{.*\}', extraction_text, re.DOTALL)
            filters = {}
            if json_match:
                try:
                    filters = json.loads(json_match.group(0))
                except Exception as e:
                    filters = {}
            
            with open("debug_chat.txt", "w", encoding="utf-8") as f:
                f.write(f"Query: {final_query_en}\n")
                f.write(f"Extraction Text: {extraction_text}\n")
                f.write(f"Parsed Filters: {filters}\n")
            
            mask = pd.Series(True, index=df.index)
            has_filters = False
            fallback_triggered = False
            
            if filters.get('district'):
                dist_val = str(filters['district']).lower()
                
                # Handle common alternative names the LLM might output
                if "bangalore" in dist_val: dist_val = "bengaluru"
                elif "mangalore" in dist_val: dist_val = "mangaluru"
                elif "mysore" in dist_val: dist_val = "mysuru"
                elif "hubli" in dist_val: dist_val = "hubballi"
                
                dist_words = [w for w in dist_val.split() if len(w) > 2]
                if dist_words:
                    word_mask = pd.Series(False, index=df.index)
                    for word in dist_words:
                        word_mask |= df['District_Name'].astype(str).str.lower().str.contains(word, na=False) | df['UnitName'].astype(str).str.lower().str.contains(word, na=False)
                    mask &= word_mask
                has_filters = True
            if filters.get('unit'):
                unit_val = str(filters['unit']).lower()
                unit_words = [w for w in unit_val.split() if w not in ['ps', 'police', 'station', 'the', 'in', 'at'] and len(w) > 2]
                if unit_words:
                    word_mask = pd.Series(False, index=df.index)
                    for word in unit_words:
                        word_mask |= df['District_Name'].astype(str).str.lower().str.contains(word, na=False) | df['UnitName'].astype(str).str.lower().str.contains(word, na=False)
                    mask &= word_mask
                has_filters = True

            if filters.get('year'):
                mask &= (df['FIR_YEAR'].astype(str).str.contains(str(filters['year']), na=False))
                has_filters = True
            if filters.get('crime_types') and isinstance(filters['crime_types'], list):
                type_mask = pd.Series(False, index=df.index)
                for ct in filters['crime_types']:
                    type_mask |= (df['CrimeHead_Name'] == ct)
                mask &= type_mask
                has_filters = True
            
            if has_filters:
                unit_cases = df[mask]
                # Fallback: if 0 cases found due to misspelled unit (e.g. Gadhinagar), drop the unit mask and try again
                if len(unit_cases) == 0 and filters.get('unit'):
                    fallback_triggered = True
                    # Re-calculate mask without unit
                    mask2 = pd.Series(True, index=df.index)
                    if filters.get('district'):
                        dist_val = str(filters['district']).lower()
                        if "bangalore" in dist_val: dist_val = "bengaluru"
                        elif "mangalore" in dist_val: dist_val = "mangaluru"
                        elif "mysore" in dist_val: dist_val = "mysuru"
                        elif "hubli" in dist_val: dist_val = "hubballi"
                        dist_words = [w for w in dist_val.split() if len(w) > 2]
                        if dist_words:
                            word_mask = pd.Series(False, index=df.index)
                            for word in dist_words:
                                word_mask |= df['District_Name'].astype(str).str.lower().str.contains(word, na=False) | df['UnitName'].astype(str).str.lower().str.contains(word, na=False)
                            mask2 &= word_mask
                    if filters.get('year'):
                        mask2 &= (df['FIR_YEAR'].astype(str).str.contains(str(filters['year']), na=False))
                    if filters.get('crime_types') and isinstance(filters['crime_types'], list):
                        type_mask = pd.Series(False, index=df.index)
                        for ct in filters['crime_types']:
                            type_mask |= (df['CrimeHead_Name'] == ct)
                        mask2 &= type_mask
                    unit_cases = df[mask2]
                    
                with open("debug_chat.txt", "a", encoding="utf-8") as f:
                    f.write(f"Has filters: True. Cases found: {len(unit_cases)}\n")
                    
                for _, r in unit_cases.head(5).iterrows():
                    context_cases.append(f"- Filter Match in {r.get('District_Name')} district, {r.get('UnitName')} ({r.get('FIR_YEAR')}): {r.get('CrimeHead_Name')} - {str(r.get('Description', ''))[:200]}. Victims: {r.get('Male', 0)} M, {r.get('Female', 0)} F. IO: {r.get('IOName', 'Unknown')} (KGID: {r.get('KGID', 'Unknown')})")
                    sources_list.append({
                        "fir_id": r.get('FIRNo', 'Unknown'),
                        "district": r.get('District_Name', 'Unknown'),
                        "year": r.get('FIR_YEAR', 'Unknown'),
                        "crime_type": r.get('CrimeHead_Name', 'Unknown')
                    })
            
            # 2. Semantic Search Context (Fallback if no filters or no cases found)
            if not context_cases:
                import sys
                
                # Removed sys.path hack
                
                import mo_linking
                df_index = mo_linking.load_mo_index()
                
                if df_index is not None:
                    top_cases = mo_linking.find_similar_mo(final_query_en, df_index, top_k=5)
                    for _, r in top_cases.iterrows():
                        context_cases.append(f"- Similar Case in {r.get('District_Name')} district, {r.get('Place of Offence', '').split(',')[0]} ({r.get('FIR_YEAR')}): {r.get('CrimeHead_Name')} - {str(r.get('Description', ''))[:200]}. Victims: {r.get('Male', 0)} Male, {r.get('Female', 0)} Female, {r.get('Boy', 0)} Boy, {r.get('Girl', 0)} Girl. Accused Count: {r.get('Accused Count', 0)}. IO: {r.get('IOName', 'Unknown')} (KGID: {r.get('KGID', 'Unknown')})")
                        sources_list.append({
                            "fir_id": r.get('FIRNo', 'Unknown'),
                            "district": r.get('District_Name', 'Unknown'),
                            "year": r.get('FIR_YEAR', 'Unknown'),
                            "crime_type": r.get('CrimeHead_Name', 'Unknown')
                        })
            
            if context_cases:
                context_str = "Context of relevant historical cases:\n" + "\n".join(context_cases)
                if fallback_triggered:
                    context_str += "\n\n[SYSTEM NOTE TO AI]: We could not find the specific police station the user asked for. We fetched cases from the broader district instead. You MUST inform the user that their specific station wasn't found, and then summarize these broader district cases for them instead of refusing to answer."
            else:
                context_str = "No specific historical cases retrieved."
        except Exception as e:
            with open("error_log.txt", "w", encoding="utf-8") as f:
                import traceback
                f.write(traceback.format_exc())
            print(f"Semantic search failed: {e}")
            sources_list = []
        
        if intent == "SEARCH":
            mem["last_cases"] = context_str
            mem["last_sources"] = sources_list
    else:
        sources_list = mem.get("last_sources", [])


    # 3. LLM Response Generation
    try:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in mem["history"][-4:]])
        sys_prompt = f"""You are a professional KSP Crime Analyst. Answer the user's query based ONLY on the provided historical context and chat history.
CRITICAL RULES:
1. ONLY USE THE CASES PROVIDED IN 'HISTORICAL CONTEXT'.
2. If the 'HISTORICAL CONTEXT' is completely empty ("No specific historical cases retrieved."), output exactly: "I don't have information on that in the current database." HOWEVER, if cases ARE provided in the context but they don't perfectly match the specific street, market, or sub-location the user asked for, YOU MUST STILL LIST AND SUMMARIZE the provided cases. Just add a polite note explaining that these are the closest available cases from that district/area. Do NOT refuse to answer if cases are provided.
3. NEVER make up FIR numbers, IO names, or case details.
4. YOU MUST ALWAYS LIST ALL THE CASES provided in the 'HISTORICAL CONTEXT' to the user. Do not just summarize them abstractly. Present them clearly using a strict plain-text Markdown table framework (Factors: Crime, Location, MO, Suspect) or bullet points so the user can see exactly what cases were found.
5. DO NOT alter the capitalization or remove details from the IO Names (e.g., keep 'Y S HANUMANTHAPPA (PI)' exactly as provided).
6. Do NOT use markdown bolding (**text**) or headers (# text). Keep the text clean.
7. IMPORTANT: You MUST translate your entire answer into the detected language ({detected_lang}). If it is 'kn', write fluently in Kannada script. If it is 'hi', write fluently in Hindi script. DO NOT transliterate technical police terms or system words like 'Filter Match', 'IO', 'PS', 'FIR', 'KGID' - keep those exact words in English script. Keep ALL JSON keys strictly in English.
8. If the user asks questions completely unrelated to policing, investigations, crime, or the provided context (e.g., asking to write a resume, coding, or explaining software vulnerabilities), put exactly "Security Policy Violation: I am a Police Intelligence Copilot and can only assist with investigative and law enforcement matters." in the 'response' JSON field.
9. YOU MUST OUTPUT STRICTLY IN JSON FORMAT matching this schema:
{{
  "response": "Your actual answer to the user translated into {detected_lang}.",
  "confidence_score": 95, 
  "citations": "Citations in English."
}}

CHAT HISTORY:
{history_str}

HISTORICAL CONTEXT:
{context_str}

Query: {final_query_en}"""
        # Check for FIR Document Analysis
        fir_data = None
        is_fir = "I have uploaded a FIR document" in user_query or "FIRST INFORMATION REPORT" in user_query.upper() or "F.I.R" in user_query.upper()
        if is_fir:
            try:
                from parser import parse_fir
                fir_data = parse_fir(llm, user_query, df, language=detected_lang)
                if fir_data:
                    display_msg = "📄 [Uploaded FIR Document for Analysis]" if len(user_query) > 500 else final_query_en
                    mem["history"].append({"role": "User", "content": display_msg})
                    mem["history"].append({"role": "AI", "content": "I have successfully analyzed the FIR document. Here is the structured breakdown:", "fir_data": fir_data})
                    response_text = "I have successfully analyzed the FIR document. Here is the structured breakdown:"
            except Exception as e:
                print(f"Error parsing FIR: {e}")

        if not fir_data:
            response = llm.invoke(sys_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Since we removed the JSON requirement, the LLM output is purely the response text.
            response_text = content.strip()
            
            # Clean up any accidental markdown blocks if it still thinks it's coding
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                if len(lines) > 2:
                    response_text = '\n'.join(lines[1:-1]).strip()
                    
            confidence_score = 95
            citations = "Historical Context"
                
            mem["history"].append({"role": "User", "content": final_query_en})
            mem["history"].append({"role": "AI", "content": response_text, "confidence_score": confidence_score, "citations": citations})
        
        # Auto-title generation for new chats
        new_title = None
        if len(mem["history"]) <= 2:
            try:
                title_prompt = f"Generate a short 3-5 word title for this chat based on the user's first query. Do NOT use quotes. Query: {final_query_en}"
                res_title = llm.invoke(title_prompt)
                new_title = str(res_title.content).strip().strip('"')
            except:
                pass
                
        save_memory(new_title)

        
        # 4. Generate TTS Audio Response
        audio_url = None
        if temp_wav:  # Only generate audio response if the user sent an audio request
            try:
                out_mp3 = f"static/response_{uuid.uuid4()}.mp3"
                os.makedirs("static", exist_ok=True)
                
                if tts_provider == "Zoho Catalyst":
                    import requests
                    
                    speaker = "Mary"
                    z_lang = detected_lang
                    if detected_lang == "hi": speaker = "Divya"
                    elif detected_lang == "kn": speaker = "Anu"
                    elif detected_lang not in ["en", "hi", "kn"]:
                        z_lang = "en"
                        
                    payload = {
                        "text": response_text[:1000],
                        "language": z_lang,
                        "speaker": speaker,
                        "pitch": "moderate",
                        "speed": "moderate",
                        "emotion": "neutral"
                    }
                    
                    zoho_token = os.environ.get("ZOHO_CATALYST_TOKEN", "dummy_token")
                    headers = {
                        "CATALYST-ORG": "60077218671", 
                        "Authorization": f"Zoho-oauthtoken {zoho_token}" 
                    }
                    
                    try:
                        res = requests.post("https://api.catalyst.zoho.in/quickml/api/v1/models/zia/tts/synthesize", json=payload, headers=headers, timeout=5)
                        if res.status_code == 200:
                            out_wav = out_mp3.replace(".mp3", ".wav")
                            with open(out_wav, "wb") as f:
                                f.write(res.content)
                            audio_url = f"/{out_wav}"
                    except Exception as e:
                        print(f"Zoho TTS Failed, falling back: {e}")
                        
                if not audio_url:
                    from gtts import gTTS
                    
                    z_lang = detected_lang
                    if z_lang not in ['en', 'hi', 'kn']:
                        z_lang = 'en'
                    
                    try:
                        tts = gTTS(text=response_text[:1000], lang=z_lang, slow=False)
                        tts.save(out_mp3)
                        audio_url = f"/{out_mp3}"
                    except Exception as e:
                        print(f"gTTS failed: {e}")
            except Exception as e:
                print(f"TTS failed: {e}")

        # Multi-API Multi-threaded Audio Verification Debug
        audio_debug = []


        return {
            "response": response_text,
            "fir_data": locals().get("fir_data"),
            "confidence_score": locals().get("confidence_score", 0),
            "citations": locals().get("citations", ""),
            "detected_query": user_query,
            "detected_lang": detected_lang,
            "audio_url": audio_url,
            "session_id": session_id,
            "sources": sources_list,
            "audio_debug": audio_debug
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
