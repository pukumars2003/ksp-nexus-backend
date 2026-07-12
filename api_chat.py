import os
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
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
    return WhisperModel("base", device="cpu", compute_type="int8")

class ChatRequest(BaseModel):
    text: str = ""
    language: str = "en"
    use_semantic: bool = True

@router.post("/api/chat")
async def chat_endpoint(
    text: str = Form(default=""),
    language: str = Form(default="en"),
    use_semantic: bool = Form(default=True),
    district: str = Form(default="All"),
    crime_type: str = Form(default="All"),
    audio: Optional[UploadFile] = File(default=None),
    user: dict = Depends(get_current_user)
):
    from main import app_state
    
    llm = app_state.get("chat_llm")
    if not llm:
        import os
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)

    df = app_state.get("df")
    if df is None or df.empty:
        import pandas as pd
        import os
        local_path = os.path.join(os.path.dirname(__file__), "ksp_cleaned_prototype.pkl")
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "ksp_prototype_data", "ksp_cleaned_prototype.pkl")
        data_path = local_path if os.path.exists(local_path) else fallback_path
        try:
            df = pd.read_pickle(data_path)
            app_state["df"] = df
        except:
            pass
    
    if df is not None and user["jurisdiction"] != "All":
        df = df[df["District_Name"] == user["jurisdiction"]]
    
    if not llm or df is None:
        raise HTTPException(status_code=500, detail="Models or database not loaded")

    user_query = text
    detected_lang = language

    # 1. Handle Audio Input (STT)
    if audio:
        try:
            whisper_model = get_whisper()
            temp_wav = f"temp_{uuid.uuid4()}.wav"
            with open(temp_wav, "wb") as f:
                f.write(await audio.read())
                
            segments, info = whisper_model.transcribe(temp_wav, language=None)
            transcription = " ".join([s.text for s in segments]).strip()
            
            if text and "Analyze cases from" in text:
                user_query = f"{text} {transcription}"
            else:
                user_query = transcription
                
            detected_lang = info.language
            os.remove(temp_wav)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")

    if not user_query:
        raise HTTPException(status_code=400, detail="No text or audio provided")

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
            final_query_en = GoogleTranslator(source=detected_lang, target="en").translate(user_query)
        except Exception as e:
            print(f"Translation failed: {e}")


    # --- 1.5 MEMORY & INTENT ---
    username = user.get("username", "Officer")
    chat_memory = app_state.get("chat_memory", {})
    if username not in chat_memory:
        chat_memory[username] = {"history": [], "last_cases": ""}
    mem = chat_memory[username]

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

    # 2. Semantic Search Context Retrieval
    context_str = "No specific historical cases retrieved."
    if intent == "INVESTIGATE" and mem.get("last_cases"):
        context_str = mem["last_cases"]
    elif use_semantic and not df.empty:

        try:
            context_cases = []
            
            # 1. LLM Entity Extraction for Natural Language Filtering
            import json
            extraction_prompt = f"""Extract filter parameters from this query if present. Return ONLY a valid JSON object with keys: 'district', 'unit', 'year', 'crime_type'. Use null if not present.
Note: Bellary is 'Ballari', Bangalore is 'Bengaluru', Mangalore is 'Mangaluru', Mysore is 'Mysuru'.
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
                    print(f"Filter extraction failed: {e}")
            
            mask = pd.Series(True, index=df.index)
            has_filters = False
            
            if filters.get('district'):
                dist_val = str(filters['district']).lower()
                mask &= (df['District_Name'].str.lower().str.contains(dist_val, na=False) | df['UnitName'].str.lower().str.contains(dist_val, na=False))
                has_filters = True
            if filters.get('unit'):
                unit_val = str(filters['unit']).lower()
                unit_words = [w for w in unit_val.split() if w not in ['ps', 'police', 'station', 'the', 'in', 'at'] and len(w) > 2]
                if unit_words:
                    word_mask = pd.Series(False, index=df.index)
                    for word in unit_words:
                        word_mask |= df['District_Name'].str.lower().str.contains(word, na=False) | df['UnitName'].str.lower().str.contains(word, na=False)
                    mask &= word_mask
                has_filters = True
            if filters.get('year'):
                mask &= (df['FIR_YEAR'].astype(str) == str(filters['year']))
                has_filters = True
            if filters.get('crime_type'):
                mask &= (df['CrimeHead_Name'].str.lower().str.contains(str(filters['crime_type']).lower(), na=False))
                has_filters = True
            
            if has_filters:
                unit_cases = df[mask]
                for _, r in unit_cases.head(5).iterrows():
                    context_cases.append(f"- Filter Match in {r.get('UnitName')} ({r.get('FIR_YEAR')}): {r.get('CrimeHead_Name')} - {str(r.get('Description', ''))[:200]}. Victims: {r.get('Male', 0)} M, {r.get('Female', 0)} F. IO: {r.get('IOName', 'Unknown')} (KGID: {r.get('KGID', 'Unknown')})")
            
            # 2. Semantic Search Context (Fallback if no filters or no cases found)
            if not context_cases:
                import sys
                
                # Removed sys.path hack
                
                import mo_linking
                df_index = mo_linking.load_mo_index()
                
                if df_index is not None:
                    top_cases = mo_linking.find_similar_mo(final_query_en, df_index, top_k=5)
                    for _, r in top_cases.iterrows():
                        context_cases.append(f"- Similar Case in {r.get('Place of Offence', '').split(',')[0]} ({r.get('FIR_YEAR')}): {r.get('CrimeHead_Name')} - {str(r.get('Description', ''))[:200]}. Victims: {r.get('Male', 0)} Male, {r.get('Female', 0)} Female, {r.get('Boy', 0)} Boy, {r.get('Girl', 0)} Girl. Accused Count: {r.get('Accused Count', 0)}. IO: {r.get('IOName', 'Unknown')} (KGID: {r.get('KGID', 'Unknown')})")
            
            if context_cases:
                context_str = "Context of relevant historical cases:\n" + "\n".join(context_cases)
            else:
                context_str = "No specific historical cases retrieved."
        except Exception as e:
            print(f"Semantic search failed: {e}")
        
        if intent == "SEARCH":
            mem["last_cases"] = context_str


    # 3. LLM Response Generation
    try:
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in mem["history"][-4:]])
        sys_prompt = f"""You are a professional KSP Crime Analyst. Answer the user's query based ONLY on the provided historical context and chat history.
CRITICAL RULES:
1. Only present a MAXIMUM of 5 cases at a time. If more exist, tell the user to ask for specific case details to investigate further.
2. If the user wants to investigate a case, perform deep Entity Extraction but MUST strictly categorize roles: [Known Accused], [Suspects], [Victims], [Witness], [Investigating Officer]. Do NOT confuse IOs with suspects.
3. REASONING SAFETY: NEVER assume or state that cases are connected simply because they share the same Investigating Officer or Police Station. Only connect cases based on hard evidence (Same MO, Same Suspect, Same Phone, Same Vehicle).
4. If comparing multiple cases, always use a strict plain-text Markdown table framework (Factors: Crime, Location, MO, Suspect) to avoid false conclusions.
5. DO NOT alter the capitalization or remove details from the IO Names (e.g., keep 'Y S HANUMANTHAPPA (PI)' exactly as provided).
6. Do NOT use markdown bolding (**text**) or headers (# text). Keep the text clean.
7. Respond in language '{detected_lang}'.

CHAT HISTORY:
{history_str}

HISTORICAL CONTEXT:
{context_str}

Query: {user_query}"""
        response = llm.invoke(sys_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        mem["history"].append({"role": "User", "content": final_query_en})
        mem["history"].append({"role": "AI", "content": response_text})
        
        # 3.5 Translate response back to user's language if necessary
        if detected_lang != "en":
            try:
                from deep_translator import GoogleTranslator
                response_text = GoogleTranslator(source="en", target=detected_lang).translate(response_text)
            except Exception as e:
                print(f"Output translation failed: {e}")
        
        # 4. Generate TTS Audio Response
        audio_url = None
        try:
            import edge_tts, asyncio
            voice_map = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural", "kn": "kn-IN-GaganNeural"}
            voice = voice_map.get(detected_lang, "en-IN-NeerjaNeural")
            
            out_mp3 = f"static/response_{uuid.uuid4()}.mp3"
            os.makedirs("static", exist_ok=True)
            
            async def speak():
                communicate = edge_tts.Communicate(response_text, voice)
                await communicate.save(out_mp3)
                
            await speak()
            audio_url = f"/{out_mp3}"
        except Exception as e:
            print(f"TTS failed: {e}")

        return {
            "response": response_text,
            "detected_query": user_query,
            "detected_lang": detected_lang,
            "audio_url": audio_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
