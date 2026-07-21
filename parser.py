import json
import re
import pandas as pd
from functools import lru_cache
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_fir(text: str) -> dict:
    metadata_match = re.search(r'(.*?)(?:5\.\s*Place of Occurrence|6\.\s*Complainant)', text, re.DOTALL | re.IGNORECASE)
    parties_match = re.search(r'(6\.\s*Complainant.*?)(?:8\.\s*Particulars|9\.\s*Inquest|10\.\s*F\.I\.R|12\.\s*First information)', text, re.DOTALL | re.IGNORECASE)
    narrative_match = re.search(r'(?:10\.\s*F\.I\.R\s*Contents|12\.\s*First information contents)(.*)', text, re.DOTALL | re.IGNORECASE)
    
    metadata = metadata_match.group(1).strip() if metadata_match else "N/A"
    parties = parties_match.group(1).strip() if parties_match else "N/A"
    narrative = narrative_match.group(1).strip() if narrative_match else text
    
    is_kannada = bool(re.search(r'[\u0C80-\u0CFF]', narrative))
    
    return {
        "metadata_chunk": metadata,
        "parties_chunk": parties,
        "narrative_chunk": narrative,
        "is_kannada": is_kannada
    }

def parse_fir(llm, new_fir_text: str, df: pd.DataFrame = None, language: str = "en") -> dict:
    if len(new_fir_text.strip()) < 50:
        return {
            "Metadata": { "Crime_No": "N/A", "Police_Station": "N/A", "District": "N/A", "FIR_Date": "N/A", "IO_Name": "N/A" },
            "Crime_Type": "Invalid Input",
            "Complainant": "N/A",
            "Accused": "N/A",
            "IPC_Sections": [],
            "Property_Summary": { "Items": [], "Total_Value": "N/A" },
            "Timeline": [],
            "MO_Summary": "The provided text is too short to be a valid FIR narrative.",
            "Behavior_Profile": {"Risk_Level": "Unknown", "Financial_Motive": "Unknown", "Violence": "Unknown", "Group_Size": 0},
            "Recommendations": "Please provide a full, detailed FIR narrative for analysis.",
            "Similar_Cases": []
        }

    chunks = preprocess_fir(new_fir_text)
    lang_instruction = f"Output all string values in the language corresponding to language code '{language}' (e.g. 'hi' for Hindi, 'kn' for Kannada) but keep the JSON keys in English." if language != "en" else "Output in English."
    
    extract_prompt = ChatPromptTemplate.from_template("""
You are an expert criminal intelligence analyst. Parse the FIR chunks into structured JSON format. 

--- FIR METADATA CHUNK ---
{metadata_chunk}

--- FIR PARTIES CHUNK ---
{parties_chunk}

--- FIR NARRATIVE CHUNK ---
{narrative_chunk}

CRITICAL RULES:
1. ONLY extract information that is explicitly stated in the chunks. DO NOT hallucinate.
2. MISSING VALUES: If a value is missing, output "N/A".
3. CRIME TYPE: Be specific. If it is a house theft, classify as 'House Theft' or 'Residential Burglary' (ಮನೆ ಕಳ್ಳತನ), not just 'Theft'. Differentiate Theft (ಕಳ್ಳತನ) from Robbery (ದರೋಡೆ).
4. PARTIES & ROLES: Include their designations/roles.
5. INVESTIGATING OFFICER: Look carefully at the end of the FIR for "Name: " and their rank (e.g., 'SURESH.W - PSI').
6. ACTS & SECTIONS: Extract exactly what is written from the METADATA CHUNK.
7. FINANCIAL & PROPERTY DETAILS: Extract property weights/values EXACTLY as written in the text (e.g., '550 gms', do NOT truncate to '55g'). Do not mistranslate OCR artifacts (e.g., translate gold rings as 'ಬಂಗಾರದ ಉಂಗುರ', avoid nonsensical translations like 'ಬಿರಿಯಾಣಿ').
8. TIMELINE: Extract all specific events if present. Distinguish between 'Incident Time', 'FIR Registered/Received', 'Read Over', and 'Dispatch to Court' with their respective exact times.
9. BEHAVIOR PROFILE: Note that this is an analytical derived profile.
10. AI_SUMMARY: Generate a concise 5-7 bullet point analytical summary detailing: Offence, Estimated Loss, Victim, Suspects, Likely Motive, Primary Targets, and Recommended Investigation Steps.
11. {lang_instruction} Output strictly valid JSON matching the schema below.

Required JSON Schema:
{{
  "Metadata": {{
    "Crime_No": "string",
    "Police_Station": "string",
    "District": "string",
    "FIR_Date": "string",
    "IO_Name": "string"
  }},
  "Crime_Type": "string",
  "Complainant": "string",
  "Accused": "string",
  "IPC_Sections": ["string"],
  "Property_Summary": {{
    "Items": [{{ "name": "string", "quantity": "string", "value": "string" }}],
    "Total_Value": "string"
  }},
  "Timeline": [{{ "time": "string", "event": "string" }}],
  "MO_Summary": "string",
  "Behavior_Profile": {{
    "Risk_Level": "string (High/Medium/Low)",
    "Financial_Motive": "string (High/Medium/Low)",
    "Violence": "string (Yes/No)",
    "Group_Size": 0
  }},
  "AI_Summary": ["string (bullet point 1)", "string (bullet point 2)"]
}}
""")
    extract_chain = extract_prompt | llm | StrOutputParser()
    raw_output = extract_chain.invoke({
        "metadata_chunk": chunks["metadata_chunk"],
        "parties_chunk": chunks["parties_chunk"],
        "narrative_chunk": chunks["narrative_chunk"],
        "is_kannada": str(chunks["is_kannada"]),
        "lang_instruction": lang_instruction
    })
    
    parsed_data = {}
    try:
        import re
        
        # 1. Extract just the JSON block using regex if there's surrounding text
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            cleaned_output = json_match.group(0)
        else:
            cleaned_output = raw_output
            
        # 2. Strip C-style comments (e.g. // Assuming the IPC section)
        cleaned_output = re.sub(r'//.*', '', cleaned_output)
        
        # 3. Strip trailing commas
        cleaned_output = re.sub(r',\s*([}\]])', r'\1', cleaned_output)
        
        raw_parsed = json.loads(cleaned_output)
        
        # 4. Normalize keys to match exact schema (from Streamlit app.py)
        for k, v in raw_parsed.items():
            normalized_k = k.title().replace(" ", "_")
            if "Mo" in normalized_k: normalized_k = normalized_k.replace("Mo", "MO")
            if "Ipc" in normalized_k: normalized_k = normalized_k.replace("Ipc", "IPC")
            parsed_data[normalized_k] = v
            
        if 'Behavior_Profile' in parsed_data and isinstance(parsed_data['Behavior_Profile'], dict):
            bp = parsed_data['Behavior_Profile']
            parsed_data['Behavior_Profile'] = {bk.title().replace(" ", "_"): bv for bk, bv in bp.items()}
            
    except Exception as e:
        print(f"JSON PARSING ERROR: {e}")
        print(f"RAW OUTPUT WAS:\n{raw_output}")
        parsed_data = {
            "Metadata": { "Crime_No": "N/A", "Police_Station": "N/A", "District": "N/A", "FIR_Date": "N/A", "IO_Name": "N/A" },
            "Crime_Type": "Unknown (Parsing Error)",
            "Complainant": "N/A",
            "Accused": "N/A",
            "IPC_Sections": [],
            "Property_Summary": { "Items": [], "Total_Value": "N/A" },
            "Timeline": [],
            "MO_Summary": raw_output[:500] + "...",
            "Behavior_Profile": {"Risk_Level": "Unknown"}
        }
    
    # 1. Generate Recommendations
    rec_prompt = ChatPromptTemplate.from_template("""
Based on this MO and Behavior Profile, provide 3 actionable investigation recommendations for the police officer (e.g., check CCTV, check financial records). Keep them brief.

Crime Type: {crime}
MO: {mo}
Profile: {profile}
""")
    rec_chain = rec_prompt | llm | StrOutputParser()
    recs = rec_chain.invoke({
        "crime": parsed_data.get('Crime_Type', 'N/A'),
        "mo": parsed_data.get('MO_Summary', 'N/A'),
        "profile": str(parsed_data.get('Behavior_Profile', 'N/A'))
    })
    
    parsed_data["Recommendations"] = recs
    parsed_data["Similar_Cases"] = []

    # 2. Semantic Search for similar cases using mo_linking index
    is_valid_crime = parsed_data.get('Crime_Type') not in ['N/A', 'Unknown', 'Unknown (Parsing Error)']
    is_valid_mo = parsed_data.get('MO_Summary') not in ['N/A', 'Unknown']
    
    if df is not None and not df.empty and (is_valid_crime or is_valid_mo):
        try:
            import sys
            import os
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
                
            import mo_linking
            df_index = mo_linking.load_mo_index()
            
            if df_index is not None:
                embedding_string = f"Crime: {parsed_data.get('Crime_Type', '')}. MO: {parsed_data.get('MO_Summary', '')}"
                
                # Use mo_linking's optimized search
                top_cases = mo_linking.find_similar_mo(embedding_string, df_index, top_k=5, min_similarity=0.1)
                
                similar_cases = []
                for _, r in top_cases.iterrows():
                    match_factors = []
                    if str(r.get('CrimeHead_Name')).lower() in str(parsed_data.get('Crime_Type', '')).lower():
                        match_factors.append("Same Crime Type")
                    if r['similarity'] > 0.8:
                        match_factors.append("High MO Semantic Match")
                    elif r['similarity'] > 0.6:
                        match_factors.append("Moderate MO Match")
                    
                    similar_cases.append({
                        "UnitName": str(r.get('Place of Offence', '').split(',')[0]),
                        "CrimeHead": str(r.get('CrimeHead_Name', '')),
                        "Year": str(r.get('FIR_YEAR', '')),
                        "Factors": ", ".join(match_factors) if match_factors else "General semantic similarity",
                        "Score": f"{(r['similarity'] * 100):.1f}%"
                    })
                
                parsed_data["Similar_Cases"] = similar_cases
            else:
                print("mo_embeddings.pkl not found, skipping semantic search.")
            
        except Exception as e:
            print(f"Semantic search failed: {e}")

    return parsed_data
