import os
import json
import pandas as pd
from dotenv import load_dotenv

# Load env to get API keys
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def get_llm_response(prompt: str) -> str:
    # 1. Try OpenRouter (if available)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            import requests
            print("Trying OpenRouter...")
            res = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                },
                json={
                    "model": "google/gemini-2.5-flash",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"OpenRouter failed: {e}. Falling back to Groq...")

    # 2. Fallback to Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            print("Trying Groq...")
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=groq_key)
            res = llm.invoke(prompt)
            return res.content
        except Exception as e:
            print(f"Groq failed: {e}")
            
    raise Exception("Both OpenRouter and Groq failed, or keys are missing.")

def main():
    local_path = os.path.join(os.path.dirname(__file__), "ksp_cleaned_prototype.pkl")
    fallback_path = os.path.join(os.path.dirname(__file__), "..", "ksp_prototype_data", "ksp_cleaned_prototype.pkl")
    data_path = local_path if os.path.exists(local_path) else fallback_path
    
    print(f"Loading dataframe from {data_path}...")
    df = pd.read_pickle(data_path)
    
    # Get unique crime headers
    unique_crimes = df['CrimeHead_Name'].dropna().unique().tolist()
    unique_crimes_str = json.dumps(unique_crimes, indent=2)
    print(f"Found {len(unique_crimes)} unique CrimeHead_Names.")
    
    prompt = f"""You are a police data ontology expert. I have an array of exact 'CrimeHead_Name' values from an FIR database. 
I need you to categorize ALL of them into a standard set of 15-20 semantic buckets (like "theft_robbery", "assault", "gambling", "narcotics", "fraud", "traffic", "sexual_crimes", "property_damage", "miscellaneous", etc).

Here is the raw list:
{unique_crimes_str}

Return a RAW valid JSON object where keys are the standardized semantic buckets, and values are arrays containing the exact strings from my list that belong in that bucket.
Every single string from my list MUST appear in exactly one bucket. Do not alter the strings.
DO NOT wrap the response in markdown code blocks. Just return the raw JSON object string.
"""

    print("Sending prompt to LLM to generate glossary...")
    res_text = get_llm_response(prompt)
    
    # Clean up response if the model returned markdown
    res_text = res_text.strip()
    if res_text.startswith("```json"):
        res_text = res_text[7:]
    if res_text.startswith("```"):
        res_text = res_text[3:]
    if res_text.endswith("```"):
        res_text = res_text[:-3]
    res_text = res_text.strip()
    
    try:
        glossary = json.loads(res_text)
        
        out_path = os.path.join(os.path.dirname(__file__), "crime_glossary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(glossary, f, indent=4)
        print(f"Success! Glossary saved to {out_path} with {len(glossary)} categories.")
        
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM response.")
        print(e)
        print("Raw response:")
        print(res_text)

if __name__ == "__main__":
    main()
