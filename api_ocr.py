from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import httpx
import os
import json
import base64
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

@router.post("/api/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    doc_type: str = Form("FIR"),
    language: str = Form("en")
):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key is missing from environment.")

    filename = file.filename.lower()
    if not filename.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or Images.")

    content = await file.read()
    b64_images = []
    
    if filename.endswith('.pdf'):
        try:
            import fitz
            pdf = fitz.open(stream=content, filetype="pdf")
            for i in range(min(3, len(pdf))): # Limit to 3 pages for cost/context limits
                page = pdf[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("jpeg")
                b64_images.append(base64.b64encode(img_data).decode('utf-8'))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF parsing error: {str(e)}")
    else:
        b64_images.append(base64.b64encode(content).decode('utf-8'))

    content_list = [
        {"type": "text", "text": "Extract all text from the provided document images exactly as it appears. Ensure you preserve formatting. Do not output anything else but the extracted text."}
    ]
    for b64 in b64_images:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "max_tokens": 4000,
        "messages": [
            {
                "role": "user",
                "content": content_list
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload, timeout=90.0)
        
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OCR API Error: {resp.text}")
        
    extracted_text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    source = "qwen2_vl_openrouter"

    # 3. Multi-LLM Auditor Workflow
    debate_text = ""
    try:
        from state import app_state
        llm1 = app_state.get("analyze_llm") # OpenRouter
        llm2 = app_state.get("chat_llm")    # Groq Llama 3.3
        
        if llm1 and llm2 and extracted_text.strip():
            lang_instruction = f" Respond completely in the language corresponding to language code '{language}' (e.g. 'hi' for Hindi, 'kn' for Kannada)." if language != 'en' else ""
            
            # Investigator 1: Summarizes
            prompt1 = f"You are Investigator 1. Read the following raw OCR extracted from a {doc_type}. Summarize the key facts, entities, and events mentioned. Keep it structured and concise.{lang_instruction}\n\nRAW OCR:\n{extracted_text}"
            res1 = llm1.invoke(prompt1)
            summary1 = res1.content
            
            # Investigator 2: Audits
            prompt2 = f"You are Investigator 2 (The Auditor). Your job is to fact-check Investigator 1's summary against the raw OCR text. Look for missing details, hallucinations, or contradictions. Be critical and thorough.{lang_instruction}\n\nDOCUMENT TYPE: {doc_type}\n\nRAW OCR:\n{extracted_text}\n\nINVESTIGATOR 1 SUMMARY:\n{summary1}\n\nProvide your audit report."
            res2 = llm2.invoke(prompt2)
            audit2 = res2.content
            
            debate_text = f"**Investigator 1 (Summary):**\n{summary1}\n\n**Investigator 2 (Audit):**\n{audit2}"
    except Exception as e:
        print(f"Error in LLM audit workflow: {e}")

    return {
        "success": True, 
        "text": extracted_text, 
        "source": source,
        "debate": debate_text
    }
