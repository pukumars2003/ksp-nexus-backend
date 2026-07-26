from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import httpx
import os
import json
import base64
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

@router.post("/api/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    doc_type: str = Form("FIR"),
    language: str = Form("en")
):
    keys_to_try = [os.environ.get("OPENROUTER_API_KEY", ""), os.environ.get("OPENROUTER_API_KEY_2", "")]
    keys_to_try = [k for k in keys_to_try if k]
    
    if not keys_to_try:
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
        {"type": "text", "text": "You are a strict Police Intelligence AI. First, analyze the document images to determine if they are relevant to law enforcement, investigations, policing, crime, evidence, identity verification, CCTV footage, or legal proceedings. If the document is clearly irrelevant (e.g., a resume, marketing flyer, random webpage, or technical security vulnerability report), you MUST output ONLY the exact string: 'REJECTED_DOCUMENT_UNAUTHORIZED' and nothing else. If it is a valid investigative or police document, extract all text from the provided document images exactly as it appears, preserving formatting. Do not output anything else but the extracted text in this case."}
    ]
    for b64 in b64_images:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": content_list
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        resp = None
        for attempt_key in keys_to_try:
            headers["Authorization"] = f"Bearer {attempt_key}"
            resp = await client.post(url, headers=headers, json=payload, timeout=90.0)
            if resp.status_code == 200:
                break
                
        if resp is None or resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OCR API Error: {resp.text if resp else 'Unknown'}")
        
    extracted_text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    
    if "REJECTED_DOCUMENT_UNAUTHORIZED" in extracted_text:
        raise HTTPException(status_code=400, detail="Security Policy Violation: Uploaded document is not relevant to police investigations (e.g., resume, vulnerability report, unrelated page).")

    source = "gpt-4o-mini"

    # 3. Multi-LLM Parallel Auditor Workflow
    debate_text = ""
    try:
        from state import app_state
        llm1 = app_state.get("analyze_llm") # OpenRouter
        llm2 = app_state.get("chat_llm")    # Groq Llama 3.3
        
        if llm1 and llm2 and extracted_text.strip():
            lang_instruction = f" Respond completely in the language corresponding to language code '{language}' (e.g. 'hi' for Hindi, 'kn' for Kannada)." if language != 'en' else ""
            
            # Independent prompts so they can run in parallel
            prompt1 = f"You are Investigator 1. Extract only the Key Entities, Suspects, and Evidence from this raw OCR.{lang_instruction}\n\nRAW OCR:\n{extracted_text}"
            prompt2 = f"You are Investigator 2. Extract only the Timeline of Events and Modus Operandi from this raw OCR.{lang_instruction}\n\nRAW OCR:\n{extracted_text}"
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Use Groq (llm2) for BOTH tasks because OpenRouter (llm1) adds 10+ seconds of latency!
                future1 = executor.submit(llm2.invoke, prompt1)
                future2 = executor.submit(llm2.invoke, prompt2)
                
                res1 = future1.result(timeout=15)
                res2 = future2.result(timeout=15)
                
            summary1 = res1.content
            audit2 = res2.content
            
            debate_text = f"**Investigator 1 (Entities & Evidence):**\n{summary1}\n\n**Investigator 2 (Timeline & MO):**\n{audit2}"
    except Exception as e:
        print(f"Error in parallel LLM workflow: {e}")

    return {
        "success": True, 
        "text": extracted_text, 
        "source": source,
        "debate": debate_text
    }
