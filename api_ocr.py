from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import httpx
import os
import json
from typing import Dict, Any

router = APIRouter()

# Read the QuickML Webhook URL from environment variables
QUICKML_OCR_URL = os.environ.get("QUICKML_OCR_URL", "")

@router.post("/api/ocr")
async def process_ocr(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or Images.")

    # Read the file content
    content = await file.read()
    
    # Check if QuickML is configured
    if not QUICKML_OCR_URL:
        print("Warning: QUICKML_OCR_URL is not set. Using a mock response for now.")
        # In the absence of the real QuickML endpoint, we mock an OCR extraction
        # that mimics what QuickML would return, specifically for Datathon purposes.
        mock_text = f"Simulated OCR extraction from {file.filename}. Found entities: Suspect John Doe, Phone 9876543210."
        return {"success": True, "text": mock_text, "source": "mock_quickml"}
    
    # Real QuickML integration
    try:
        async with httpx.AsyncClient() as client:
            # We send the file to QuickML via a multipart form data request
            files = {"file": (file.filename, content, file.content_type)}
            response = await client.post(QUICKML_OCR_URL, files=files, timeout=30.0)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"QuickML Error: {response.text}")
            
            data = response.json()
            # Assuming QuickML returns a JSON with a 'text' or 'output' field
            extracted_text = data.get("text") or data.get("output") or json.dumps(data)
            
            return {"success": True, "text": extracted_text, "source": "catalyst_quickml"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to QuickML: {str(e)}")
