from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
from contextlib import asynccontextmanager
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

from parser import parse_fir

# Global State for models and data
app_state = {}

from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Chat LLM (Groq)...")
    app_state["chat_llm"] = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.0,
        api_key=os.environ.get("GROQ_API_KEY", "")
    )
    
    print("Loading Analyze LLM (OpenRouter)...")
    app_state["analyze_llm"] = ChatOpenAI(
        model="openrouter/free", 
        temperature=0.0,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1"
    )
    
    print("Loading KSP Prototype Data...")
    try:
        local_path = os.path.join(os.path.dirname(__file__), "ksp_cleaned_prototype.pkl")
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "ksp_prototype_data", "ksp_cleaned_prototype.pkl")
        data_path = local_path if os.path.exists(local_path) else fallback_path
        app_state["df"] = pd.read_pickle(data_path)
        print(f"Loaded {len(app_state['df'])} records from PKL.")
    except Exception as e:
        print(f"Warning: Could not load data. {e}")
        app_state["df"] = pd.DataFrame()

    app_state["chat_memory"] = {}

    yield
    # Cleanup
    app_state.clear()


from fastapi.staticfiles import StaticFiles

app = FastAPI(title="KSP Nexus API", lifespan=lifespan)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORSMiddleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from api_analytics import router as analytics_router
from api_chat import router as chat_router
from api_auth import router as auth_router
from api_investigation import router as inv_router
from api_ocr import router as ocr_router
from api_admin import router as admin_router
from api_audit import router as audit_router
from api_predictive import router as predictive_router

app.include_router(auth_router)
app.include_router(analytics_router)
app.include_router(chat_router)
app.include_router(inv_router)
app.include_router(ocr_router)
app.include_router(audit_router)
app.include_router(predictive_router)
app.include_router(admin_router)
app.include_router(audit_router)

class FIRRequest(BaseModel):
    text: str
    language: str = "en"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "KSP Nexus Backend is running"}

@app.post("/api/analyze-fir")
async def analyze_fir(req: FIRRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty FIR text provided.")
    
    # Use chat_llm (Groq) instead of analyze_llm (OpenRouter) to prevent Catalyst 30s timeouts
    llm = app_state.get("chat_llm")
    df = app_state.get("df")
    
    if not llm:
        raise HTTPException(status_code=500, detail="Analyze LLM not initialized")
    try:
        # Pass the dataframe to parser to perform semantic search
        result = parse_fir(llm, req.text, df, language=req.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
