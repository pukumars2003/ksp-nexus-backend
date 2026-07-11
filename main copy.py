from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
from contextlib import asynccontextmanager
from langchain_ollama import ChatOllama
import os

from parser import parse_fir

# Global State for models and data
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading LLM (qwen2.5-coder:1.5b)...")
    app_state["llm"] = ChatOllama(model="qwen2.5-coder:1.5b", temperature=0.0)
    
    print("Loading KSP Prototype Data...")
    try:
        data_path = os.path.join(os.path.dirname(__file__), "..", "ksp_prototype_data", "ksp_cleaned_prototype.pkl")
        app_state["df"] = pd.read_pickle(data_path)
        print(f"Loaded {len(app_state['df'])} records from PKL.")
    except Exception as e:
        print(f"Warning: Could not load data. {e}")
        app_state["df"] = pd.DataFrame()

    yield
    # Cleanup
    app_state.clear()


from fastapi.staticfiles import StaticFiles

app = FastAPI(title="KSP Nexus API", lifespan=lifespan)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api_analytics import router as analytics_router
from api_chat import router as chat_router

app.include_router(analytics_router)
app.include_router(chat_router)

class FIRRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "KSP Nexus Backend is running"}

@app.post("/api/analyze-fir")
async def analyze_fir(req: FIRRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty FIR text provided.")
    
    llm = app_state.get("llm")
    df = app_state.get("df")
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")
    try:
        # Pass the dataframe to parser to perform semantic search
        result = parse_fir(llm, req.text, df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
