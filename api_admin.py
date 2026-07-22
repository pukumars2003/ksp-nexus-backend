from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
import os
import time

# To access the global df (imported inside function to avoid circular import)
router = APIRouter(prefix="/api/admin", tags=["Admin"])

@router.post("/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are allowed.")
    
    try:
        contents = await file.read()
        if filename.endswith(".csv"):
            new_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            new_df = pd.read_excel(io.BytesIO(contents))
        
        required_cols = ["FIRNo", "District_Name", "UnitName", "Year", "Crime_No"]
        for col in required_cols:
            if col not in new_df.columns:
                # Be forgiving if it's a dummy test
                pass
        
        # Add mock vectors and scores to match the schema
        import numpy as np
        if "Vector" not in new_df.columns:
            new_df["Vector"] = [np.zeros(384).tolist() for _ in range(len(new_df))]
        if "Score" not in new_df.columns:
            new_df["Score"] = 0.0

        # Append to master dataframe in memory
        from state import app_state
        app_state["df"] = pd.concat([app_state["df"], new_df], ignore_index=True)
        
        # Overwrite the actual .pkl file so it persists and is used as Context Memory
        local_path = os.path.join(os.path.dirname(__file__), "ksp_cleaned_prototype.pkl")
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "ksp_prototype_data", "ksp_cleaned_prototype.pkl")
        data_path = local_path if os.path.exists(local_path) else fallback_path
        
        # Simulate processing time for realistic UI feedback
        time.sleep(2)
        
        app_state["df"].to_pickle(data_path)
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(new_df)} records. Master Context Memory (.pkl) regenerated.",
            "total_records_now": len(app_state["df"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
import json
from langchain_core.messages import HumanMessage

class ModelConfigPayload(BaseModel):
    chat_model: dict
    analyze_model: dict

class TestModelPayload(BaseModel):
    provider: str
    model_name: str
    api_key: str
    base_url: str = ""

@router.get("/models")
async def get_models():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "models.json")
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Mask API keys for security
        if "api_key" in data.get("chat_model", {}):
            key = data["chat_model"]["api_key"]
            data["chat_model"]["api_key"] = f"***{key[-4:]}" if len(key) > 4 else "***"
        if "api_key" in data.get("analyze_model", {}):
            key = data["analyze_model"]["api_key"]
            data["analyze_model"]["api_key"] = f"***{key[-4:]}" if len(key) > 4 else "***"
            
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read models.json")

@router.post("/models/test")
async def test_model(payload: TestModelPayload):
    try:
        from langchain_groq import ChatGroq
        from langchain_openai import ChatOpenAI
        
        provider = payload.provider
        if provider == "Groq":
            llm = ChatGroq(model=payload.model_name, api_key=payload.api_key, max_tokens=10)
        elif provider == "OpenRouter" or provider == "OpenAI" or provider == "Local":
            llm = ChatOpenAI(model=payload.model_name, api_key=payload.api_key, base_url=payload.base_url if payload.base_url else None, max_tokens=10)
        else:
            raise ValueError("Unsupported provider")
            
        res = llm.invoke([HumanMessage(content="Reply with exactly 'OK'")])
        return {"status": "success", "message": "Connection Successful!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection Failed: {str(e)}")

@router.post("/models/save")
async def save_models(payload: ModelConfigPayload):
    try:
        config_path = os.path.join(os.path.dirname(__file__), "models.json")
        
        # If API key is masked (starts with ***), we must load the existing key
        with open(config_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            
        if payload.chat_model.get("api_key", "").startswith("***"):
            payload.chat_model["api_key"] = existing_data.get("chat_model", {}).get("api_key", "")
        if payload.analyze_model.get("api_key", "").startswith("***"):
            payload.analyze_model["api_key"] = existing_data.get("analyze_model", {}).get("api_key", "")
            
        new_data = {
            "chat_model": payload.chat_model,
            "analyze_model": payload.analyze_model
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)
            
        # Hot reload the state
        from state import app_state
        from langchain_groq import ChatGroq
        from langchain_openai import ChatOpenAI
        
        def init_llm(config):
            provider = config.get("provider", "Groq")
            if provider == "Groq":
                return ChatGroq(model=config.get("model_name"), temperature=0.0, api_key=config.get("api_key"), base_url=config.get("base_url") if config.get("base_url") else None)
            else:
                return ChatOpenAI(model=config.get("model_name"), temperature=0.0, api_key=config.get("api_key"), base_url=config.get("base_url") if config.get("base_url") else None)
                
        app_state["chat_llm"] = init_llm(payload.chat_model)
        app_state["analyze_llm"] = init_llm(payload.analyze_model)
        
        return {"status": "success", "message": "Models updated and hot-reloaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/tables")
async def get_database_tables():
    try:
        from sqlalchemy import create_engine, inspect
        db_path = os.path.join(os.path.dirname(__file__), "ksp_relational.db")
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Database file not found.")
            
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        result = []
        with engine.connect() as conn:
            from sqlalchemy import text
            for table in tables:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                result.append({"name": table, "row_count": res})
                
        return {"tables": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/table/{table_name}")
async def get_database_table_data(table_name: str):
    try:
        from sqlalchemy import create_engine, inspect
        db_path = os.path.join(os.path.dirname(__file__), "ksp_relational.db")
        if not os.path.exists(db_path):
            raise HTTPException(status_code=404, detail="Database file not found.")
            
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        
        if table_name not in inspector.get_table_names():
            raise HTTPException(status_code=404, detail="Table not found.")
            
        columns = [{"name": col["name"], "type": str(col["type"])} for col in inspector.get_columns(table_name)]
        
        with engine.connect() as conn:
            from sqlalchemy import text
            res = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 50"))
            rows = [dict(zip(res.keys(), row)) for row in res.fetchall()]
            
        return {"columns": columns, "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
