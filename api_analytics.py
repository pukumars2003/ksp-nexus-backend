from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
from typing import Dict, Any
from pydantic import BaseModel
from api_auth import get_current_user

router = APIRouter()

# Note: We will inject the global df from main.py's app_state
def get_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"error": "Data not loaded"}
        
    # Example aggregations based on app.py logic
    total_firs = len(df)
    
    # Crimes by district
    if "UnitName" in df.columns:
        crimes_by_district = df["UnitName"].value_counts().head(10).to_dict()
    else:
        crimes_by_district = {}
        
    # High risk offenders (assuming AccusedName exists like in app.py)
    high_risk_count = 0
    if "AccusedName" in df.columns:
        # Just an example mock of the logic since the real dataframe has varying column names
        high_risk_count = df["AccusedName"].nunique()
        
    return {
        "total_firs": total_firs,
        "high_risk_offenders": high_risk_count,
        "crimes_by_district": crimes_by_district,
        "recent_trends": [
            {"month": "Jan", "count": 1200},
            {"month": "Feb", "count": 1900},
            {"month": "Mar", "count": 1500},
            {"month": "Apr", "count": 2100}
        ]
    }

@router.get("/api/dashboard/stats")
async def dashboard_stats(user: dict = Depends(get_current_user)):
    # To avoid circular imports, we import app_state here
    from main import app_state
    
    df = app_state.get("df")
    if df is not None and user["jurisdiction"] != "All":
        df = df[df["District_Name"] == user["jurisdiction"]]
    
    if df is None:
        return {
            "total_firs": 0,
            "high_risk_offenders": 0,
            "crimes_by_district": [],
            "recent_trends": []
        }
        
    return get_analytics(df)

@router.get("/api/analytics/risk-offenders")
async def get_risk_offenders(user: dict = Depends(get_current_user)):
    from main import app_state
    import sys
    import os
    
    # Removed sys.path hack since files are colocated
        
    try:
        import risk_scoring
    except ImportError:
        raise HTTPException(status_code=500, detail="risk_scoring module not found in root")
        
    df = app_state.get("df")
    if df is None:
        return []
        
    if user["jurisdiction"] != "All":
        df = df[df["District_Name"] == user["jurisdiction"]]
        
    try:
        # Get top 15 risk offenders using the real algorithm
        risk_df = risk_scoring.get_top_risk_offenders(df, top_n=15)
        # Convert NaN to None for JSON serialization
        risk_df = risk_df.where(pd.notnull(risk_df), None)
        return risk_df.to_dict(orient="records")
    except Exception as e:
        print(f"Risk Scoring Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AnalyticsFilter(BaseModel):
    district: str = "All"
    crime_type: str = "All"
    unit: str = "All"

@router.post("/api/analytics/filtered")
async def get_filtered_analytics(filters: AnalyticsFilter, user: dict = Depends(get_current_user)):
    from main import app_state
    df = app_state.get("df")
    
    if df is not None and user["jurisdiction"] != "All":
        df = df[df["District_Name"] == user["jurisdiction"]]
    if df is None or df.empty or 'CrimeHead_Name' not in df.columns:
        return {
            "total": 0,
            "bar_data": [],
            "pie_data": [],
            "hotspot_data": [],
            "temporal_data": [],
            "demographic_data": [],
            "available_filters": {"districts": ["All"], "crimes": ["All"], "units": ["All"]}
        }
        
    mask = pd.Series(True, index=df.index)
    if filters.district != "All":
        mask &= (df['District_Name'] == filters.district)
    if filters.crime_type != "All":
        mask &= (df['CrimeHead_Name'] == filters.crime_type)
    if filters.unit != "All":
        mask &= (df['UnitName'] == filters.unit)
        
    plot_df = df[mask]
    
    # 1. Bar Chart (Crimes by Type)
    crime_counts = plot_df['CrimeHead_Name'].value_counts().head(10).reset_index()
    crime_counts.columns = ['name', 'count']
    bar_data = crime_counts.to_dict(orient="records")
    
    # 2. Pie Chart (Crimes by District)
    district_counts = plot_df['District_Name'].value_counts().head(10).reset_index()
    district_counts.columns = ['name', 'count']
    pie_data = district_counts.to_dict(orient="records")
    
    # 3. Hotspot Map
    hotspot_data = []
    if 'Latitude' in plot_df.columns and 'Longitude' in plot_df.columns:
        # Group by UnitName
        unit_group = plot_df.groupby('UnitName').size().reset_index(name='count')
        # Get coordinates for units
        coords = df.dropna(subset=['Latitude', 'Longitude']).groupby('UnitName')[['Latitude', 'Longitude']].mean().reset_index()
        unit_group = unit_group.merge(coords, on='UnitName', how='left')
        
        # Fill missing with default center (Hubli)
        unit_group['Latitude'] = unit_group['Latitude'].fillna(15.3173)
        unit_group['Longitude'] = unit_group['Longitude'].fillna(75.7139)
        
        hotspot_data = unit_group.to_dict(orient="records")
        
    # 4. Temporal Analytics
    temporal_data = []
    if 'FIR_YEAR' in plot_df.columns and 'FIR_MONTH' in plot_df.columns:
        temp_group = plot_df.groupby(['FIR_YEAR', 'FIR_MONTH']).size().reset_index(name='count')
        temp_group = temp_group.sort_values(['FIR_YEAR', 'FIR_MONTH'])
        # Limit to reasonable chart length
        temp_group = temp_group.tail(24)
        temp_group['period'] = temp_group['FIR_YEAR'].astype(str) + "-" + temp_group['FIR_MONTH'].astype(str).str.zfill(2)
        temporal_data = temp_group[['period', 'count']].to_dict(orient="records")

    # 5. Demographic Analytics
    demographic_data = []
    demo_cols = ['Male', 'Female', 'Boy', 'Girl']
    for col in demo_cols:
        if col in plot_df.columns:
            count = pd.to_numeric(plot_df[col], errors='coerce').sum()
            if count > 0:
                demographic_data.append({"name": col, "count": int(count)})
        
    # Filters available
    available_districts = ["All"] + sorted([str(x) for x in df['District_Name'].dropna().unique()])
    available_crimes = ["All"] + sorted([str(x) for x in df['CrimeHead_Name'].dropna().unique()])
    available_units = ["All"] + sorted([str(x) for x in df['UnitName'].dropna().unique()])

    return {
        "bar_data": bar_data,
        "pie_data": pie_data,
        "hotspot_data": hotspot_data,
        "temporal_data": temporal_data,
        "demographic_data": demographic_data,
        "total": len(plot_df),
        "available_filters": {
            "districts": available_districts,
            "crimes": available_crimes,
            "units": available_units
        }
    }
