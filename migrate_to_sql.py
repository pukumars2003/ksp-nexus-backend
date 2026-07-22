import os
import pandas as pd
from sqlalchemy.orm import sessionmaker
from database_schema import (
    create_db, State, District, Unit, CrimeHead, CaseMaster, ComplainantDetails, Victim, Accused
)

def migrate_data():
    print("Connecting to ksp_relational.db...")
    engine = create_db('sqlite:///ksp_relational.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Loading prototype dataset...")
    pkl_path = "ksp_cleaned_prototype.pkl"
    if not os.path.exists(pkl_path):
        pkl_path = "../ksp_prototype_data/ksp_cleaned_prototype.pkl"
    
    if not os.path.exists(pkl_path):
        print("Data file not found. Please ensure ksp_cleaned_prototype.pkl is available.")
        return

    df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(df)} records. Starting migration...")

    # 1. Populate State (assuming all are Karnataka for now)
    karnataka = session.query(State).filter_by(StateName="KARNATAKA").first()
    if not karnataka:
        karnataka = State(StateName="KARNATAKA", NationalityID=1, Active=True)
        session.add(karnataka)
        session.commit()

    # 2. Extract and Populate Districts
    district_map = {}
    unique_districts = df['District_Name'].dropna().unique()
    for d_name in unique_districts:
        district = session.query(District).filter_by(DistrictName=d_name).first()
        if not district:
            district = District(DistrictName=d_name, StateID=karnataka.StateID, Active=True)
            session.add(district)
            session.commit()
        district_map[d_name] = district.DistrictID

    # 3. Extract and Populate Units
    unit_map = {}
    # Unique combination of District and Unit
    unique_units = df[['District_Name', 'UnitName']].drop_duplicates().dropna()
    for _, row in unique_units.iterrows():
        d_name = row['District_Name']
        u_name = row['UnitName']
        unit = session.query(Unit).filter_by(UnitName=u_name).first()
        if not unit:
            unit = Unit(
                UnitName=u_name, 
                StateID=karnataka.StateID, 
                DistrictID=district_map.get(d_name), 
                Active=True
            )
            session.add(unit)
            session.commit()
        unit_map[u_name] = unit.UnitID

    # 4. Extract and Populate Crime Heads
    crime_head_map = {}
    unique_crimes = df['CrimeHead_Name'].dropna().unique()
    for c_name in unique_crimes:
        ch = session.query(CrimeHead).filter_by(CrimeGroupName=c_name).first()
        if not ch:
            ch = CrimeHead(CrimeGroupName=c_name, Active=True)
            session.add(ch)
            session.commit()
        crime_head_map[c_name] = ch.CrimeHeadID

    # 5. Populate CaseMaster
    for idx, row in df.iterrows():
        case = CaseMaster(
            CrimeNo=str(row.get('FIRNo', '')),
            CaseNo=str(row.get('Crime_No', '')),
            PoliceStationID=unit_map.get(row.get('UnitName')),
            CrimeMajorHeadID=crime_head_map.get(row.get('CrimeHead_Name')),
            BriefFacts=str(row.get('Description', ''))
        )
        session.add(case)
        # We flush to get the autoincremented CaseMasterID
        session.flush()

        # Add Complainant if available
        # Note: Prototype dataset doesn't have explicit complainant, victim, accused columns parsed cleanly, 
        # so we will use mock values based on data logic if needed or skip.
        # This script sets up the baseline infrastructure for the hackathon.

    session.commit()
    print("Migration complete!")

if __name__ == "__main__":
    migrate_data()
