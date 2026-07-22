import os
import random
from sqlalchemy.orm import sessionmaker
from database_schema import (
    create_db, Act, Section, CaseCategory, CaseStatusMaster, Court, Rank, Designation,
    ReligionMaster, OccupationMaster, CasteMaster, GravityOffence, UnitType, CrimeSubHead,
    CaseMaster, Accused, Victim, Employee, ComplainantDetails, ActSectionAssociation
)

def populate_lookups(session):
    print("Populating lookup tables...")
    
    # Act
    acts = [
        ("IPC", "Indian Penal Code", "IPC", True),
        ("NDPS", "Narcotic Drugs and Psychotropic Substances Act", "NDPS", True),
        ("MVA", "Motor Vehicles Act", "MVA", True),
        ("POCSO", "Protection of Children from Sexual Offences", "POCSO", True)
    ]
    for code, desc, short, active in acts:
        if not session.query(Act).filter_by(ActCode=code).first():
            session.add(Act(ActCode=code, ActDescription=desc, ShortName=short, Active=active))

    # Section
    sections = [
        ("IPC", "302", "Punishment for murder"),
        ("IPC", "379", "Punishment for theft"),
        ("IPC", "420", "Cheating and dishonestly inducing delivery of property"),
        ("NDPS", "20", "Punishment for contravention in relation to cannabis plant and cannabis")
    ]
    for act, code, desc in sections:
        if not session.query(Section).filter_by(SectionCode=code).first():
            session.add(Section(ActCode=act, SectionCode=code, SectionDescription=desc, Active=True))

    # CaseCategory
    categories = ["FIR", "UDR", "Zero FIR", "PAR"]
    for c in categories:
        if not session.query(CaseCategory).filter_by(LookupValue=c).first():
            session.add(CaseCategory(LookupValue=c))

    # CaseStatusMaster
    statuses = ["Under Investigation", "Charge Sheeted", "Closed", "Pending Trial"]
    for s in statuses:
        if not session.query(CaseStatusMaster).filter_by(CaseStatusName=s).first():
            session.add(CaseStatusMaster(CaseStatusName=s))

    # Court
    courts = ["District and Sessions Court", "High Court of Karnataka", "Magistrate Court", "Fast Track Court"]
    for c in courts:
        if not session.query(Court).filter_by(CourtName=c).first():
            session.add(Court(CourtName=c, DistrictID=1, StateID=1, Active=True))

    # Rank
    ranks = [("Constable", 5), ("Head Constable", 4), ("Sub-Inspector", 3), ("Inspector", 2), ("DSP", 1)]
    for r, h in ranks:
        if not session.query(Rank).filter_by(RankName=r).first():
            session.add(Rank(RankName=r, Hierarchy=h, Active=True))

    # Designation
    designations = ["Investigating Officer", "Station House Officer", "Cyber Cell Head"]
    for d in designations:
        if not session.query(Designation).filter_by(DesignationName=d).first():
            session.add(Designation(DesignationName=d, Active=True))

    # ReligionMaster
    religions = ["Hindu", "Muslim", "Christian", "Sikh", "Other"]
    for r in religions:
        if not session.query(ReligionMaster).filter_by(ReligionName=r).first():
            session.add(ReligionMaster(ReligionName=r))

    # OccupationMaster
    occupations = ["Farmer", "Government Employee", "Business", "Student", "Private Employee", "Unemployed"]
    for o in occupations:
        if not session.query(OccupationMaster).filter_by(OccupationName=o).first():
            session.add(OccupationMaster(OccupationName=o))

    # CasteMaster
    castes = ["General", "OBC", "SC", "ST"]
    for c in castes:
        if not session.query(CasteMaster).filter_by(caste_master_name=c).first():
            session.add(CasteMaster(caste_master_name=c))
            
    # GravityOffence
    gravities = ["Heinous", "Non-Heinous"]
    for g in gravities:
        if not session.query(GravityOffence).filter_by(LookupValue=g).first():
            session.add(GravityOffence(LookupValue=g))
            
    # UnitType
    unit_types = ["Police Station", "Cyber Crime Unit", "Traffic Police Station", "Women Police Station"]
    for ut in unit_types:
        if not session.query(UnitType).filter_by(UnitTypeName=ut).first():
            session.add(UnitType(UnitTypeName=ut, Active=True))

    session.commit()

def generate_dummy_data(session):
    print("Generating thousands of dummy records for transactional tables...")
    
    first_names = ["Ramesh", "Suresh", "Priya", "Anjali", "Kiran", "Amit", "Rahul", "Neha", "Pooja", "Vikram", "Sunil", "Anita"]
    last_names = ["Kumar", "Sharma", "Singh", "Reddy", "Gowda", "Patil", "Desai", "Rao", "Jain", "Hegde"]
    
    # Populate Employees
    ranks = session.query(Rank).all()
    desigs = session.query(Designation).all()
    if not session.query(Employee).first():
        print("Creating 1000 Employees...")
        for i in range(1000):
            emp = Employee(
                FirstName=f"{random.choice(first_names)} {random.choice(last_names)}",
                RankID=random.choice(ranks).RankID if ranks else None,
                DesignationID=random.choice(desigs).DesignationID if desigs else None,
                KGID=f"KG{random.randint(10000, 99999)}",
                GenderID=random.choice([1, 2])
            )
            session.add(emp)
        session.commit()
        
    employees = session.query(Employee).all()
    
    # Link CaseMaster with dummy relations
    print("Creating dummy Accused, Victims, and Complainants for 1000 recent cases...")
    cases = session.query(CaseMaster).limit(1000).all()
    acts = session.query(Act).all()
    
    for case in cases:
        # 1-3 Accused
        for _ in range(random.randint(1, 3)):
            accused = Accused(
                CaseMasterID=case.CaseMasterID,
                AccusedName=f"{random.choice(first_names)} {random.choice(last_names)}",
                AgeYear=random.randint(18, 65),
                GenderID=random.choice([1, 2]),
                PersonID=f"A{random.randint(1, 3)}"
            )
            session.add(accused)
            
        # 1-2 Victims
        for _ in range(random.randint(1, 2)):
            victim = Victim(
                CaseMasterID=case.CaseMasterID,
                VictimName=f"{random.choice(first_names)} {random.choice(last_names)}",
                AgeYear=random.randint(10, 80),
                GenderID=random.choice([1, 2]),
                VictimPolice=str(random.choice([0, 1]))
            )
            session.add(victim)
            
        # 1 Complainant
        complainant = ComplainantDetails(
            CaseMasterID=case.CaseMasterID,
            ComplainantName=f"{random.choice(first_names)} {random.choice(last_names)}",
            AgeYear=random.randint(20, 70),
            GenderID=random.choice([1, 2])
        )
        session.add(complainant)
        
        # 1-2 Acts
        if acts:
            for _ in range(random.randint(1, 2)):
                act = random.choice(acts)
                assoc = ActSectionAssociation(
                    CaseMasterID=case.CaseMasterID,
                    ActID=act.ActCode,
                    SectionID=str(random.randint(100, 500))
                )
                session.add(assoc)
                
        # Assign an IO
        if employees:
            case.PolicePersonID = random.choice(employees).EmployeeID

    session.commit()
    print("Dummy data generation complete!")

if __name__ == "__main__":
    engine = create_db('sqlite:///ksp_relational.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    populate_lookups(session)
    generate_dummy_data(session)
