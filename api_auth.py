from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel
import jwt
import datetime
import sqlite3
from typing import Optional, List
from passlib.context import CryptContext

router = APIRouter()

SECRET_KEY = "ksp_nexus_super_secret"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str
    jurisdiction: str

class UserResponse(BaseModel):
    username: str
    role: str
    jurisdiction: str

def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.split("Bearer ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/api/auth/login")
def login(req: LoginRequest):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (req.username.lower(),))
    user = c.fetchone()
    conn.close()

    if not user or not pwd_context.verify(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    payload = {
        "username": user["username"],
        "role": user["role"],
        "jurisdiction": user["jurisdiction"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=20)
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    from api_audit import log_audit
    log_audit(user["username"], user["role"], "LOGIN", "")
    
    return {"token": token, "role": user["role"], "jurisdiction": user["jurisdiction"], "username": user["username"]}


@router.post("/api/auth/users", response_model=UserResponse)
def create_user(req: UserCreateRequest, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "Administrator":
        raise HTTPException(status_code=403, detail="Only Administrators can create users")
    
    hashed_password = pwd_context.hash(req.password)
    
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, role, jurisdiction) VALUES (?, ?, ?, ?)",
                  (req.username.lower(), hashed_password, req.role, req.jurisdiction))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    
    conn.close()
    return UserResponse(username=req.username.lower(), role=req.role, jurisdiction=req.jurisdiction)


@router.get("/api/auth/users", response_model=List[UserResponse])
def get_users(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "Administrator":
        raise HTTPException(status_code=403, detail="Only Administrators can view users")
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT username, role, jurisdiction FROM users")
    users = c.fetchall()
    conn.close()
    
    return [UserResponse(username=u["username"], role=u["role"], jurisdiction=u["jurisdiction"]) for u in users]
