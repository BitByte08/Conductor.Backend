from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Dict, List, Optional
import json
import logging
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import timedelta

import models, database, auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConductorBackend")

# Create tables
models.Base.metadata.create_all(bind=database.engine)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS to allow the CDN/front-end origin (cd.bitworkspace.kr) and common dev origins
origins = [
    "https://cd.bitworkspace.kr",
    "http://cd.bitworkspace.kr",
    "https://conductor.bitbyte08.workers.dev",
    "http://conductor.bitbyte08.workers.dev",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Permissions helpers
async def is_agent_owner(agent_id: str, user: models.User, db: Session) -> bool:
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    return agent is not None and agent.owner_id == user.id

async def has_manage_permission(agent_id: str, user: models.User, db: Session) -> bool:
    if await is_agent_owner(agent_id, user, db):
        return True
    collab = db.query(models.AgentCollaborator).filter(models.AgentCollaborator.agent_id == agent_id, models.AgentCollaborator.user_id == user.id).first()
    if collab and collab.role == 'manager':
        return True
    return False

async def is_collaborator(agent_id: str, user: models.User, db: Session) -> bool:
    collab = db.query(models.AgentCollaborator).filter(models.AgentCollaborator.agent_id == agent_id, models.AgentCollaborator.user_id == user.id).first()
    return collab is not None

class ConnectionManager:
    def __init__(self):
        # agent_id -> WebSocket (The actual Agent)
        self.active_agents: Dict[str, WebSocket] = {}
        # agent_id -> List[WebSocket] (Frontend clients watching this agent)
        self.active_clients: Dict[str, List[WebSocket]] = {}
        # agent_id -> last seen metadata string
        self.agent_metadata: Dict[str, str] = {}
        # agent_id -> last server status (ONLINE/OFFLINE)
        self.agent_server_status: Dict[str, str] = {}

    async def connect_agent(self, agent_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_agents[agent_id] = websocket
        logger.info(f"Agent connected: {agent_id}")
        await self.broadcast_to_clients(agent_id, json.dumps({
            "type": "AGENT_STATUS",
            "status": "ONLINE"
        }))

    async def disconnect_agent(self, agent_id: str):
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        logger.info(f"Agent disconnected: {agent_id}")
        await self.broadcast_to_clients(agent_id, json.dumps({
            "type": "AGENT_STATUS",
            "status": "OFFLINE"
        }))

    async def connect_client(self, agent_id: str, websocket: WebSocket):
        await websocket.accept()
        if agent_id not in self.active_clients:
            self.active_clients[agent_id] = []
        self.active_clients[agent_id].append(websocket)
        logger.info(f"Client connected to view agent: {agent_id}")
        
        # Send initial status
        status = "ONLINE" if agent_id in self.active_agents else "OFFLINE"
        await websocket.send_text(json.dumps({
            "type": "AGENT_STATUS",
            "status": status
        }))

    def disconnect_client(self, agent_id: str, websocket: WebSocket):
        if agent_id in self.active_clients:
            if websocket in self.active_clients[agent_id]:
                self.active_clients[agent_id].remove(websocket)

    async def broadcast_to_clients(self, agent_id: str, message: str):
        """Send message from Agent to all interested Clients"""
        if agent_id in self.active_clients:
            for connection in self.active_clients[agent_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")

    async def send_to_agent(self, agent_id: str, message: dict):
        if agent_id in self.active_agents:
            await self.active_agents[agent_id].send_text(json.dumps(message))
        else:
            raise Exception("Agent not connected")

manager = ConnectionManager()

class CommandRequest(BaseModel):
    command: str

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AgentCreate(BaseModel):
    name: str

# Auth Routes
@app.post("/auth/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    new_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = auth.create_access_token(data={"sub": new_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except auth.JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

@app.get("/auth/me")
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username}

@app.post("/api/agents/create")
async def create_agent(agent_data: AgentCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    import uuid
    agent_id = str(uuid.uuid4())
    new_agent = models.Agent(id=agent_id, name=agent_data.name, owner_id=current_user.id)
    db.add(new_agent)
    db.commit()
    return {"id": agent_id, "name": agent_data.name}


@app.get("/")
async def root():
    return {"message": "Conductor Backend Online"}

@app.get("/api/agents")
async def get_agents(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Return agents owned by user OR shared with user
    owned_agents = db.query(models.Agent).filter(models.Agent.owner_id == current_user.id).all()
    collab_rows = db.query(models.AgentCollaborator).filter(models.AgentCollaborator.user_id == current_user.id).all()
    collab_agent_ids = {c.agent_id for c in collab_rows}

    result = []

    # Owned agents (owner role)
    for agent in owned_agents:
        status = "ONLINE" if agent.id in manager.active_agents else "OFFLINE"
        metadata = manager.agent_metadata.get(agent.id, "Unknown")
        server_status = manager.agent_server_status.get(agent.id, "OFFLINE")
        result.append({"id": agent.id, "name": agent.name, "status": status, "metadata": metadata, "server_status": server_status, "role": "owner"})

    # Shared agents
    for agent_id in collab_agent_ids:
        agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
        if not agent:
            continue
        # avoid duplicate if owned
        if agent.owner_id == current_user.id:
            continue
        status = "ONLINE" if agent.id in manager.active_agents else "OFFLINE"
        metadata = manager.agent_metadata.get(agent.id, "Unknown")
        server_status = manager.agent_server_status.get(agent.id, "OFFLINE")
        collab = next((c for c in collab_rows if c.agent_id == agent.id), None)
        role = collab.role if collab else "viewer"
        result.append({"id": agent.id, "name": agent.name, "status": status, "metadata": metadata, "server_status": server_status, "role": role})

    return result

# --- Agent Control API ---

class CollaboratorRequest(BaseModel):
    username: str
    role: Optional[str] = "viewer"

@app.post("/api/agent/{agent_id}/start")
async def start_server(agent_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # permission: must be owner or manager
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")
    try:
        await manager.send_to_agent(agent_id, {
            "type": "START_SERVER",
            "payload": { "jar_path": "server.jar" }
        })
        return {"status": "sent"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/agent/{agent_id}/collaborators")
async def invite_collaborator(agent_id: str, req: CollaboratorRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # only owner can invite
    if not await is_agent_owner(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Only owner can invite collaborators")

    user = db.query(models.User).filter(models.User.username == req.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing = db.query(models.AgentCollaborator).filter(models.AgentCollaborator.agent_id == agent_id, models.AgentCollaborator.user_id == user.id).first()
    if existing:
        existing.role = req.role or existing.role
        db.add(existing)
    else:
        collab = models.AgentCollaborator(agent_id=agent_id, user_id=user.id, role=req.role or 'viewer')
        db.add(collab)
    db.commit()
    return {"status": "invited", "user": user.username, "role": req.role}

@app.get("/api/agent/{agent_id}/collaborators")
async def list_collaborators(agent_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # only owner or collaborator can view
    if not (await is_agent_owner(agent_id, current_user, db) or await is_collaborator(agent_id, current_user, db)):
        raise HTTPException(status_code=403, detail="Permission denied")
    rows = db.query(models.AgentCollaborator).filter(models.AgentCollaborator.agent_id == agent_id).all()
    result = []
    for r in rows:
        u = db.query(models.User).filter(models.User.id == r.user_id).first()
        result.append({"id": r.user_id, "username": u.username if u else None, "role": r.role})
    return result

@app.delete("/api/agent/{agent_id}/collaborators/{user_id}")
async def remove_collaborator(agent_id: str, user_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # only owner or manager can remove (managers can remove others, owner can remove anyone)
    if not (await is_agent_owner(agent_id, current_user, db) or await has_manage_permission(agent_id, current_user, db)):
        raise HTTPException(status_code=403, detail="Permission denied")
    row = db.query(models.AgentCollaborator).filter(models.AgentCollaborator.agent_id == agent_id, models.AgentCollaborator.user_id == user_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not a collaborator")
    db.delete(row)
    db.commit()
    return {"status": "removed"}

@app.delete("/api/agent/{agent_id}")
async def delete_agent(agent_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # only owner can delete
    if not await is_agent_owner(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Only owner can delete agent")
    
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Delete all collaborators first
    db.query(models.AgentCollaborator).filter(models.AgentCollaborator.agent_id == agent_id).delete()
    
    # Delete the agent
    db.delete(agent)
    db.commit()
    
    return {"status": "deleted"}

@app.post("/api/agent/{agent_id}/stop")
async def stop_server(agent_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")
    try:
        await manager.send_to_agent(agent_id, {
            "type": "STOP_SERVER",
            "payload": {}
        })
        return {"status": "sent"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/agent/{agent_id}/config")
async def update_config(agent_id: str, payload: Dict[str, str], current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")
    ram_mb = payload.get("ram_mb")
    if not ram_mb:
        raise HTTPException(status_code=400, detail="ram_mb required")
    try:
        await manager.send_to_agent(agent_id, {
            "type": "UPDATE_CONFIG",
            "payload": { "ram_mb": ram_mb }
        })
        return {"status": "sent"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/agent/{agent_id}/command")
async def send_command(agent_id: str, req: CommandRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")
    try:
        await manager.send_to_agent(agent_id, {
            "type": "COMMAND",
            "payload": { "command": req.command }
        })
        return {"status": "sent"}
    except Exception as e:
        return {"error": str(e)}

# --- WebSockets ---

import httpx

# ... existing imports ...

@app.get("/api/metadata/versions/{type}")
async def get_versions(type: str, q: str = "", limit: int = 100):
    """Return versions for given type. Optional query `q` filters versions (substring match)."""
    q = (q or "").strip()
    limit = max(1, min(limit, 1000))

    if type == "vanilla":
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://piston-meta.mojang.com/mc/game/version_manifest_v2.json")
            data = resp.json()
            # Return release versions only
            versions = [v["id"] for v in data["versions"] if v["type"] == "release"]
            if q:
                filtered = [v for v in versions if q in v]
            else:
                filtered = versions[::-1]  # newest first
            return filtered[:limit]

    elif type == "paper":
        async with httpx.AsyncClient() as client:
            # Paper lists versions, return them (newest first)
            resp = await client.get("https://api.papermc.io/v2/projects/paper")
            data = resp.json()
            versions = data.get("versions", [])
            if q:
                filtered = [v for v in versions if q in v]
            else:
                filtered = versions[::-1]
            return filtered[:limit]

    return []

@app.post("/api/agent/{agent_id}/install")
async def install_server(agent_id: str, payload: Dict[str, str], current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # permission: must be owner or manager
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")

    # payload: { "type": "vanilla", "version": "1.20.4" }
    version = payload.get("version")
    server_type = payload.get("type")
    
    url = ""
    filename = "server.jar"
    
    if server_type == "vanilla":
        # We need to fetch the download URL from the manifest
        # This is expensive to do here, ideally cache it.
        # For MVP, we can just send the command if we knew the URL.
        # Let's do a quick lookup.
        async with httpx.AsyncClient() as client:
            manifest_resp = await client.get("https://piston-meta.mojang.com/mc/game/version_manifest_v2.json")
            manifest = manifest_resp.json()
            version_url = next((v["url"] for v in manifest["versions"] if v["id"] == version), None)
            if version_url:
                v_resp = await client.get(version_url)
                v_data = v_resp.json()
                url = v_data["downloads"]["server"]["url"]
                
    elif server_type == "paper":
        # https://api.papermc.io/v2/projects/paper/versions/{version}/builds
        # Get latest build
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://api.papermc.io/v2/projects/paper/versions/{version}/builds")
            if resp.status_code != 200:
                 return {"error": f"Version {version} not found (PaperMC API)"}
            
            data = resp.json()
            builds = data.get("builds", [])
            if not builds:
                return {"error": f"No builds found for version {version}"}
                
            latest_build = builds[-1]["build"]
            filename = f"paper-{version}-{latest_build}.jar"
            url = f"https://api.papermc.io/v2/projects/paper/versions/{version}/builds/{latest_build}/downloads/{filename}"
            filename = "server.jar" # Rename to generic for simplicity

    if not url:
        return {"error": "Could not resolve download URL"}

    try:
        await manager.send_to_agent(agent_id, {
            "type": "INSTALL_SERVER",
            "payload": { 
                "url": url, 
                "filename": filename,
                "server_type": server_type,
                "version": version
            }
        })
    except Exception as e:
        return {"error": str(e)}
    return {"status": "installing", "url": url}

@app.get("/api/mods/search")
async def search_mods(query: str, version: str = ""):
    # https://api.modrinth.com/v2/search?query=jei&facets=[["categories:forge"],["versions:1.20.1"]]
    # Simplify for MVP: just query
    async with httpx.AsyncClient() as client:
        # We can add facets if needed, e.g. facets=[["project_type:mod"]]
        params = {"query": query}
        if version:
            # Modrinth facets syntax is weird json string
            pass 
        
        resp = await client.get("https://api.modrinth.com/v2/search", params=params)
        data = resp.json()
        return data.get("hits", [])

@app.post("/api/agent/{agent_id}/mods")
async def install_mod(agent_id: str, payload: Dict[str, str], current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # payload: { "url": "...", "filename": "..." }
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")

    url = payload.get("url")
    filename = payload.get("filename")
    
    if not url or not filename:
        return {"error": "Missing url or filename"}

    if agent_id not in manager.active_agents:
        return {"error": "Agent not connected"}

    await manager.send_to_agent(agent_id, {
        "type": "INSTALL_MOD",
        "payload": { "url": url, "filename": filename }
    })
    return {"status": "installing_mod", "file": filename}



@app.post("/api/agent/{agent_id}/properties/fetch")
async def fetch_properties(agent_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not await is_collaborator(agent_id, current_user, db) and not await is_agent_owner(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")
    try:
        await manager.send_to_agent(agent_id, {
            "type": "READ_PROPERTIES",
            "payload": {}
        })
        return {"status": "requested"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/agent/{agent_id}/properties/update")
async def update_properties(agent_id: str, payload: Dict[str, str], current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not await has_manage_permission(agent_id, current_user, db):
        raise HTTPException(status_code=403, detail="Permission denied")
    try:
        await manager.send_to_agent(agent_id, {
            "type": "WRITE_PROPERTIES",
            "payload": { "properties": payload }
        })
        return {"status": "sent"}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/agent/{agent_id}")
async def ws_agent(websocket: WebSocket, agent_id: str, db: Session = Depends(get_db)):
    # Note: WebSocket dependencies can be tricky. For simplicty we trust agent for now?
    # Or strict check: 
    # agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    # if not agent:
    #     await websocket.close(code=4003) # Forbidden
    #     return
    
    # Actually, Depends doesn't work easily in WebSocket decorator in some versions,
    # but manually using session context manager works.
    
    with database.SessionLocal() as session:
        agent = session.query(models.Agent).filter(models.Agent.id == agent_id).first()
        if not agent:
            # Allow "test-agent" for debugging if needed, or strictly reject.
            if agent_id != "test-agent":
                logger.warning(f"Unknown agent tried to connect: {agent_id}")
                await websocket.close()
                return

    await manager.connect_agent(agent_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # If this is a JSON heartbeat, extract metadata and status for dashboard
            try:
                obj = json.loads(data)
                t = obj.get("type")
                payload = obj.get("payload") or {}
                if t == "HEARTBEAT":
                    md = payload.get("metadata")
                    st = payload.get("server_status")
                    if md:
                        manager.agent_metadata[agent_id] = md
                    if st:
                        manager.agent_server_status[agent_id] = st
            except Exception:
                pass
            # Broadcast everything from agent to clients (Logs, Heartbeats)
            await manager.broadcast_to_clients(agent_id, data)
    except WebSocketDisconnect:
        logger.info(f"Agent {agent_id} disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"Agent WS error: {e}")
    finally:
        await manager.disconnect_agent(agent_id)


@app.websocket("/ws/client/{agent_id}")
async def ws_client(websocket: WebSocket, agent_id: str):
    await manager.connect_client(agent_id, websocket)
    try:
        while True:
            # Clients might send pings? For now ignore inputs.
            # Only listen.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_client(agent_id, websocket)
