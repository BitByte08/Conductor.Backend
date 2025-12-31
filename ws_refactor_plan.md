# WebSocket Architecture Refactor

## Problem
Currently, `connected_agents[agent_id] = websocket` allows only one connection. If Frontend connects using the same ID, it kicks the Agent.

## Solution
Create a `ConnectionManager`.

### Structures
*   `active_agents`: `Dict[str, WebSocket]` (The actual Agent process)
*   `active_clients`: `Dict[str, List[WebSocket]]` (Frontend users watching an agent)

### Flows
1.  **Agent Connects**: `/ws/agent/{id}` -> Stored in `active_agents`.
2.  **Frontend Connects**: `/ws/client/{id}` -> Stored in `active_clients[{id}]`.
3.  **Log from Agent**:
    *   Backend receives msg from `active_agents[{id}]`.
    *   Backend broadcasts to all `active_clients[{id}]`.
4.  **Command from Frontend**:
    *   Frontend sends HTTP POST `/api/agent/{id}/command`.
    *   Backend sends WS message to `active_agents[{id}]`.

## Implementation
*   Modify `backend/main.py` to use a `ConnectionManager` class.
*   Update `frontend/src/hooks/useAgentSocket.ts` to connect to `/ws/client/{id}`.
