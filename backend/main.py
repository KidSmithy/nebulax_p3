"""
backend/main.py
===============================================================================
NebulaX P3 Unified Rail Digital Twin - Primary Service Entry Point
===============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'backend...' imports work from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json

from backend.core.harmonizer import TelemetryHarmonizer
from backend.services.whatif_engine import WhatIfEngine
from backend.api.websocket_streamer import ConnectionManager
from backend.api.routes import setup_routes
from backend.services.demo_seed import NSL_DEMO_ENABLED, seed_nsl_findings


harmonizer = TelemetryHarmonizer()
whatif_engine = WhatIfEngine(harmonizer=harmonizer)
manager = ConnectionManager(harmonizer=harmonizer, whatif_engine=whatif_engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch background broadcast task
    stream_task = asyncio.create_task(manager.start_stream_loop())
    # Scoring four files takes a few seconds; do it off the event loop so the server is up immediately.
    seed_task = asyncio.create_task(asyncio.to_thread(seed_nsl_findings, harmonizer)) if NSL_DEMO_ENABLED else None
    yield
    # Shutdown: cancel task
    if seed_task:
        seed_task.cancel()
    stream_task.cancel()
    try:
        await stream_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="NebulaX P3 Unified Rail Digital Twin",
    version="1.0.0",
    description="Full-stack hybrid vehicle-infrastructure digital twin backend",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API
app.include_router(setup_routes(harmonizer, whatif_engine))

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                data = json.loads(raw_text)
                await manager.handle_client_message(data)
            except Exception as e:
                print(f"[WebSocket] Error handling client message: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket] Unexpected disconnect: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    target = "main:app" if Path.cwd().name == "backend" else "backend.main:app"
    uvicorn.run(target, host="0.0.0.0", port=8000, reload=True)

