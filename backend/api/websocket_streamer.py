"""
backend/api/websocket_streamer.py
===============================================================================
WebSocket Telemetry Streaming Manager:
- Full-duplex WebSocket communication at 10 Hz (100ms ticks)
- Handles client commands: play, pause, speed multiplier (1x - 10x), seek chainage
- Dispatches counterfactual mutations
===============================================================================
"""

import json
import asyncio
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self, harmonizer, whatif_engine):
        self.active_connections: Set[WebSocket] = set()
        self.harmonizer = harmonizer
        self.whatif_engine = whatif_engine
        self.is_playing = True
        self.speed_multiplier = 1.0
        self.stream_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"[WebSocket] Client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"[WebSocket] Client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        payload = json.dumps(message)
        to_remove = set()
        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
            except Exception:
                to_remove.add(conn)
        for dead in to_remove:
            self.active_connections.discard(dead)

    async def handle_client_message(self, data: dict):
        msg_type = data.get("type")
        if msg_type == "playback":
            if "playing" in data:
                self.is_playing = bool(data["playing"])
            if "speed" in data:
                self.speed_multiplier = max(0.2, min(float(data["speed"]), 10.0))
        elif msg_type == "seek":
            target_kp = data.get("kp")
            if target_kp is not None:
                self.harmonizer.seek_chainage(float(target_kp))
        elif msg_type == "whatif":
            action = data.get("action")
            enabled = data.get("enabled", True)
            if action:
                result = self.whatif_engine.simulate_action(action, enabled)
                # The measured impact is the whole point of the sandbox, so it
                # is broadcast back rather than discarded. Without this the
                # operator toggles a switch and sees no confirmation at all.
                await self.broadcast({"type": "whatif_result", **result})
        elif msg_type == "reset_whatif":
            self.harmonizer.reset_actions()
            self.whatif_engine.active_interventions = dict(self.harmonizer.interventions)
            await self.broadcast({
                "type": "whatif_result",
                "status": "RESET",
                "active_interventions": dict(self.harmonizer.interventions),
            })

    async def start_stream_loop(self):
        print("[WebSocket] Starting 10 Hz broadcast loop...")
        while True:
            try:
                if self.is_playing and self.active_connections:
                    frame = self.harmonizer.generate_next_frame(
                        dt=0.1,
                        speed_multiplier=self.speed_multiplier
                    )
                    frame["type"] = "frame"
                    await self.broadcast(frame)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WebSocket] Error in broadcast loop: {e}")
                await asyncio.sleep(0.1)
