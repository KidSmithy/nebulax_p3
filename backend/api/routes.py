"""
backend/api/routes.py
===============================================================================
REST API Endpoints for NebulaX P3 Rail Digital Twin.
===============================================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.core.config import CORRUGATION_ZONES

router = APIRouter(prefix="/api")

class WhatIfRequest(BaseModel):
    action: str
    enabled: bool = True

def setup_routes(harmonizer, whatif_engine):
    @router.get("/health")
    async def health_check():
        return {
            "status": "ONLINE",
            "service": "NebulaX P3 Unified Rail Digital Twin Backend",
            "version": "1.0.0",
            "subsystems": ["Door", "ACV", "SHM", "Rail_Corrugation"]
        }

    @router.get("/telemetry/current")
    async def get_current_telemetry():
        return harmonizer.generate_next_frame(dt=0.0)

    @router.get("/corrugation/zones")
    async def get_corrugation_zones():
        return {"zones": CORRUGATION_ZONES}

    @router.post("/whatif")
    async def simulate_whatif(req: WhatIfRequest):
        res = whatif_engine.simulate_action(req.action, req.enabled)
        return res

    return router
