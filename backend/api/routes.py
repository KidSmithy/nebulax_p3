"""
backend/api/routes.py
===============================================================================
REST API Endpoints for NebulaX P3 Rail Digital Twin.
===============================================================================
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.core.config import CORRUGATION_ZONES, INTERVENTION_ACTIONS, METRIC_THRESHOLDS
from backend.services.ai_insight import AIInsightService
from backend.services.whatif_engine import ACTION_CATALOG

router = APIRouter(prefix="/api")

ai_service = AIInsightService()


class WhatIfRequest(BaseModel):
    action: str
    enabled: bool = True


class AskRequest(BaseModel):
    question: str = Field(..., max_length=500)


class InsightRequest(BaseModel):
    force: bool = False


def setup_routes(harmonizer, whatif_engine):
    @router.get("/health")
    async def health_check():
        return {
            "status": "ONLINE",
            "service": "NebulaX P3 Unified Rail Digital Twin Backend",
            "version": "1.1.0",
            "subsystems": ["Door", "ACV", "SHM", "Rail_Corrugation"],
            "ai": ai_service.status(),
        }

    @router.get("/telemetry/current")
    async def get_current_telemetry():
        return harmonizer.generate_next_frame(dt=0.0)

    @router.get("/corrugation/zones")
    async def get_corrugation_zones():
        return {"zones": CORRUGATION_ZONES}

    @router.get("/metrics/glossary")
    async def get_metric_glossary():
        """Thresholds the backend uses, so the UI can never disagree with it."""
        return {"thresholds": METRIC_THRESHOLDS}

    @router.get("/whatif/actions")
    async def get_whatif_actions():
        return {
            "actions": [
                {"action": a, **ACTION_CATALOG.get(a, {})} for a in INTERVENTION_ACTIONS
            ],
            "active": dict(harmonizer.interventions),
        }

    @router.post("/whatif")
    async def simulate_whatif(req: WhatIfRequest):
        return whatif_engine.simulate_action(req.action, req.enabled)

    @router.post("/whatif/reset")
    async def reset_whatif():
        harmonizer.reset_actions()
        whatif_engine.active_interventions = dict(harmonizer.interventions)
        return {"status": "RESET", "active": dict(harmonizer.interventions)}

    # ------------------------------------------------------------------
    # AI assistance
    # ------------------------------------------------------------------
    @router.get("/ai/status")
    async def ai_status():
        return ai_service.status()

    @router.post("/ai/insight")
    async def ai_insight(req: Optional[InsightRequest] = None):
        frame = harmonizer.generate_next_frame(dt=0.0)
        insight = ai_service.get_insight(frame, force=bool(req and req.force))
        return {
            "insight": insight,
            "frame_id": frame["frame_id"],
            "chainage_km": frame["track_chainage_km"],
        }

    @router.post("/ai/ask")
    async def ai_ask(req: AskRequest):
        frame = harmonizer.generate_next_frame(dt=0.0)
        return ai_service.ask(req.question, frame)

    return router
