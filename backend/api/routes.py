"""
backend/api/routes.py
===============================================================================
REST API Endpoints for NebulaX P3 Rail Digital Twin.
===============================================================================
"""

import io
import re
from typing import Any, Dict, List, Optional, Tuple
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.core.harmonizer import LINES
from backend.core.config import CORRUGATION_ZONES, INTERVENTION_ACTIONS, METRIC_THRESHOLDS
from backend.services.ai_insight import AIInsightService
from backend.services.whatif_engine import ACTION_CATALOG

router = APIRouter(prefix="/api")

ai_service = AIInsightService()


def _natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def _extract_zip_files(content: bytes) -> List[Tuple[str, bytes]]:
    """Extracts valid data files (.csv, .txt, .log, .xlsx) from a ZIP archive."""
    extracted = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = sorted(z.namelist(), key=_natural_sort_key)
        for name in names:
            if name.endswith("/") or "__MACOSX" in name or "/." in name or name.startswith("."):
                continue
            base = name.split("/")[-1]
            if base.startswith("._") or not base:
                continue
            if base.lower().endswith((".csv", ".txt", ".tsv", ".log", ".xlsx", ".xls")):
                extracted.append((base, z.read(name)))
    return extracted



class WhatIfRequest(BaseModel):
    action: str
    enabled: bool = True


class AskRequest(BaseModel):
    question: str = Field(..., max_length=500)
    line: str = "NSL"
    car: Optional[int] = None


class InsightRequest(BaseModel):
    force: bool = False
    line: str = "NSL"
    car: Optional[int] = None


class IssueRequest(BaseModel):
    subsystem: str
    line: str = "NSL"
    car: Optional[int] = None


class ResolveRequest(BaseModel):
    subsystem: str
    car: Optional[int] = None
    line: str = "NSL"


def _line(value: str) -> str:
    line = (value or "NSL").upper().strip()
    if line not in LINES:
        raise HTTPException(status_code=400, detail=f"Unknown line: {value}")
    return line


def setup_routes(harmonizer, whatif_engine):
    def frame_for(line: str, car: Optional[int] = None) -> dict:
        """Current frame, scoped to one line so the AI only sees that line's findings."""
        frame = harmonizer.generate_next_frame(dt=0.0)
        frame["latest_upload_results"] = dict(harmonizer.uploads_by_line[line])
        frame["line"], frame["car"] = line, car
        return frame

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
        frame = frame_for(_line(req.line if req else "NSL"), req.car if req else None)
        insight = ai_service.get_insight(frame, force=bool(req and req.force))
        return {
            "insight": insight,
            "frame_id": frame["frame_id"],
            "chainage_km": frame["track_chainage_km"],
        }

    @router.post("/ai/issue")
    async def ai_issue(req: IssueRequest):
        """Explanation + suggestion copy for one 3D bubble's card."""
        frame = frame_for(_line(req.line), req.car)
        result = ai_service.explain_issue(req.subsystem.lower().strip(), frame)
        if result.get("source") == "validation":
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @router.post("/ai/ask")
    async def ai_ask(req: AskRequest):
        return ai_service.ask(req.question, frame_for(_line(req.line), req.car))

    # ------------------------------------------------------------------
    # Batch / CSV Model Inference (Conductor Onboarding)
    # ------------------------------------------------------------------
    @router.post("/predict/upload")
    async def upload_predict(
        file: UploadFile = File(...), subsystem: str = Form(...), line: str = Form("NSL")
    ):
        line = _line(line)
        content = await file.read()
        sub = subsystem.lower().strip()
        filename = file.filename or "uploaded_data.csv"
        is_zip = filename.lower().endswith(".zip")

        try:
            if is_zip:
                file_list = _extract_zip_files(content)
                if not file_list:
                    raise HTTPException(status_code=422, detail="No valid CSV or data files found in ZIP archive.")

                if len(file_list) == 1:
                    fname, fbytes = file_list[0]
                    if sub in ("door", "doors"):
                        result = harmonizer.broker.door_model.predict_from_csv(fbytes, fname)
                    elif sub in ("shm", "bogie"):
                        result = harmonizer.broker.shm_model.predict_from_csv(fbytes, fname)
                    elif sub in ("acv", "aircon"):
                        result = harmonizer.broker.acv_model.predict_from_csv(fbytes, fname)
                    elif sub in ("rail", "rail_corrugation", "corrugation"):
                        result = harmonizer.broker.rail_model.predict_from_csv(fbytes, fname)
                    else:
                        raise HTTPException(status_code=400, detail=f"Unknown subsystem: {subsystem}")
                else:
                    if sub in ("door", "doors"):
                        result = harmonizer.broker.door_model.predict_batch(file_list, filename)
                    elif sub in ("shm", "bogie"):
                        result = harmonizer.broker.shm_model.predict_batch(file_list, filename)
                    elif sub in ("acv", "aircon"):
                        result = harmonizer.broker.acv_model.predict_batch(file_list, filename)
                    elif sub in ("rail", "rail_corrugation", "corrugation"):
                        result = harmonizer.broker.rail_model.predict_batch(file_list, filename)
                    else:
                        raise HTTPException(status_code=400, detail=f"Unknown subsystem: {subsystem}")
            else:
                if sub in ("door", "doors"):
                    result = harmonizer.broker.door_model.predict_from_csv(content, filename)
                elif sub in ("shm", "bogie"):
                    result = harmonizer.broker.shm_model.predict_from_csv(content, filename)
                elif sub in ("acv", "aircon"):
                    result = harmonizer.broker.acv_model.predict_from_csv(content, filename)
                elif sub in ("rail", "rail_corrugation", "corrugation"):
                    result = harmonizer.broker.rail_model.predict_from_csv(content, filename)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown subsystem: {subsystem}")
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

        harmonizer.set_upload_result(sub, result, line=line)
        return result

    @router.get("/predict/latest")
    async def get_latest_predictions():
        return {"predictions": harmonizer.uploads_by_line}

    # ------------------------------------------------------------------
    # Resolve an uploaded finding: apply its recommended repair for a real
    # measured delta (same engine the Repairs tab uses), then log it.
    # ------------------------------------------------------------------
    @router.post("/predict/resolve")
    async def resolve_prediction(req: ResolveRequest):
        line = _line(req.line)
        finding = harmonizer.uploads_by_line[line].get(req.subsystem)
        if finding is None:
            raise HTTPException(status_code=404, detail=f"No active finding for '{req.subsystem}'.")

        action = finding.get("recommended_action")
        measured_impact = None
        if action and action != "NONE":
            whatif_result = whatif_engine.simulate_action(action, True)
            measured_impact = whatif_result.get("measured_impact")

        entry = harmonizer.resolve_upload(
            req.subsystem, car=req.car, measured_impact=measured_impact, line=line
        )
        return entry

    @router.get("/log")
    async def get_resolved_log():
        return {"log": harmonizer.resolved_log}

    @router.post("/acv/predict")
    async def acv_predict(file: UploadFile = File(...)):
        content = await file.read()
        filename = file.filename or "acv_data.csv"
        try:
            if filename.lower().endswith(".zip"):
                file_list = _extract_zip_files(content)
                if not file_list:
                    raise HTTPException(status_code=422, detail="No valid CSV or Excel files found in ZIP archive.")
                if len(file_list) == 1:
                    result = harmonizer.broker.acv_model.predict_from_csv(file_list[0][1], file_list[0][0])
                else:
                    result = harmonizer.broker.acv_model.predict_batch(file_list, filename)
            else:
                result = harmonizer.broker.acv_model.predict_from_csv(content, filename)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")
        harmonizer.set_upload_result("acv", result)
        return result

    return router
