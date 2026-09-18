"""
backend/services/ai_insight.py
===============================================================================
AI Insight Service
-------------------------------------------------------------------------------
Turns a raw 10 Hz engineering telemetry frame into an explanation a non-expert
can act on, using an OpenAI chat model (default: gpt-5.6-terra).

Design constraints:
  * The API key is read from backend/.env and never leaves the server. The
    browser talks only to this backend.
  * The model is given a PRE-INTERPRETED summary (values + thresholds + verdicts
    computed in Python), not a raw JSON dump. Grounding the model in the same
    thresholds the UI uses stops it inventing its own idea of "normal".
  * Calls are throttled and cached. A 10 Hz stream must never become 10 model
    calls per second.
  * If the key is missing, the quota is exhausted, or the API errors, a
    deterministic rule-based explanation is returned instead. The dashboard
    degrades in quality, never in availability.
===============================================================================
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from backend.core.config import (
    AI_INSIGHT_MIN_INTERVAL_SEC,
    METRIC_THRESHOLDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    classify_metric,
)

SYSTEM_PROMPT = """You are the onboard maintenance advisor for a Singapore MRT train, \
built into a live digital-twin dashboard. Your audience is NOT a rail engineer: \
assume an operations trainee, a manager, or a member of the public.

Rules:
1. Explain in plain English. No jargon unless you immediately define it in the same sentence.
2. Never invent numbers. Use only the readings supplied to you. Quote them with units.
3. Be specific about cause and effect, and about which subsystem is involved.
4. Say clearly whether someone needs to act now, act soon, or do nothing.
5. Be calm and factual. Do not exaggerate. If everything is fine, say so plainly.
6. Analogies to everyday experience are encouraged when they aid understanding \
(e.g. rail corrugation is like a washboard road surface).

Respond with JSON only, matching this shape exactly:
{
  "headline": "one short sentence, max 90 chars, what is happening right now",
  "severity": "GOOD" | "WATCH" | "ACTION_NEEDED",
  "summary": "2-3 sentences a non-expert can follow",
  "findings": [
    {"subsystem": "door|acv|shm|rail|fleet",
     "what_it_means": "plain explanation referencing the actual reading",
     "why_it_matters": "the practical consequence for passengers or cost"}
  ],
  "recommended_action": "the single most useful next step, or 'No action needed'",
  "urgency": "NONE" | "ROUTINE" | "WITHIN_7_DAYS" | "WITHIN_48_HOURS",
  "analogy": "one everyday comparison that makes the main issue intuitive"
}"""

ASK_SYSTEM_PROMPT = """You are the onboard maintenance advisor for a Singapore MRT train \
digital twin, answering a question from someone who is not a rail engineer.

Rules:
1. Answer in plain English, 2-5 sentences. Define any technical term you use.
2. Ground every claim in the live readings supplied. Never invent numbers.
3. If the readings cannot answer the question, say so and explain what would be needed.
4. Do not speculate about anything outside this train's telemetry.
Respond in plain prose, not JSON, not markdown headings."""

SUBSYSTEM_LABELS = {
    "door": "Passenger door (Door 3R)",
    "acv": "Air conditioning pack",
    "shm": "Structural health monitoring (SHM) - bogie",
    "rail_corrugation": "Rail corrugation",
}

# Which metrics to include in the model's context, with plain names.
SUMMARY_METRICS = [
    ("door", "motor_current_amps", "electric current the door motor draws"),
    ("door", "nominal_current_amps", "current a healthy door of this type should draw"),
    ("door", "transit_time_seconds", "seconds the door takes to travel"),
    ("door", "anomaly_score", "door fault probability"),
    ("door", "ghost_deviation_mm", "how far behind schedule the door leaf is, in mm"),
    ("door", "fault_type", "door fault classification"),
    ("acv", "supply_temp_c", "temperature of air blown into the cabin"),
    ("acv", "return_temp_c", "temperature of air drawn back out of the cabin"),
    ("acv", "delta_temp_c", "cooling achieved across the unit"),
    ("acv", "efficiency_rating", "aircon efficiency, 1.0 is perfect"),
    ("acv", "compressor_power_kw", "electricity the compressor consumes"),
    ("acv", "fault_type", "aircon fault classification"),
    ("shm", "vibration_rms_g", "average shaking at the axle box, in g"),
    ("shm", "peak_frequency_hz", "dominant vibration frequency"),
    ("shm", "bearing_defect_prob", "probability the wheel bearing is damaged"),
    ("shm", "fatigue_damage_index", "accumulated metal fatigue, 1.0 is end of life"),
    ("shm", "critical_weld_node", "most stressed weld joint"),
    ("rail_corrugation", "depth_microns", "depth of ripples worn into the rail"),
    ("rail_corrugation", "wavelength_class", "ripple pattern type"),
    ("rail_corrugation", "severity_score", "how bad this stretch of track is"),
    ("rail_corrugation", "maintenance_urgency", "track maintenance recommendation"),
]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def build_frame_summary(frame: Dict[str, Any]) -> str:
    """
    Render the frame as a compact, pre-interpreted briefing.

    Each numeric line carries the reading, the threshold band it falls in, and
    the verdict that the dashboard itself is displaying, so the model's words
    can never contradict the colour of the UI.
    """
    subs = frame.get("subsystems", {})
    lines: List[str] = [
        "LIVE TRAIN READINGS",
        f"Train: C151B-SET-402 car 3, North-South Line",
        f"Position: kilometre post {_fmt(frame.get('track_chainage_km'))} km",
        f"Speed: {_fmt(frame.get('train_speed_kmh'))} km/h",
    ]

    health = frame.get("fleet_health_index")
    if health is not None:
        lines.append(
            f"Overall train health: {health * 100:.0f}% "
            f"[verdict: {classify_metric('fleet_health_index', health)}]"
        )

    for group, key, meaning in SUMMARY_METRICS:
        node = subs.get(group) or {}
        if key not in node:
            continue
        value = node[key]
        path = f"{group}.{key}"
        spec = METRIC_THRESHOLDS.get(path)
        line = f"- {SUBSYSTEM_LABELS.get(group, group)} / {meaning}: {_fmt(value)}"
        if spec:
            line += f" {spec['unit']}".rstrip()
            better = "higher is better" if spec["higher_is_better"] else "lower is better"
            line += (
                f" [healthy at {_fmt(spec['good'])}, concerning past {_fmt(spec['warn'])}, "
                f"{better}; verdict: {classify_metric(path, value)}]"
            )
        lines.append(line)

    cf = frame.get("counterfactual") or {}
    if cf.get("active"):
        lines.append("")
        lines.append(
            "MAINTENANCE CURRENTLY SIMULATED: " + ", ".join(cf.get("active_actions", []))
        )
        for m in cf.get("metrics", []):
            if m.get("direction") != "unchanged":
                lines.append(
                    f"- {m['label']}: {_fmt(m['before'])} -> {_fmt(m['after'])} "
                    f"{m['unit']} ({m['direction']})"
                )

    uploads = frame.get("latest_upload_results") or {}
    if uploads:
        lines.append("")
        lines.append("RECENTLY UPLOADED CSV BATCH MODEL PREDICTIONS:")
        for sub_id, up in uploads.items():
            lines.append(f"- Subsystem [{sub_id.upper()}]: File '{up.get('file_name', 'data.csv')}' processed.")
            lines.append(f"  Model Verdict: {up.get('verdict', 'UNKNOWN')} | Anomaly score: {up.get('anomaly_score', 0)}")
            if up.get('conductor_summary'):
                lines.append(f"  Model Output & Finding: {up.get('conductor_summary')}")
            if up.get('recommended_action') and up.get('recommended_action') != 'NONE':
                lines.append(f"  Recommended Action: {up.get('recommended_action')}")

    zone = frame.get("next_corrugation_zone") or {}
    if zone:
        if zone.get("is_inside_zone"):
            lines.append(
                f"\nThe train is currently INSIDE a known rough track section "
                f"(KP {zone['kp_start']}-{zone['kp_end']})."
            )
        else:
            lines.append(
                f"\nNext known rough track section: KP {zone.get('kp_start')} "
                f"({_fmt(zone.get('distance_km'))} km ahead)."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
FALLBACK_EXPLANATIONS = {
    "door.anomaly_score": (
        "door",
        "The door motor is pulling more electricity than a healthy door needs, "
        "which points to friction in the tracks the door slides along.",
        "Left alone this leads to doors that stick or fail to close, which holds "
        "trains at the platform and delays the whole line.",
    ),
    "acv.efficiency_rating": (
        "acv",
        "The air-conditioning is producing less cooling than it should for the "
        "electricity it consumes.",
        "Passengers feel a warm cabin and the unit wastes energy, raising running costs.",
    ),
    "shm.bearing_defect_prob": (
        "shm",
        "Vibration analysis shows a repeating knock at the wheel bearing's "
        "characteristic frequency, the signature of a damaged bearing surface.",
        "A bearing that seizes in service can cause a train withdrawal, so this is "
        "caught early on purpose.",
    ),
    "shm.vibration_rms_g": (
        "shm",
        "The wheel assembly is shaking harder than normal.",
        "Sustained shaking accelerates metal fatigue in the bogie welds.",
    ),
    "rail_corrugation.severity_score": (
        "rail",
        "The top surface of the rail has worn into small ripples, like a washboard road.",
        "The ripples hammer the wheels, which is felt as noise and vibration inside the "
        "cabin and shortens the life of both rail and train.",
    ),
}

URGENCY_BY_SEVERITY = {
    "GOOD": ("NONE", "No action needed"),
    "WATCH": ("ROUTINE", "Keep monitoring at the next scheduled depot check"),
    "ACTION_NEEDED": ("WITHIN_7_DAYS", "Raise a maintenance job for the affected subsystem"),
}


def rule_based_insight(frame: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    """Deterministic explanation used whenever the model is unavailable."""
    subs = frame.get("subsystems", {})
    findings: List[Dict[str, str]] = []
    worst = "GOOD"

    for path, (subsystem, what, why) in FALLBACK_EXPLANATIONS.items():
        group, key = path.split(".")
        node = subs.get(group) or {}
        if key not in node:
            continue
        value = node[key]
        verdict = classify_metric(path, value)
        if verdict in ("WATCH", "ACTION_NEEDED"):
            spec = METRIC_THRESHOLDS.get(path, {})
            findings.append({
                "subsystem": subsystem,
                "what_it_means": f"{what} (reading: {_fmt(value)}{spec.get('unit', '')})",
                "why_it_matters": why,
            })
            if verdict == "ACTION_NEEDED":
                worst = "ACTION_NEEDED"
            elif worst == "GOOD":
                worst = "WATCH"

    health = frame.get("fleet_health_index")
    health_verdict = classify_metric("fleet_health_index", health) if health is not None else "GOOD"
    if health_verdict == "ACTION_NEEDED":
        worst = "ACTION_NEEDED"

    urgency, action = URGENCY_BY_SEVERITY[worst]

    if worst == "GOOD":
        headline = "All four subsystems are reading normal"
        summary = (
            "Nothing on this train needs attention right now. The doors, air-conditioning, "
            "wheel assembly and the track underneath are all within their normal ranges."
        )
        analogy = "Like a car dashboard with no warning lights on."
    else:
        headline = f"{len(findings)} subsystem(s) reading outside normal"
        names = ", ".join(sorted({f['subsystem'] for f in findings})) or "the train"
        summary = (
            f"Overall train health is {(health or 0) * 100:.0f}%. "
            f"The readings that stand out involve {names}. "
            "Each finding below explains what the number means and why it matters."
        )
        analogy = findings[0]["what_it_means"] if findings else ""

    return {
        "headline": headline,
        "severity": worst,
        "summary": summary,
        "findings": findings,
        "recommended_action": action,
        "urgency": urgency,
        "analogy": analogy,
        "source": "rule_based",
        "degraded_reason": reason or None,
    }


# ---------------------------------------------------------------------------
# Per-issue card copy (the 3D bubble's Explanation / Suggestion card)
# ---------------------------------------------------------------------------
ISSUE_SYSTEM_PROMPT = """You are a senior rolling-stock maintenance engineer for the SMRT North-South \\
Line (Alstom C151B trains). A batch-analysis model has just flagged a finding from an uploaded \\
sensor file. You write the two short paragraphs on the pop-up card a maintenance planner reads \\
before deciding what to do. The reader is technical enough to plan work but is not a specialist \\
in this subsystem, so name the real component and cause, and keep any jargon to one plain clause.

Write like an engineer briefing a colleague: what the data shows, the most probable physical cause, \\
and what is at stake if it is left. Do not restate the numbers as a list; pick the one or two that \\
carry the argument.

LENGTH IS A HARD LIMIT. The card is small; a reader glances at it. Short, plain sentences.

"explanation": 2 sentences, at most 3, and at most 45 words in total. Every sentence under 20 words.
  1. What the data shows: the key reading with units, and the car if one is named.
  2. The most probable physical cause, worded as likely (this is inferred from sensor data).
  3. Only if it fits the word limit: the consequence if left.
  Name ONE most probable cause, not a list of alternatives.

"suggestion": ONE sentence, at most 25 words, imperative voice: the concrete task on the named
  component (and car, if named), with the timeframe from the facts. It must fit the repair listed
  under "Repair on offer" but be tailored to this finding, not that repair's generic wording.

Style example (different subsystem, invented numbers, do not reuse its wording):
  explanation: "Nine of 40 door cycles drew abnormal motor current, peaking at 6.2 A. The likely cause is dry or dirty guide rails making the door drag."
  suggestion: "Clean and lubricate the door guide rails within the week, then re-check motor current."

Hard rules:
- Use only numbers that appear in the facts. Never invent readings, part numbers, or standards.
- Never mention the file name, the words "model", "algorithm", "confidence margin" or "anomaly score". \\
  Translate confidence into plain words ("clearly", "likely").
- If the status is GOOD: explanation says the readings are healthy in 2 sentences, suggestion is \\
  "No action needed." plus, optionally, one routine-monitoring note.
- No markdown, no bullet points, no headings.

Respond with JSON only: {"explanation": "...", "suggestion": "..."}"""

# What each subsystem's fault mechanisms look like, so the model reasons like the
# engineer rather than paraphrasing the finding.
ISSUE_DOMAIN_NOTES = {
    "door": (
        "Passenger plug door on a C151B car. The signal is door-motor current and travel time per "
        "open/close cycle. Sustained high current or slow cycles mean the motor is working against "
        "extra resistance: typically dry or contaminated guide rails, worn hanger rollers or "
        "bearings, a misaligned or rubbing door leaf, or debris in the sill track. Left alone this "
        "leads to door-fault trips, delayed departures and trains being taken out of service. "
        "Usual work: clean and lubricate the guide rails, inspect rollers, check leaf alignment."
    ),
    "acv": (
        "Air-conditioning (ACV) pack. The analysis compares each car's cabin-temperature behaviour "
        "against the consist median to point at the car most likely losing cooling. That points to "
        "the CAR, not the exact part: likely causes are refrigerant pressure loss (a slow leak), a "
        "restricted air filter or duct, or a degrading compressor. Passengers feel a warm cabin and "
        "the pack overworks. Usual work: leak-test the refrigerant circuit on that car, replace the "
        "filter, then recharge to specification and re-check supply/return temperatures."
    ),
    "shm": (
        "Bogie structural health monitoring from an axle-box accelerometer. Healthy vibration is up "
        "to about 2.5 g RMS, concerning above 3.5 g; fatigue-damage index is healthy below 0.40 and "
        "concerning above 0.70 (1.0 is end of life). A repeating knock at the bearing defect "
        "frequency indicates a damaged axle-box bearing race or roller; broadband high RMS can also "
        "come from wheel flats or loose fasteners. Fatigue concentrates at the named weld/rib node. "
        "A seized bearing can force a withdrawal or worse. Usual work: ultrasonic/vibration check "
        "of the axle-box bearing, re-grease or replace it, and inspect the named structural node."
    ),
    "rail": (
        "Rail corrugation: periodic wear ripples on the rail head, measured as depth in microns "
        "from vibration data. Ripples cause wheel-rail impact, cabin noise and vibration, and "
        "accelerate wear of bearings, fasteners and the rail itself. Short-pitch corrugation is the "
        "more aggressive class. The corrective work is rail grinding over the affected chainage."
    ),
}


EXPLANATION_MAX_WORDS = 45
SUGGESTION_MAX_WORDS = 25


def _too_long(parsed: Dict[str, Any]) -> bool:
    words = lambda k: len(str(parsed.get(k, "")).split())
    return words("explanation") > EXPLANATION_MAX_WORDS + 10 or words("suggestion") > SUGGESTION_MAX_WORDS + 8


def _urgency_phrase(subsystem: str, finding: Dict[str, Any]) -> str:
    """Timeframe the engineer would give, derived from the finding, never invented by the model."""
    status = finding.get("status")
    if subsystem == "rail":
        return {
            "IMMEDIATE_GRINDING_48H": "within 48 hours",
            "SCHEDULE_GRINDING_7D": "within 7 days",
        }.get(finding.get("maintenance_urgency", ""), "at the next scheduled track-access window")
    return {
        "ACTION_NEEDED": "within 48 hours" if subsystem == "shm" else "within the week",
        "WATCH": "at the next scheduled depot visit",
    }.get(status, "no urgency")


def _finding_facts(subsystem: str, f: Dict[str, Any]) -> List[str]:
    """Only the finding's own measurements, in plain lines. The live simulation is deliberately absent."""
    facts = [f"Status: {f.get('status')}"]
    if subsystem == "door":
        facts += [
            f"Door cycles analysed: {f.get('total_cycles')}; abnormal: {f.get('abnormal_cycles')} "
            f"({f.get('fault_rate_pct')}%)",
            f"Mean motor current (RMS): {f.get('mean_current_rms_a')} A; peak {f.get('max_current_peak_a')} A",
            f"Mean cycle time: {f.get('mean_duration_s')} s",
            f"Fault class: {f.get('fault_type')}",
        ]
    elif subsystem == "acv":
        car = f.get("most_likely_faulty_car")
        facts += [
            f"Car flagged as the most likely source: {car if car else 'none'}",
            f"Localisation strength: {f.get('confidence')}",
            "Basis: that car's cabin temperature stays elevated relative to the peer-car median",
        ]
    elif subsystem == "shm":
        facts += [
            f"Vibration RMS: {f.get('vibration_rms_g')} g (peak {f.get('abs_peak_g')} g)",
            f"Fatigue-damage index: {f.get('fatigue_damage_index')}",
            f"Bearing defect probability: {round((f.get('bearing_defect_prob') or 0) * 100)}%",
            f"Most stressed structural node: {f.get('critical_weld_node')}",
        ]
    elif subsystem == "rail":
        facts += [
            f"Estimated corrugation depth: {f.get('depth_microns')} microns",
            f"Wavelength class: {f.get('wavelength_class')}",
            f"Maintenance urgency class: {f.get('maintenance_urgency')}",
        ]
    facts.append(f"Timeframe to state in the suggestion: {_urgency_phrase(subsystem, f)}")
    # Lazy: whatif_engine drags in the whole model stack, which this module should not need to load.
    from backend.services.whatif_engine import ACTION_CATALOG

    repair = ACTION_CATALOG.get(f.get("recommended_action") or "")
    if repair:
        facts.append(f"Repair on offer: {repair['title']} - {repair['technical_description']}")
    return facts


def _rule_based_copy(subsystem: str, f: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic 2-3 sentence card copy, used whenever the model is unavailable."""
    when = _urgency_phrase(subsystem, f)
    status = f.get("status")
    if status == "GOOD":
        return {
            "explanation": ISSUE_HEALTHY[subsystem],
            "suggestion": "No action needed; keep to the routine inspection cycle.",
        }
    if subsystem == "door":
        return {
            "explanation": (
                f"{f.get('abnormal_cycles')} of {f.get('total_cycles')} door cycles drew abnormal "
                f"motor current, peaking at {f.get('max_current_peak_a')} A. That pattern usually means "
                "friction in the guide rails or worn hanger rollers. Left alone it leads to door "
                "faults and delayed departures."
            ),
            "suggestion": (
                f"Clean and lubricate the door guide rails and inspect the hanger rollers and leaf "
                f"alignment {when}, then re-check the motor current."
            ),
        }
    if subsystem == "acv":
        car = f.get("most_likely_faulty_car")
        where = f"Car {car}" if car else "One car"
        return {
            "explanation": (
                f"{where} is running clearly warmer than the rest of the consist, so its cooling is "
                "under-performing. The most likely causes are a slow refrigerant leak or a restricted "
                "filter or duct. Passengers will notice a warm cabin and the pack will overwork."
            ),
            "suggestion": (
                f"Leak-test the refrigerant circuit on {'Car ' + str(car) if car else 'the flagged car'}, "
                f"replace the filter and recharge to specification {when}; re-check supply and return "
                "temperatures afterwards."
            ),
        }
    if subsystem == "shm":
        return {
            "explanation": (
                f"The bogie is vibrating at {f.get('vibration_rms_g')} g RMS with a "
                f"{round((f.get('bearing_defect_prob') or 0) * 100)}% bearing-defect probability and a "
                f"fatigue index of {f.get('fatigue_damage_index')}. That points to a wearing axle-box "
                f"bearing, with stress building at {f.get('critical_weld_node')}. A seized bearing "
                "can force a train withdrawal."
            ),
            "suggestion": (
                f"Run an ultrasonic check of the axle-box bearing and re-grease or replace it {when}, "
                f"and inspect {f.get('critical_weld_node')} for cracking."
            ),
        }
    return {
        "explanation": (
            f"The rail head has worn ripples about {f.get('depth_microns')} microns deep "
            f"({str(f.get('wavelength_class', '')).replace('_', ' ').lower()}). Each ripple makes the wheels "
            "hammer the rail, which shows up as noise and vibration and wears bearings and fasteners faster."
        ),
        "suggestion": f"Schedule a rail-grinding pass over this stretch {when}, then re-measure the depth.",
    }


# Frame group name for each bubble type (the card is fed the finding, not the live frame).
ISSUE_GROUP = {"door": "door", "acv": "acv", "shm": "shm", "rail": "rail_corrugation"}
ISSUE_HEALTHY = {
    "door": "The door motor is drawing normal current and the doors are cycling on time.",
    "acv": "Every car is cooling in step with the rest of the consist.",
    "shm": "Vibration and fatigue at the bogie are within their normal range.",
    "rail": "The rail surface is smooth enough that it is not hammering the wheels.",
}


def rule_based_issue(subsystem: str, frame: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    finding = (frame.get("latest_upload_results") or {}).get(subsystem)
    copy = _rule_based_copy(subsystem, finding) if finding else {
        "explanation": ISSUE_HEALTHY[subsystem],
        "suggestion": "No action needed.",
    }
    return {
        "subsystem": subsystem,
        "status": (finding or {}).get("status", "GOOD"),
        **copy,
        "source": "rule_based",
        "degraded_reason": reason or None,
    }


# ---------------------------------------------------------------------------
class AIInsightService:
    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = OPENAI_MODEL):
        self.model = model
        self.min_interval = AI_INSIGHT_MIN_INTERVAL_SEC
        self._client = None
        self._init_error: Optional[str] = None
        self._cache: Dict[str, Any] = {}
        self._cache_key: Optional[str] = None
        self._last_call_ts = 0.0
        self._issue_cache: Dict[str, Dict[str, Any]] = {}
        self.last_error: Optional[str] = None

        if not api_key:
            self._init_error = "OPENAI_API_KEY is not set in backend/.env"
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, timeout=25.0, max_retries=1)
        except Exception as exc:  # pragma: no cover
            self._init_error = f"OpenAI client unavailable: {exc}"

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def status(self) -> Dict[str, Any]:
        return {
            "ai_enabled": self.enabled,
            "model": self.model if self.enabled else None,
            "min_interval_sec": self.min_interval,
            "init_error": self._init_error,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _cache_signature(frame: Dict[str, Any]) -> str:
        """
        Coarse fingerprint of the operational situation.

        Frames arrive 10x per second but the *explanation* only needs to change
        when the situation does, so the signature deliberately buckets the
        continuous values.
        """
        subs = frame.get("subsystems", {})
        cf = frame.get("counterfactual") or {}
        parts = [
            str(round((frame.get("fleet_health_index") or 0) * 10)),
            (subs.get("door") or {}).get("fault_type", ""),
            (subs.get("acv") or {}).get("fault_type", ""),
            str(round(((subs.get("shm") or {}).get("bearing_defect_prob") or 0) * 5)),
            str(round(((subs.get("shm") or {}).get("vibration_rms_g") or 0) * 2)),
            (subs.get("rail_corrugation") or {}).get("wavelength_class", ""),
            ",".join(sorted(cf.get("active_actions", []))),
        ]
        return "|".join(parts)

    def get_insight(self, frame: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        if not self.enabled:
            return rule_based_insight(frame, self._init_error or "AI disabled")

        signature = self._cache_signature(frame)
        now = time.time()

        # Serve cache when the situation is unchanged, or when the throttle
        # window has not elapsed and we already have something to show.
        if not force and self._cache:
            if signature == self._cache_key:
                return self._cache
            if now - self._last_call_ts < self.min_interval:
                stale = dict(self._cache)
                stale["stale"] = True
                return stale

        briefing = build_frame_summary(frame)
        try:
            content = self._chat(
                SYSTEM_PROMPT,
                f"{briefing}\n\nExplain the current situation for a non-expert.",
                json_mode=True,
            )
            data = json.loads(content)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return rule_based_insight(frame, self.last_error)

        data.setdefault("findings", [])
        data["source"] = "openai"
        data["model"] = self.model
        data["degraded_reason"] = None
        self.last_error = None

        self._cache, self._cache_key, self._last_call_ts = data, signature, now
        return data

    def explain_issue(self, subsystem: str, frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explanation + suggestion for one 3D bubble, written from the uploaded
        finding's own measurements. Cached per finding, so re-opening a card
        (or the 10 Hz stream ticking) never re-calls the model.
        """
        if subsystem not in ISSUE_GROUP:
            return {"error": f"Unknown subsystem '{subsystem}'", "source": "validation"}
        finding = (frame.get("latest_upload_results") or {}).get(subsystem)
        if not finding or not self.enabled:
            return rule_based_issue(subsystem, frame, "" if not finding else (self._init_error or "AI disabled"))

        signature = "|".join(str(finding.get(k, "")) for k in
                             ("file_name", "status", "anomaly_score", "most_likely_faulty_car"))
        cached = self._issue_cache.get(subsystem)
        if cached and cached["sig"] == signature:
            return cached["data"]

        user_msg = (
            f"Subsystem: {SUBSYSTEM_LABELS.get(ISSUE_GROUP[subsystem], subsystem)}\n"
            f"Engineering background: {ISSUE_DOMAIN_NOTES[subsystem]}\n\n"
            "FACTS FROM THE FLAGGED FINDING\n"
            + "\n".join(f"- {line}" for line in _finding_facts(subsystem, finding))
        )
        try:
            parsed = json.loads(self._chat(ISSUE_SYSTEM_PROMPT, user_msg, json_mode=True))
            # Models drift long; one corrective retry keeps the card glanceable.
            if _too_long(parsed):
                parsed = json.loads(self._chat(
                    ISSUE_SYSTEM_PROMPT,
                    user_msg + f"\n\nYour previous draft was too long: {json.dumps(parsed)}\n"
                    f"Rewrite it: explanation at most {EXPLANATION_MAX_WORDS} words, "
                    f"suggestion one sentence of at most {SUGGESTION_MAX_WORDS} words.",
                    json_mode=True,
                ))
            data = {
                "subsystem": subsystem,
                "status": finding.get("status"),
                "explanation": str(parsed["explanation"]).strip(),
                "suggestion": str(parsed["suggestion"]).strip(),
                "source": "openai",
                "model": self.model,
                "degraded_reason": None,
            }
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return rule_based_issue(subsystem, frame, self.last_error)

        self.last_error = None
        self._issue_cache[subsystem] = {"sig": signature, "data": data}
        return data

    def ask(self, question: str, frame: Dict[str, Any]) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"answer": "Please type a question first.", "source": "validation"}
        if len(question) > 500:
            question = question[:500]

        if not self.enabled:
            return {
                "answer": (
                    "The AI assistant is not configured, so I cannot answer free-text "
                    "questions. Set OPENAI_API_KEY in backend/.env to enable it. "
                    "The plain-language tooltips on each metric still work."
                ),
                "source": "rule_based",
                "degraded_reason": self._init_error,
            }

        briefing = build_frame_summary(frame)
        try:
            answer = self._chat(
                ASK_SYSTEM_PROMPT,
                f"{briefing}\n\nQuestion from the operator: {question}",
                json_mode=False,
            )
            self.last_error = None
            return {"answer": answer.strip(), "source": "openai", "model": self.model}
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "answer": (
                    "I could not reach the AI service just now. The readings and their "
                    "plain-language explanations on the dashboard are still live."
                ),
                "source": "rule_based",
                "degraded_reason": self.last_error,
            }

    # ------------------------------------------------------------------
    def _chat(self, system: str, user: str, json_mode: bool) -> str:
        """
        Single call helper.

        Newer reasoning-capable models reject the legacy 'temperature' and
        'max_tokens' arguments, so neither is sent; defaults are appropriate
        for this task.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
