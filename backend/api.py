# api.py
# Flask REST API that wraps the intent pipeline for the frontend.
# Run with: python api.py
# Listens on http://localhost:5000

import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

from intent_engine import IntentEngine
from capability_intent_engine import CapabilityIntentEngine
from semantic_bundle_detector import SemanticBundleIntentDetector
from goal_merge_strategy import GoalMergeStrategy
from capability_inference import CapabilityInferenceEngine
from planner import parse_constraints, interpret_constraints, build_plan
from config import EMBEDDING_MODEL_NAME
from devices import DEVICE_REGISTRY

# Serve index.html and any static files from the same folder as api.py
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ── One-time initialisation at server startup ──────────────────────────────────
print("Loading shared embedding model …")
_t0 = time.time()
_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Model loaded in {time.time() - _t0:.2f}s")

_capability_intent_engine = CapabilityIntentEngine(model=_MODEL)
_domain_intent_engine     = IntentEngine(model=_MODEL)
_bundle_detector          = SemanticBundleIntentDetector(model=_MODEL)
_goal_merger              = GoalMergeStrategy()

print("Enriching device registry …")
_t1 = time.time()
_inference_engine = CapabilityInferenceEngine(model=_MODEL)
_ENRICHED_DEVICES = _inference_engine.enrich_registry(DEVICE_REGISTRY)
print(f"Registry enriched in {time.time() - _t1:.2f}s")
print("API ready.\n")
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/plan", methods=["POST"])
def plan():
    """
    POST /api/plan
    Body: { "query": "I want to build a smart home" }

    Returns a JSON object with the full pipeline result:
    {
      "query": "...",
      "intent": "...",
      "domain": "SMART_HOME" | "HOME_OFFICE" | "unknown",
      "domain_confidence": 0.97,
      "is_bundle": true,
      "bundle_reason": "...",
      "focused_capabilities": ["motion_detection"],
      "goal_capabilities": ["lighting", "voice_control", "motion_detection"],
      "constraints": { "low_power": false, "budget": false },
      "devices": [
        {
          "id": "hub_zigbee",
          "description": "...",
          "provides": ["zigbee_network", "wifi_network"],
          "requires": [],
          "power": 5,
          "cost": 80,
          "inferred": false
        },
        ...
      ],
      "total_cost": 167,
      "total_power": 22,
      "error": null
    }
    """
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Step 1 — Constraints
    flags = parse_constraints(query)
    rules = interpret_constraints(flags)

    # Step 2 — Capability intent
    focused_capabilities = _capability_intent_engine.detect(query)

    # Step 3 — Domain intent
    domain, domain_confidence = _domain_intent_engine.detect(query)

    # Step 4 — Bundle intent
    is_bundle, bundle_reason = _bundle_detector.detect(query, domain, domain_confidence)

    # Step 5 — Goal merge
    try:
        goal_capabilities = _goal_merger.merge(
            is_bundle=is_bundle,
            domain=domain,
            focused_capabilities=focused_capabilities,
        )
    except ValueError as e:
        return jsonify({
            "query": query,
            "error": str(e),
            "domain": domain,
            "domain_confidence": round(domain_confidence, 3),
            "is_bundle": is_bundle,
            "bundle_reason": bundle_reason,
            "focused_capabilities": list(focused_capabilities),
            "goal_capabilities": [],
            "constraints": flags,
            "devices": [],
            "total_cost": 0,
            "total_power": 0,
            "intent": "Unable to determine intent",
        }), 200

    # Step 6 — Build plan
    try:
        devices = build_plan(goal_capabilities, rules, _ENRICHED_DEVICES)
    except RuntimeError as e:
        return jsonify({
            "query": query,
            "error": str(e),
            "domain": domain,
            "domain_confidence": round(domain_confidence, 3),
            "is_bundle": is_bundle,
            "bundle_reason": bundle_reason,
            "focused_capabilities": list(focused_capabilities),
            "goal_capabilities": list(goal_capabilities),
            "constraints": flags,
            "devices": [],
            "total_cost": 0,
            "total_power": 0,
            "intent": _goal_merger.explain(is_bundle, domain, focused_capabilities),
        }), 200

    # Serialise devices (sets → lists for JSON)
    serialised = []
    for d in devices:
        serialised.append({
            "id": d["id"],
            "description": d["description"],
            "provides": sorted(list(d["provides"])),
            "requires": sorted(list(d["requires"])),
            "power": d["metrics"]["power"],
            "cost": d["metrics"]["cost"],
            "inferred": bool(d.get("originally_empty", False)),
            "tags": d.get("tags", []),
        })

    return jsonify({
        "query": query,
        "error": None,
        "intent": _goal_merger.explain(is_bundle, domain, focused_capabilities),
        "domain": domain,
        "domain_confidence": round(domain_confidence, 3),
        "is_bundle": is_bundle,
        "bundle_reason": bundle_reason,
        "focused_capabilities": sorted(list(focused_capabilities)),
        "goal_capabilities": sorted(list(goal_capabilities)),
        "constraints": flags,
        "devices": serialised,
        "total_cost": sum(d["metrics"]["cost"] for d in devices),
        "total_power": sum(d["metrics"]["power"] for d in devices),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)