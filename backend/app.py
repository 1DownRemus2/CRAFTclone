# app.py
import time
from sentence_transformers import SentenceTransformer

from intent_engine import IntentEngine
from capability_intent_engine import CapabilityIntentEngine
from semantic_bundle_detector import SemanticBundleIntentDetector
from goal_merge_strategy import GoalMergeStrategy
from capability_inference import CapabilityInferenceEngine
from planner import parse_constraints, interpret_constraints, build_plan
from config import EMBEDDING_MODEL_NAME
from devices import DEVICE_REGISTRY

# ── One-time initialisation ────────────────────────────────────────────────────
# The model is loaded ONCE here and injected into every engine that needs it.
# Previously each engine loaded its own copy, multiplying cold-start time by 4.
print("Loading shared embedding model …")
_t0 = time.time()
_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Model loaded in {time.time() - _t0:.2f}s\n")

# Engines are instantiated ONCE and reused across all run() calls.
# Previously they were recreated (including all embedding builds) per call.
_capability_intent_engine = CapabilityIntentEngine(model=_MODEL)
_domain_intent_engine     = IntentEngine(model=_MODEL)
_bundle_detector          = SemanticBundleIntentDetector(model=_MODEL)
_goal_merger              = GoalMergeStrategy()

# The device registry never changes at runtime, so enrichment runs once.
# Previously enrich_registry() re-encoded all 15 devices on every call.
print("Enriching device registry …")
_t1 = time.time()
_inference_engine  = CapabilityInferenceEngine(model=_MODEL)
_ENRICHED_DEVICES  = _inference_engine.enrich_registry(DEVICE_REGISTRY)
print(f"Registry enriched in {time.time() - _t1:.2f}s\n")
# ──────────────────────────────────────────────────────────────────────────────


def run(query: str) -> None:
    """
    Execute the full intent → plan pipeline for a single user query.

    Pipeline steps
    ──────────────
    1. Extract constraints (low_power, budget keywords)
    2. Detect capability intent (what features the user explicitly mentioned)
    3. Detect domain intent (SMART_HOME / HOME_OFFICE / unknown)
    4. Detect bundle intent (full setup vs single targeted item)
    5. Merge goals into final capability set
    6. Build execution plan from the pre-enriched device registry
    """
    print("\n" + "=" * 80)
    print("USER QUERY:", query)
    print("=" * 80)

    # Step 1 ── Constraint extraction
    print("\n[1] CONSTRAINT EXTRACTION")
    flags = parse_constraints(query)
    rules = interpret_constraints(flags)
    print(f"  Detected constraints: {flags}")

    # Step 2 ── Capability intent
    print("\n[2] CAPABILITY INTENT DETECTION")
    focused_capabilities = _capability_intent_engine.detect(query)
    if not focused_capabilities:
        print("  No explicit capability mentions detected")

    # Step 3 ── Domain intent
    print("\n[3] DOMAIN INTENT DETECTION")
    domain, domain_confidence = _domain_intent_engine.detect(query)
    print(f"  Detected domain: {domain}")
    print(f"  Confidence: {domain_confidence:.3f}")

    # Step 4 ── Bundle intent
    print("\n[4] BUNDLE INTENT DETECTION (Semantic)")
    is_bundle, bundle_reason = _bundle_detector.detect(query, domain, domain_confidence)
    print(f"  Bundle intent: {is_bundle}")
    print(f"  Reason: {bundle_reason}")

    # Step 5 ── Goal merge
    print("\n[5] GOAL MERGE STRATEGY")
    try:
        goal_capabilities = _goal_merger.merge(
            is_bundle=is_bundle,
            domain=domain,
            focused_capabilities=focused_capabilities,
        )
    except ValueError as e:
        print(f"\n✗ GOAL DETERMINATION FAILED: {e}")
        print("=" * 80)
        return

    # Step 6 ── Build plan  (registry is already enriched at startup)
    print("\n[6] BUILDING EXECUTION PLAN")
    try:
        devices = build_plan(goal_capabilities, rules, _ENRICHED_DEVICES)
    except RuntimeError as e:
        print(f"\n✗ PLANNING FAILED: {e}")
        print("=" * 80)
        return

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("✓ EXECUTION PLAN GENERATED")
    print("=" * 80)
    print(f"Intent: {_goal_merger.explain(is_bundle, domain, focused_capabilities)}")
    print("=" * 80)

    total_cost  = sum(d["metrics"]["cost"]  for d in devices)
    total_power = sum(d["metrics"]["power"] for d in devices)

    for i, d in enumerate(devices, 1):
        # originally_empty is set by enrich_device() — no hard-coded ID list needed
        inferred_marker = " [INFERRED]" if d.get("originally_empty") else ""
        print(f"\n{i}. {d['id']}{inferred_marker}")
        print(f"   Description: {d['description']}")
        print(f"   Provides: {d['provides']}")
        print(f"   Requires: {d['requires'] if d['requires'] else 'None'}")
        print(f"   Power: {d['metrics']['power']}W | Cost: ${d['metrics']['cost']}")

    print(f"\n{'─' * 80}")
    print(f"TOTAL: {len(devices)} devices | ${total_cost} | {total_power}W")
    print("=" * 80)


def run_test_suite() -> None:
    """Run the hardcoded test suite (invoke with: python app.py --test)."""
    print("\n" + "█" * 80)
    print("OPTIMISED INTENT PIPELINE — TEST SUITE")
    print("█" * 80)

    suite_start = time.time()

    run("I want to build a smart home with good quality motion sensors")
    run("I only want motion sensors")
    run("Setup smart home")
    run("I need a curved monitor")
    run("Create a complete home office workspace")
    run("I just need a desk")
    run("home office")

    print(f"\nAll queries completed in {time.time() - suite_start:.2f}s")


def run_interactive() -> None:
    """
    Interactive mode — reads queries from the user one at a time.
    Type 'quit' or 'exit' or press Ctrl+C to stop.
    """
    print("\n" + "█" * 80)
    print("SMART HOME / OFFICE PLANNER  —  Interactive Mode")
    print("█" * 80)
    print("\nDescribe what you want to set up and I'll build a device plan for you.")
    print("Examples:")
    print("  • I want to build a smart home")
    print("  • I only need motion sensors")
    print("  • Create a complete home office")
    print("  • I just need a desk")
    print("\nType 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not query:
            print("  (Please enter a query or type 'quit' to exit.)\n")
            continue

        if query.lower() in {"quit", "exit", "q", "bye"}:
            print("Goodbye!")
            break

        run(query)
        print()  # blank line between queries for readability


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        run_test_suite()
    else:
        run_interactive()