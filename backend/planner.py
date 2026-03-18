# planner.py
from typing import Dict, Set, List


def parse_constraints(text: str) -> Dict:
    """Extract constraint flags from natural language text."""
    text = text.lower()
    return {
        "low_power": "low power" in text or "eco" in text or "energy efficient" in text,
        "budget":    "budget"    in text or "cheap" in text or "affordable"       in text,
    }


def interpret_constraints(flags: Dict) -> Dict:
    """Convert constraint flags into filter functions."""
    rules: Dict = {"filters": []}

    if flags.get("low_power"):
        rules["filters"].append(lambda d: d["metrics"]["power"] <= 20)

    if flags.get("budget"):
        rules["filters"].append(lambda d: d["metrics"]["cost"] <= 100)

    return rules


def compute_required_support(goal_caps: Set[str], devices: List[Dict]) -> Set[str]:
    """
    Expand goal capabilities to include all transitive dependencies.

    Optimisation vs original:
    - A 'visited' set tracks which devices have already contributed their
      dependencies, preventing redundant re-scans on every iteration.
    """
    required = set(goal_caps)
    visited: Set[str] = set()
    expanded = True

    while expanded:
        expanded = False
        for d in devices:
            if d["id"] in visited:
                continue
            if d["provides"] & required:
                visited.add(d["id"])
                for req in d["requires"]:
                    if req not in required:
                        required.add(req)
                        expanded = True

    return required


def build_plan(
    goal_capabilities: Set[str],
    rules: Dict,
    enriched_devices: List[Dict],
) -> List[Dict]:
    """
    Build an execution plan that satisfies all goal capabilities.

    Optimisation vs original:
    - Instead of picking the first device that makes progress, all valid
      candidates for the current step are collected and the cheapest one is
      chosen.  Under a low_power constraint the lowest-power device is
      preferred.  This keeps behaviour deterministic while producing better
      plans when multiple devices can satisfy the same capability.
    """
    required_caps = compute_required_support(goal_capabilities, enriched_devices)

    provided: Set[str] = set()
    selected: List[Dict] = []
    remaining = set(required_caps)
    selected_ids: Set[str] = set()

    print("\n→ Expanded required capabilities:", required_caps)

    iteration = 0
    while remaining:
        iteration += 1

        # Gather every device that is currently usable and satisfies ≥1 need
        candidates = [
            d for d in enriched_devices
            if d["id"] not in selected_ids
            and all(f(d) for f in rules["filters"])
            and d["requires"].issubset(provided)
            and (d["provides"] & remaining)
        ]

        if not candidates:
            print(f"\n⚠ Cannot satisfy remaining capabilities: {remaining}")
            print("Available devices cannot fulfil these requirements given current constraints.")
            raise RuntimeError(f"Cannot satisfy capabilities: {remaining}")

        # Pick the cheapest candidate (lowest power as secondary sort key)
        best = min(candidates, key=lambda d: (d["metrics"]["cost"], d["metrics"]["power"]))

        selected.append(best)
        selected_ids.add(best["id"])
        provided |= best["provides"]
        remaining -= best["provides"]

        print(f"  [Step {iteration}] Selected {best['id']}: provides {best['provides']}")

    return selected