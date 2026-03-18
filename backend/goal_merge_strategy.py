# goal_merge_strategy.py
from typing import Set
from domains import DOMAINS


class GoalMergeStrategy:
    """
    Merges bundle intent and focused capabilities into final goal capabilities.

    Logic:
      Case 1 — bundle + known domain : domain defaults ∪ focused_capabilities
      Case 2 — focused capabilities only : focused_capabilities
      Case 3 — known domain, no bundle : domain defaults
      Case 4 — nothing determinable   : raise ValueError

    Optimisation vs original:
    - _get_domain_caps() extracts the repeated DOMAINS lookup so both
      merge() and explain() share a single source of truth.
    """

    @staticmethod
    def _get_domain_caps(domain: str) -> Set[str]:
        """Return the set of capability names for a known domain."""
        return set(DOMAINS[domain]["goal_capabilities"].keys())

    def merge(
        self,
        is_bundle: bool,
        domain: str,
        focused_capabilities: Set[str],
    ) -> Set[str]:
        """
        Merge intents into final goal capabilities.

        Args:
            is_bundle:            Whether the user wants a full system.
            domain:               Detected domain name, or "unknown".
            focused_capabilities: Explicitly mentioned capabilities.

        Returns:
            Set of goal capability names to pass to the planner.

        Raises:
            ValueError: If no valid goals can be determined.
        """
        # Case 1: bundle + known domain
        if is_bundle and domain != "unknown":
            domain_caps = self._get_domain_caps(domain)
            merged = domain_caps | focused_capabilities
            print(f"  ✓ Bundle + Domain: {domain} defaults + focused capabilities")
            print(f"    Domain defaults: {domain_caps}")
            print(f"    Focused: {focused_capabilities if focused_capabilities else 'none'}")
            print(f"    Merged goals: {merged}")
            return merged

        # Case 2: only focused capabilities
        if focused_capabilities:
            print("  ✓ Focused intent: using explicit capabilities only")
            print(f"    Goals: {focused_capabilities}")
            return focused_capabilities

        # Case 3: domain known, not a bundle request
        if domain != "unknown":
            domain_caps = self._get_domain_caps(domain)
            print(f"  ✓ Domain fallback: using {domain} defaults")
            print(f"    Goals: {domain_caps}")
            return domain_caps

        # Case 4: cannot determine intent
        raise ValueError(
            "Cannot determine goals: no bundle intent, no focused capabilities, "
            "and no valid domain detected. Try being more specific about what you want."
        )

    def explain(
        self,
        is_bundle: bool,
        domain: str,
        focused_capabilities: Set[str],
    ) -> str:
        """Generate a human-readable summary of the merge decision."""
        if is_bundle and domain != "unknown":
            emphasis = focused_capabilities if focused_capabilities else "all features"
            return f"Full {domain} system with emphasis on: {emphasis}"
        if focused_capabilities:
            return f"Targeted solution for: {', '.join(focused_capabilities)}"
        if domain != "unknown":
            return f"Standard {domain} configuration"
        return "Unable to determine intent"