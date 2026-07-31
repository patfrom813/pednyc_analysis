"""Central scenario-condition metadata shared by trajectory analyses."""

from __future__ import annotations


SCENARIO_DESCRIPTIONS = {
    3: "Orthogonal approach — pedestrian near vehicle lane, same side",
    7: "Orthogonal approach — pedestrian near vehicle lane, opposite side",
    12: "Orthogonal approach — pedestrian far from vehicle lane, opposite side",
    15: (
        "Parallel/contra-directional approach — "
        "pedestrian near vehicle lane, opposite side"
    ),
    16: (
        "Parallel/contra-directional approach — "
        "pedestrian far from vehicle lane, opposite side"
    ),
    21: (
        "Parallel/contra-directional approach — "
        "pedestrian near vehicle lane, same side"
    ),
}


def scenario_description(scenario: int) -> str:
    """Return the configured condition label or fail clearly."""
    try:
        return SCENARIO_DESCRIPTIONS[int(scenario)]
    except KeyError as error:
        raise ValueError(
            f"No condition description configured for scenario {scenario}."
        ) from error
