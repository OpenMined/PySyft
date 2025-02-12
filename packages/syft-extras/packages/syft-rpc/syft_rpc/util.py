from __future__ import annotations

import re
from datetime import timedelta


def parse_duration(duration: str) -> timedelta:
    """Convert duration strings like '1h', '3d', '24h', '30s' into timedelta."""
    pattern = r"(\d+)([dhms])"  # Matches number + unit (d, h, m, s)
    match = re.fullmatch(pattern, duration.strip().lower())

    if not match:
        raise ValueError("Invalid duration format. Use 'Nd', 'Nh', 'Nm', or 'Ns'.")

    value, unit = int(match.group(1)), match.group(2)

    if unit == "d":
        return timedelta(days=value)
    elif unit == "h":
        return timedelta(hours=value)
    elif unit == "m":
        return timedelta(minutes=value)
    elif unit == "s":
        return timedelta(seconds=value)

    return timedelta()  # Default case (should never reach)
