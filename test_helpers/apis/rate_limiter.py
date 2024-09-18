def is_within_rate_limit(context) -> bool:
    """Rate limiter for custom API calls made by users."""
    # stdlib
    import datetime

    state = context.state
    settings = context.settings
    email = context.user.email

    current_time = datetime.datetime.now()
    calls_last_min = [
        1 if (current_time - call_time).seconds < 60 else 0
        for call_time in state[email]
    ]

    return sum(calls_last_min) < settings.get("calls_per_min", 5)
