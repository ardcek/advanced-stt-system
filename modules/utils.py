from datetime import datetime

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def format_ts(seconds: float) -> str:
    if seconds is None:
        return "--:--"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
