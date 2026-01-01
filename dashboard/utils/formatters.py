from __future__ import annotations

from typing import Any, List


def fmt_reason_codes(rc: Any, max_items: int = 6) -> str:
    if isinstance(rc, list) and rc:
        return ", ".join(map(str, rc[:max_items]))
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    return "—"


def fmt_reason_details(details: Any) -> List[str]:
    out: List[str] = []
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict):
                code = d.get("code", "")
                msg = d.get("message", "")
                line = f"{code}: {msg}".strip(": ").strip()
                out.append(line)
            else:
                out.append(str(d))
    return out
