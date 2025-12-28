from __future__ import annotations

from typing import Any, List, Dict


def fmt_dt(v: Any) -> str:
    if v is None or v == "":
        return "—"
    return str(v)


def fmt_reason_codes(rc: Any, max_items: int = 6) -> str:
    if isinstance(rc, list) and rc:
        return ", ".join(map(str, rc[:max_items]))
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    return "—"


def fmt_reason_details(details: Any) -> List[str]:
    # expects list[{code,message}] but tolerates anything
    out: List[str] = []
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict):
                code = d.get("code", "")
                msg = d.get("message", "")
                out.append(f"{code}: {msg}".strip(": ").strip())
            else:
                out.append(str(d))
    return out
