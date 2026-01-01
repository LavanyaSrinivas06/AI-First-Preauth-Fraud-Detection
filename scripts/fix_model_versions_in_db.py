from __future__ import annotations

import re
import sqlite3
from pathlib import Path

DB = Path("artifacts/stores/inference_store.sqlite")
ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def main():
    if not DB.exists():
        raise SystemExit(f"DB not found: {DB}")

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Fix NULL/empty
    cur.execute("UPDATE decisions SET model_version='xgb-v1' WHERE model_version IS NULL OR TRIM(model_version)='';")
    cur.execute("UPDATE reviews   SET model_version='xgb-v1' WHERE model_version IS NULL OR TRIM(model_version)='';")

    # Fix timestamp-like values
    cur.execute("SELECT id, model_version FROM decisions WHERE model_version IS NOT NULL;")
    bad_ids = []
    for _id, mv in cur.fetchall():
        if mv and ISO_TS_RE.match(str(mv).strip()):
            bad_ids.append(_id)

    for _id in bad_ids:
        cur.execute("UPDATE decisions SET model_version='xgb-v1' WHERE id=?;", (_id,))

    con.commit()

    print(f"[OK] decisions fixed timestamp-like rows: {len(bad_ids)}")
    con.close()


if __name__ == "__main__":
    main()
