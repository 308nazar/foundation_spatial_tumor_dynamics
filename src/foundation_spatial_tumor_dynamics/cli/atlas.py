"""Cell-level atlas query and export helpers."""
import json
from pathlib import Path
from typing import Any

def export_records(records: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(records, indent=2, default=str))

def query_records(records: list[dict[str, Any]], cell_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    if cell_type is None:
        return records[:limit]
    return [record for record in records if record.get("cell_type") == cell_type][:limit]
