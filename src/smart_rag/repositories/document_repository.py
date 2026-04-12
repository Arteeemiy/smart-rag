import json
from pathlib import Path
from typing import Any


class DocumentRepository:
    def load_json_documents(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return data["records"]
        raise ValueError("DATA_FILE must contain either a JSON list or an object with a 'records' list")
