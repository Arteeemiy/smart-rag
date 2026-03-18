import json
from pathlib import Path
from typing import Any


class DocumentRepository:
    def load_json_documents(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("DATA_FILE must contain a JSON list")
        return data
