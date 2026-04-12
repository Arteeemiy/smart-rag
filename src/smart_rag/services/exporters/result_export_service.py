from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...config import Settings
from ...schemas.requests import ExportRequest

logger = logging.getLogger(__name__)


class ResultExportService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._export_dir = Path(settings.export_dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def export_run_result(self, run_result: dict[str, Any], export: ExportRequest | None) -> dict[str, Any] | None:
        if export is None or export.format == "none":
            return None
        if export.format == "json":
            return self._export_json(run_result=run_result, filename_prefix=export.filename_prefix)
        if export.format == "google_sheets":
            return self._export_google_sheets(run_result=run_result, export=export)
        raise ValueError(f"Unsupported export format: {export.format}")

    def build_response_payload(
        self,
        run_result: dict[str, Any],
        response_mode: str = "summary",
        inline_result_limit: int = 20,
        export_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if response_mode == "full":
            payload = dict(run_result)
            if export_info is not None:
                payload["export"] = export_info
            return payload

        results = list(run_result.get("results") or [])
        preview = results[:inline_result_limit]
        return {
            "summary": run_result.get("summary"),
            "effective_params": run_result.get("effective_params"),
            "results_preview": preview,
            "preview_count": len(preview),
            "remaining_results_count": max(len(results) - len(preview), 0),
            "export": export_info,
        }

    def _export_json(self, run_result: dict[str, Any], filename_prefix: str | None) -> dict[str, Any]:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        prefix = self._sanitize_filename(filename_prefix or "rag_run")
        file_path = self._export_dir / f"{prefix}_{timestamp}.json"
        file_path.write_text(json.dumps(run_result, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "format": "json",
            "file_path": str(file_path),
            "filename": file_path.name,
        }

    def _export_google_sheets(self, run_result: dict[str, Any], export: ExportRequest) -> dict[str, Any]:
        google_config = export.google_sheets
        if google_config is None:
            raise ValueError("google_sheets config is required")

        try:
            import gspread
        except ImportError as exc:
            raise RuntimeError("gspread is not installed; add it to the environment to use Google Sheets export") from exc

        credentials_info = self._resolve_google_credentials(google_config.credentials_json)
        if credentials_info is None:
            raise ValueError(
                "Google Sheets credentials were not provided. Supply export.google_sheets.credentials_json "
                "or configure GOOGLE_SHEETS_CREDENTIALS_FILE."
            )

        client = gspread.service_account_from_dict(credentials_info)
        if google_config.spreadsheet_id:
            spreadsheet = client.open_by_key(google_config.spreadsheet_id)
            created = False
        else:
            title_prefix = export.filename_prefix or google_config.spreadsheet_title or "smart-rag-run"
            title = f"{title_prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            spreadsheet = client.create(title)
            created = True

        if google_config.share_with_email:
            spreadsheet.share(google_config.share_with_email, perm_type="user", role="writer", notify=False)

        summary_sheet = self._get_or_create_worksheet(spreadsheet, google_config.summary_worksheet_name or "summary")
        results_sheet = self._get_or_create_worksheet(spreadsheet, google_config.worksheet_name or "results")

        summary_rows = self._build_summary_rows(run_result)
        detail_rows = self._build_detail_rows(run_result)
        summary_sheet.clear()
        results_sheet.clear()
        if summary_rows:
            summary_sheet.update("A1", summary_rows)
        if detail_rows:
            results_sheet.update("A1", detail_rows)

        return {
            "format": "google_sheets",
            "spreadsheet_id": spreadsheet.id,
            "spreadsheet_title": spreadsheet.title,
            "spreadsheet_url": getattr(spreadsheet, "url", None),
            "created": created,
            "worksheets": {
                "summary": summary_sheet.title,
                "results": results_sheet.title,
            },
        }

    def _resolve_google_credentials(self, credentials_json: str | None) -> dict[str, Any] | None:
        raw_json = credentials_json or self._read_credentials_file()
        if not raw_json:
            return None
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Google Sheets credentials_json must be valid JSON") from exc

    def _read_credentials_file(self) -> str | None:
        credentials_file = self._settings.google_sheets_credentials_file
        if not credentials_file:
            return None
        path = Path(credentials_file)
        if not path.exists():
            raise ValueError(f"Configured GOOGLE_SHEETS_CREDENTIALS_FILE does not exist: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _get_or_create_worksheet(spreadsheet: Any, title: str) -> Any:
        try:
            return spreadsheet.worksheet(title)
        except Exception:
            return spreadsheet.add_worksheet(title=title, rows=1000, cols=30)

    @staticmethod
    def _build_summary_rows(run_result: dict[str, Any]) -> list[list[Any]]:
        summary = run_result.get("summary") or {}
        effective_params = run_result.get("effective_params") or {}
        rows: list[list[Any]] = [["key", "value"]]
        for key, value in summary.items():
            rows.append([f"summary.{key}", value])
        for key, value in effective_params.items():
            rows.append([f"effective_params.{key}", json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
        return rows

    @staticmethod
    def _build_detail_rows(run_result: dict[str, Any]) -> list[list[Any]]:
        results = run_result.get("results") or []
        headers = [
            "question_id",
            "query",
            "answer",
            "has_context",
            "error",
            "rag_mode",
            "retrieval_scope",
            "sources_json",
            "retrieval_diagnostics_json",
        ]
        rows: list[list[Any]] = [headers]
        for item in results:
            diagnostics = item.get("retrieval_diagnostics") or {}
            rows.append(
                [
                    item.get("question_id"),
                    item.get("query"),
                    item.get("answer"),
                    item.get("has_context"),
                    item.get("error"),
                    item.get("rag_mode"),
                    diagnostics.get("retrieval_scope") or (run_result.get("effective_params") or {}).get("retrieval_scope"),
                    json.dumps(item.get("sources"), ensure_ascii=False),
                    json.dumps(diagnostics, ensure_ascii=False),
                ]
            )
        return rows

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "rag_run"
