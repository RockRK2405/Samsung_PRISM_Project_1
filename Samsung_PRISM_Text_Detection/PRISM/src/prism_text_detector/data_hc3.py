from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd


SYNTHETIC_LABELS = {"1", "ai", "chatgpt", "gpt", "llm", "machine", "synthetic", "generated", "fake"}
HUMAN_LABELS = {"0", "human", "real", "organic", "authentic"}


def load_hc3(path_or_paths: str | Path | Sequence[str | Path], min_chars: int = 30) -> pd.DataFrame:
    """Load HC3-style files into rows with columns: text, label, source, record_id.

    Label convention:
    - 0 = human / real
    - 1 = synthetic / AI-generated

    Supported inputs:
    - A file or folder
    - .json, .jsonl, .csv
    - Original HC3-style records with human_answers and chatgpt_answers
    - Already-normalized datasets with text and label columns
    """
    paths = _resolve_paths(path_or_paths)
    rows: list[dict[str, object]] = []
    for path in paths:
        for record_id, record in enumerate(_read_records(path)):
            rows.extend(_record_to_rows(record, source=path.name, record_id=record_id))

    if not rows:
        raise ValueError(f"No usable HC3 rows found under: {path_or_paths}")

    frame = pd.DataFrame(rows)
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() >= min_chars]
    frame = frame.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    labels = sorted(frame["label"].unique().tolist())
    if labels != [0, 1]:
        raise ValueError(f"Expected both human label 0 and synthetic label 1, found labels={labels}")

    return frame


def balanced_sample(frame: pd.DataFrame, sample_per_label: int | None, random_state: int = 42) -> pd.DataFrame:
    if not sample_per_label:
        return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    pieces = []
    for label, group in frame.groupby("label"):
        take = min(sample_per_label, len(group))
        pieces.append(group.sample(n=take, random_state=random_state))
    return pd.concat(pieces).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _resolve_paths(path_or_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        candidates = [Path(path_or_paths)]
    else:
        candidates = [Path(path) for path in path_or_paths]

    paths: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            paths.extend(sorted(p for p in candidate.rglob("*") if p.suffix.lower() in {".json", ".jsonl", ".csv"}))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise FileNotFoundError(candidate)
    return paths


def _read_records(path: Path) -> Iterator[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from _flatten_json_payload(payload)
        return

    if suffix == ".csv":
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            yield record
        return

    raise ValueError(f"Unsupported file type: {path}")


def _flatten_json_payload(payload: object) -> Iterator[dict[str, object]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        if _looks_like_record(payload):
            yield payload
            return

        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(value, dict):
                yield from _flatten_json_payload(value)


def _looks_like_record(payload: dict[str, object]) -> bool:
    keys = {key.lower() for key in payload}
    return bool(keys & {"text", "answer", "human_answers", "chatgpt_answers", "human_answer", "chatgpt_answer"})


def _record_to_rows(record: dict[str, object], source: str, record_id: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    normalized = {str(key).lower().strip(): value for key, value in record.items()}

    if "text" in normalized and "label" in normalized:
        label = _normalize_label(normalized["label"])
        if label is not None:
            rows.append(_row(normalized["text"], label, source, record_id))
        return rows

    for key in ("human_answers", "human_answer", "human", "real_answer", "real"):
        for text in _as_text_list(normalized.get(key)):
            rows.append(_row(text, 0, source, record_id))

    for key in ("chatgpt_answers", "chatgpt_answer", "gpt_answer", "synthetic_answer", "ai_answer", "generated"):
        for text in _as_text_list(normalized.get(key)):
            rows.append(_row(text, 1, source, record_id))

    return rows


def _row(text: object, label: int, source: str, record_id: int) -> dict[str, object]:
    return {
        "text": str(text),
        "label": label,
        "source": source,
        "record_id": record_id,
    }


def _normalize_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)

    label = str(value).strip().lower()
    if label in HUMAN_LABELS:
        return 0
    if label in SYNTHETIC_LABELS:
        return 1
    return None


def _as_text_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str):
        parsed = _parse_possible_list(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
        if value.strip():
            return [value]

    return [str(value)]


def _parse_possible_list(value: str) -> object:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{\"'":
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            continue
    return value

