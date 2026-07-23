"""Claude Batch API pipeline: build requests, submit, poll, and collect labels.

Submit and collect are separate CLI invocations hours apart, so all state lives on
disk under ``data/autolabel/`` (see the module-level layout note in the docs). The
``BatchTransport`` protocol is the offline seam: tests inject a fake transport, and
only :class:`AnthropicBatchTransport` ever imports the SDK (lazily).
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pydantic import ValidationError

from nuscenes_data_engine.config import get_settings, load_yaml
from nuscenes_data_engine.data_engine.autolabel.schema import (
    COUNT_FIELDS,
    SceneLabel,
    structured_output_schema,
)

logger = logging.getLogger("nuscenes_data_engine")

SYSTEM_PROMPT = (
    "You label single front-camera images from a driving dataset. Report only what is "
    "clearly visible in this image. Count object instances you can positively identify; "
    "do not guess at heavily occluded, cut-off, or very distant objects. Counts are for "
    "this single frame only."
)

# $/MTok (input, output) at standard rates; the Batch API bills at 50%.
_PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-opus-4-8": (5.0, 25.0)}
_EST_INPUT_TOKENS = 2300  # ~1850 image + prompt & schema overhead
_EST_OUTPUT_TOKENS = 350

RETRYABLE_STATUSES = {"errored_server", "expired", "canceled"}
TERMINAL_STATUSES = {
    "ok",
    "refusal",
    "truncated",
    "invalid_json",
    "schema_invalid",
    "errored_bad_request",
}


class BatchTransport(Protocol):
    """The minimal Batch API surface the pipeline needs (fake-able in tests)."""

    def submit(self, requests: list[dict[str, Any]]) -> str: ...

    def status(self, batch_id: str) -> tuple[str, dict[str, int]]: ...

    def results(self, batch_id: str) -> Iterator[dict[str, Any]]: ...


class AnthropicBatchTransport:
    """Real transport over the anthropic SDK (imported lazily)."""

    def __init__(self, api_key: str | None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def submit(self, requests: list[dict[str, Any]]) -> str:
        from typing import cast

        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        batch = self._client.messages.batches.create(
            requests=[
                Request(
                    custom_id=req["custom_id"],
                    # Params are built as plain dicts so the pipeline stays SDK-free;
                    # build_request produces exactly the TypedDict's shape.
                    params=cast(MessageCreateParamsNonStreaming, req["params"]),
                )
                for req in requests
            ]
        )
        return str(batch.id)

    def status(self, batch_id: str) -> tuple[str, dict[str, int]]:
        batch = self._client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        return str(batch.processing_status), {
            "processing": counts.processing,
            "succeeded": counts.succeeded,
            "errored": counts.errored,
            "canceled": counts.canceled,
            "expired": counts.expired,
        }

    def results(self, batch_id: str) -> Iterator[dict[str, Any]]:
        for result in self._client.messages.batches.results(batch_id):
            record: dict[str, Any] = {
                "custom_id": result.custom_id,
                "result_type": result.result.type,
                "stop_reason": None,
                "text": None,
                "error_type": None,
            }
            if result.result.type == "succeeded":
                message = result.result.message
                record["stop_reason"] = message.stop_reason
                record["text"] = next(
                    (block.text for block in message.content if block.type == "text"), None
                )
            elif result.result.type == "errored":
                record["error_type"] = result.result.error.error.type
            yield record


def to_openai_payload(params: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an Anthropic-shaped request into an OpenAI-compatible vLLM payload."""
    content: list[dict[str, Any]] = []
    for block in params["messages"][0]["content"]:
        if block["type"] == "image":
            source = block["source"]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"},
                }
            )
        else:
            content.append({"type": "text", "text": block["text"]})
    schema = params["output_config"]["format"]["schema"]
    return {
        "model": model,
        "max_tokens": params["max_tokens"],
        "messages": [
            {"role": "system", "content": params["system"]},
            {"role": "user", "content": content},
        ],
        # vLLM structured outputs (xgrammar): same sanitized schema as the Claude path.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "scene_label", "schema": schema},
        },
    }


class LocalVLMTransport:
    """Transport against a self-hosted OpenAI-compatible vLLM server (free path).

    There is no server-side batch queue: ``submit()`` executes the requests
    immediately with a thread pool and persists the normalized results to
    ``results_dir/{batch_id}.jsonl`` — the same artifact the Batch API path downloads
    later — so status/collect/retry work unchanged.
    """

    def __init__(self, base_url: str, model: str, results_dir: Path, concurrency: int = 8) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._results_dir = results_dir
        self._concurrency = concurrency

    def _call(self, client: Any, request: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "custom_id": request["custom_id"],
            "result_type": "succeeded",
            "stop_reason": None,
            "text": None,
            "error_type": None,
        }
        try:
            response = client.post(
                f"{self._base_url}/chat/completions",
                json=to_openai_payload(request["params"], self._model),
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            record["stop_reason"] = (
                "max_tokens" if choice.get("finish_reason") == "length" else "end_turn"
            )
            record["text"] = choice["message"]["content"]
        except Exception as exc:  # any transport/server failure is retryable
            logger.warning("Local VLM call failed for %s: %s", request["custom_id"], exc)
            record["result_type"] = "errored"
            record["error_type"] = "api_error"
        return record

    def submit(self, requests: list[dict[str, Any]]) -> str:
        import uuid
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import httpx

        batch_id = f"local-{uuid.uuid4().hex[:12]}"
        records: list[dict[str, Any]] = []
        with (
            httpx.Client(timeout=180) as client,
            ThreadPoolExecutor(self._concurrency) as pool,
        ):
            futures = [pool.submit(self._call, client, request) for request in requests]
            for i, future in enumerate(as_completed(futures), 1):
                records.append(future.result())
                if i % 100 == 0:
                    logger.info("Local labeling: %d/%d frames", i, len(requests))
        self._results_dir.mkdir(parents=True, exist_ok=True)
        with (self._results_dir / f"{batch_id}.jsonl").open("w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        return batch_id

    def status(self, batch_id: str) -> tuple[str, dict[str, int]]:
        path = self._results_dir / f"{batch_id}.jsonl"
        counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        if path.is_file():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                key = record["result_type"] if record["result_type"] in counts else "errored"
                counts[key] += 1
        return "ended", counts

    def results(self, batch_id: str) -> Iterator[dict[str, Any]]:
        path = self._results_dir / f"{batch_id}.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                yield json.loads(line)


def build_request(
    row: dict[str, Any],
    dataroot: Path,
    model: str,
    schema: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    """One Batch API request for one frame; image block before the text block."""
    image_b64 = base64.standard_b64encode((dataroot / row["filename"]).read_bytes()).decode()
    return {
        "custom_id": row["sample_data_token"],
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": "Label this image according to the schema."},
                    ],
                }
            ],
        },
    }


def chunk_requests(
    requests: list[dict[str, Any]], chunk_size: int, max_bytes: int
) -> list[list[dict[str, Any]]]:
    """Split into batches bounded by count AND accumulated encoded size (256MB API cap)."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for request in requests:
        size = len(json.dumps(request))
        if current and (len(current) >= chunk_size or current_bytes + size > max_bytes):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def estimate_cost(n_frames: int, model: str) -> float:
    """Rough batch-discounted USD cost for labeling n_frames with model."""
    price_in, price_out = _PRICES[model]
    per_frame = (_EST_INPUT_TOKENS * price_in + _EST_OUTPUT_TOKENS * price_out) / 1_000_000
    return n_frames * per_frame * 0.5


# --- on-disk state -----------------------------------------------------------------


def _state_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    state_dir = Path(cfg.get("state", {}).get("dir", "data/autolabel"))
    return {
        "dir": state_dir,
        "sample": state_dir / "sample.parquet",
        "batches": state_dir / "batches.json",
        "results": state_dir / "results",
        "labels": state_dir / "labels.parquet",
    }


def _load_batches(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data: list[dict[str, Any]] = json.loads(path.read_text())
    return data


def _save_batches(path: Path, batches: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(batches, indent=2))
    tmp.replace(path)


def _result_statuses(
    results_dir: Path, batches: list[dict[str, Any]]
) -> dict[tuple[str, str], str]:
    """(model, token) -> parse status derived from collected raw results."""
    model_by_batch = {b["batch_id"]: b["model"] for b in batches}
    statuses: dict[tuple[str, str], str] = {}
    if not results_dir.is_dir():
        return statuses
    for path in results_dir.glob("*.jsonl"):
        model = model_by_batch.get(path.stem, "?")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            statuses[(model, record["custom_id"])] = _parse_status(record)[0]
    return statuses


def _parse_status(record: dict[str, Any]) -> tuple[str, SceneLabel | None]:
    """Classify one normalized result record; return (parse_status, label-or-None)."""
    if record["result_type"] == "errored":
        terminal = record.get("error_type") == "invalid_request"
        return ("errored_bad_request" if terminal else "errored_server"), None
    if record["result_type"] in ("expired", "canceled"):
        return record["result_type"], None
    if record.get("stop_reason") == "refusal":
        return "refusal", None
    if record.get("stop_reason") == "max_tokens":
        return "truncated", None
    text = record.get("text")
    if not text:
        return "invalid_json", None
    try:
        return "ok", SceneLabel.model_validate_json(text)
    except ValidationError:
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return "invalid_json", None
        return "schema_invalid", None


# --- pipeline stages ----------------------------------------------------------------


def _pending_tokens(
    tokens: list[str],
    model: str,
    statuses: dict[tuple[str, str], str],
    batches: list[dict[str, Any]],
) -> list[str]:
    """Tokens still needing a request: not terminal, not in a non-ended batch."""
    inflight = {
        token
        for b in batches
        if b["model"] == model and b.get("status") != "ended"
        for token in b["custom_ids"]
    }
    return [
        token
        for token in tokens
        if statuses.get((model, token)) not in TERMINAL_STATUSES and token not in inflight
    ]


def _resolve_provider(cfg: dict[str, Any], provider: str | None) -> str:
    resolved = provider or cfg.get("provider", "anthropic")
    if resolved not in ("anthropic", "local"):
        raise ValueError(f"Unknown provider {resolved!r} (expected 'anthropic' or 'local')")
    return str(resolved)


def _make_transport(cfg: dict[str, Any], provider: str, paths: dict[str, Path]) -> BatchTransport:
    if provider == "local":
        local_cfg = cfg.get("local", {})
        return LocalVLMTransport(
            local_cfg.get("base_url", "http://localhost:8399/v1"),
            local_cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
            paths["results"],
            concurrency=int(local_cfg.get("concurrency", 8)),
        )
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not configured. Add it to .env on this machine "
            "(see .env.example) and re-run — or use the free local provider "
            "(--provider local with a running vLLM server)."
        )
    return AnthropicBatchTransport(settings.anthropic_api_key)


def run_submit(
    config_path: Path,
    *,
    yes: bool = False,
    retry_missing: bool = False,
    dry_run: bool = False,
    provider: str | None = None,
    transport: BatchTransport | None = None,
) -> dict[str, Any]:
    """Build and submit labeling batches (or with dry_run, just size and price them)."""
    settings = get_settings()
    cfg = load_yaml(config_path)
    paths = _state_paths(cfg)
    resolved_provider = _resolve_provider(cfg, provider)
    if not paths["sample"].is_file():
        raise FileNotFoundError(f"No sample at {paths['sample']} — run `autolabel sample` first.")
    sample = pd.read_parquet(paths["sample"])
    batch_cfg = cfg.get("batch", {})
    batches = _load_batches(paths["batches"])

    if resolved_provider == "local":
        # The self-hosted model labels the full sample under its own model id.
        plan: dict[str, list[str]] = {
            cfg.get("local", {}).get("model", "local-vlm"): list(sample["sample_data_token"])
        }
    else:
        plan = {
            cfg["models"]["primary"]: list(sample["sample_data_token"]),
            cfg["models"]["comparison"]: list(
                sample[sample["in_opus_subset"]]["sample_data_token"]
            ),
        }
    if retry_missing:
        statuses = _result_statuses(paths["results"], batches)
        plan = {m: _pending_tokens(t, m, statuses, batches) for m, t in plan.items()}
    plan = {model: tokens for model, tokens in plan.items() if tokens}

    total_cost = (
        0.0
        if resolved_provider == "local"
        else sum(estimate_cost(len(tokens), model) for model, tokens in plan.items())
    )
    for model, tokens in plan.items():
        cost = 0.0 if resolved_provider == "local" else estimate_cost(len(tokens), model)
        logger.info("%s: %d frames, est. $%.2f", model, len(tokens), cost)
    logger.info("Estimated total (batch-discounted): $%.2f", total_cost)
    if not plan:
        logger.info("Nothing to submit.")
        return {"submitted": 0, "estimated_cost": 0.0}

    rows = sample.set_index("sample_data_token", drop=False)
    schema = structured_output_schema()
    dataroot = Path(settings.nuscenes_dataroot)
    max_tokens = int(batch_cfg.get("max_tokens", 800))

    if dry_run:
        # Build a handful of real requests to report request sizing without reading 5K images.
        probe = next(iter(plan.values()))[:5]
        sizes = [
            len(json.dumps(build_request(rows.loc[t].to_dict(), dataroot, "x", schema, max_tokens)))
            for t in probe
        ]
        logger.info(
            "Dry run: %d frames across %d model(s); ~%.0f KB/request; no batches submitted.",
            sum(len(t) for t in plan.values()),
            len(plan),
            (sum(sizes) / len(sizes)) / 1024 if sizes else 0,
        )
        return {"submitted": 0, "estimated_cost": total_cost, "dry_run": True}

    if resolved_provider != "local" and not yes:
        raise SystemExit("Refusing to spend money without --yes (see the cost estimate above).")
    if transport is None:
        transport = _make_transport(cfg, resolved_provider, paths)

    submitted = 0
    for model, tokens in plan.items():
        requests = [
            build_request(rows.loc[token].to_dict(), dataroot, model, schema, max_tokens)
            for token in tokens
        ]
        for chunk in chunk_requests(
            requests,
            int(batch_cfg.get("chunk_size", 500)),
            int(batch_cfg.get("max_batch_bytes", 190_000_000)),
        ):
            batch_id = transport.submit(chunk)
            batches.append(
                {
                    "batch_id": batch_id,
                    "model": model,
                    "n_requests": len(chunk),
                    "custom_ids": [req["custom_id"] for req in chunk],
                    # Local batches execute synchronously inside submit().
                    "status": "ended" if resolved_provider == "local" else "in_progress",
                }
            )
            _save_batches(paths["batches"], batches)
            submitted += len(chunk)
            logger.info("Submitted batch %s (%s, %d requests)", batch_id, model, len(chunk))
    return {"submitted": submitted, "estimated_cost": total_cost}


def run_status(
    config_path: Path,
    provider: str | None = None,
    transport: BatchTransport | None = None,
) -> list[dict[str, Any]]:
    """Refresh and report the processing status of all submitted batches."""
    cfg = load_yaml(config_path)
    paths = _state_paths(cfg)
    resolved_provider = _resolve_provider(cfg, provider)
    batches = _load_batches(paths["batches"])
    for batch in batches:
        if batch.get("status") != "ended":
            if transport is None:  # created lazily: local runs may have nothing to poll
                transport = _make_transport(cfg, resolved_provider, paths)
            batch["status"], batch["counts"] = transport.status(batch["batch_id"])
    _save_batches(paths["batches"], batches)
    for batch in batches:
        logger.info(
            "%s %-18s %-9s %s",
            batch["batch_id"],
            batch["model"],
            batch["status"],
            batch.get("counts", {}),
        )
    return batches


def run_collect(
    config_path: Path,
    provider: str | None = None,
    transport: BatchTransport | None = None,
) -> pd.DataFrame:
    """Download ended batches' results and (re)build the validated labels table."""
    cfg = load_yaml(config_path)
    paths = _state_paths(cfg)
    resolved_provider = _resolve_provider(cfg, provider)
    batches = _load_batches(paths["batches"])

    paths["results"].mkdir(parents=True, exist_ok=True)
    for batch in batches:
        out = paths["results"] / f"{batch['batch_id']}.jsonl"
        if batch.get("status") == "ended" and not out.is_file():
            if transport is None:  # local results were persisted at submit time
                transport = _make_transport(cfg, resolved_provider, paths)
            with out.open("w") as fh:
                for record in transport.results(batch["batch_id"]):
                    fh.write(json.dumps(record) + "\n")
            logger.info("Downloaded results for %s", batch["batch_id"])

    model_by_batch = {b["batch_id"]: b["model"] for b in batches}
    rows = []
    for path in sorted(paths["results"].glob("*.jsonl")):
        model = model_by_batch.get(path.stem, "?")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            status, label = _parse_status(record)
            row: dict[str, Any] = {
                "sample_data_token": record["custom_id"],
                "model": model,
                "parse_status": status,
                "time_of_day": None,
                "weather": None,
                "hazards": None,
                "notable_conditions": None,
                "label_confidence": None,
                **{field: None for field in COUNT_FIELDS},
            }
            if label is not None:
                row.update(
                    time_of_day=label.time_of_day,
                    weather=label.weather,
                    hazards=json.dumps(label.hazards),
                    notable_conditions=json.dumps(label.notable_conditions),
                    label_confidence=label.label_confidence,
                    **{field: getattr(label.object_counts, field) for field in COUNT_FIELDS},
                )
            rows.append(row)

    labels = pd.DataFrame(rows)
    if not labels.empty:
        labels = labels.drop_duplicates(["sample_data_token", "model"], keep="last")
        labels.to_parquet(paths["labels"], index=False)
        by_status = labels.groupby(["model", "parse_status"]).size()
        logger.info("Labels written to %s:\n%s", paths["labels"], by_status.to_string())
    else:
        logger.info("No results downloaded yet — nothing to collect.")
    return labels
