#!/usr/bin/env python3
"""Generate grafana/serving.json. Run from this directory."""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
M = 'model_name=~"$model"'


def target(expr: str, legend: str, ref: str = "A") -> dict:
    return {
        "datasource": DS,
        "editorMode": "code",
        "expr": expr,
        "legendFormat": legend,
        "refId": ref,
    }


def hq(metric: str, q: float) -> str:
    return (
        f"histogram_quantile({q}, "
        f"sum by (le) (rate({metric}_bucket{{{M}}}[$__rate_interval])))"
    )


def hist_avg(metric: str) -> str:
    return (
        f"sum(rate({metric}_sum{{{M}}}[$__rate_interval])) / "
        f"clamp_min(sum(rate({metric}_count{{{M}}}[$__rate_interval])), 1e-9)"
    )


def rate_c(metric: str) -> str:
    return f"sum(rate({metric}{{{M}}}[$__rate_interval]))"


def gauge(metric: str) -> str:
    return f"sum({metric}{{{M}}})"


def stat(
    pid: int,
    title: str,
    desc: str,
    x: int,
    y: int,
    expr: str,
    legend: str,
    unit: str,
    w: int = 4,
    h: int = 4,
    decimals: int | None = None,
    min_v: float | None = None,
    max_v: float | None = None,
    steps: list | None = None,
) -> dict:
    defaults: dict = {
        "thresholds": {
            "mode": "absolute",
            "steps": steps
            or [
                {"color": "green", "value": None},
            ],
        },
        "unit": unit,
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    if min_v is not None:
        defaults["min"] = min_v
    if max_v is not None:
        defaults["max"] = max_v
    return {
        "datasource": DS,
        "description": desc,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "colorMode": "value",
            "graphMode": "area",
            "reduceOptions": {"calcs": ["lastNotNull"]},
        },
        "targets": [target(expr, legend)],
        "title": title,
        "type": "stat",
    }


def timeseries(
    pid: int,
    title: str,
    desc: str,
    x: int,
    y: int,
    w: int,
    h: int,
    targets: list,
    unit: str,
) -> dict:
    return {
        "datasource": DS,
        "description": desc,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "drawStyle": "line",
                    "fillOpacity": 12,
                    "lineWidth": 1,
                    "showPoints": "never",
                    "spanNulls": True,
                },
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi"},
        },
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def row(pid: int, title: str, y: int) -> dict:
    return {
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "id": pid,
        "panels": [],
        "title": title,
        "type": "row",
    }


ACCEPT = (
    f"{rate_c('vllm:spec_decode_num_accepted_tokens_total')} / "
    f"clamp_min({rate_c('vllm:spec_decode_num_draft_tokens_total')}, 1e-9)"
)
MEAN_LEN = (
    f"1 + {rate_c('vllm:spec_decode_num_accepted_tokens_total')} / "
    f"clamp_min({rate_c('vllm:spec_decode_num_drafts_total')}, 1e-9)"
)
PREFIX_HIT = (
    f"{rate_c('vllm:prefix_cache_hits_total')} / "
    f"clamp_min({rate_c('vllm:prefix_cache_queries_total')}, 1e-9)"
)
KV_PCT = f"avg(vllm:kv_cache_usage_perc{{{M}}})"
DECODE_TPS = (
    "1 / clamp_min("
    + hq("vllm:inter_token_latency_seconds", 0.5)
    + ", 1e-9)"
)

pct_steps = [
    {"color": "green", "value": None},
    {"color": "orange", "value": 0.7},
    {"color": "red", "value": 0.9},
]
accept_steps = [
    {"color": "red", "value": None},
    {"color": "orange", "value": 0.4},
    {"color": "green", "value": 0.7},
]

panels = [
    row(100, "Overview", 0),
    stat(
        1,
        "Running",
        "Requests in RUNNING.",
        0,
        1,
        gauge("vllm:num_requests_running"),
        "running",
        "short",
        decimals=0,
    ),
    stat(
        2,
        "Waiting",
        "Requests in WAITING.",
        4,
        1,
        gauge("vllm:num_requests_waiting"),
        "waiting",
        "short",
        decimals=0,
        steps=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 4},
            {"color": "red", "value": 16},
        ],
    ),
    stat(
        3,
        "KV cache",
        "Fraction of used KV blocks (0–1).",
        8,
        1,
        KV_PCT,
        "kv",
        "percentunit",
        decimals=2,
        min_v=0,
        max_v=1,
        steps=pct_steps,
    ),
    stat(
        4,
        "Prompt tok/s",
        "Prefill throughput: rate of prompt tokens.",
        12,
        1,
        rate_c("vllm:prompt_tokens_total"),
        "prompt",
        "ops",
        decimals=0,
    ),
    stat(
        5,
        "Gen tok/s",
        "Decode throughput: rate of generated tokens.",
        16,
        1,
        rate_c("vllm:generation_tokens_total"),
        "gen",
        "ops",
        decimals=0,
    ),
    stat(
        6,
        "DSpark accept",
        "accepted_tokens / draft_tokens (PromQL from spec_decode/metrics.py).",
        20,
        1,
        ACCEPT,
        "accept",
        "percentunit",
        decimals=2,
        min_v=0,
        max_v=1,
        steps=accept_steps,
    ),
    stat(
        7,
        "TTFT p50",
        "Time to first token (prefill-dominated).",
        0,
        5,
        hq("vllm:time_to_first_token_seconds", 0.5),
        "ttft p50",
        "s",
        decimals=3,
    ),
    stat(
        8,
        "ITL p50",
        "Inter-token latency (decode).",
        4,
        5,
        hq("vllm:inter_token_latency_seconds", 0.5),
        "itl p50",
        "s",
        decimals=4,
    ),
    stat(
        9,
        "Decode tok/s p50",
        "1 / ITL p50 — per-request decode speed.",
        8,
        5,
        DECODE_TPS,
        "decode tps",
        "ops",
        decimals=1,
    ),
    stat(
        10,
        "Prefill p50",
        "Request prefill time p50.",
        12,
        5,
        hq("vllm:request_prefill_time_seconds", 0.5),
        "prefill p50",
        "s",
        decimals=3,
    ),
    stat(
        11,
        "Decode p50",
        "Request decode time p50.",
        16,
        5,
        hq("vllm:request_decode_time_seconds", 0.5),
        "decode p50",
        "s",
        decimals=3,
    ),
    stat(
        12,
        "Accept length",
        "1 + accepted_tokens / drafts (includes bonus token).",
        20,
        5,
        MEAN_LEN,
        "mean len",
        "short",
        decimals=2,
        steps=[
            {"color": "red", "value": None},
            {"color": "orange", "value": 1.5},
            {"color": "green", "value": 2.5},
        ],
    ),
    row(101, "Prefill / decode", 9),
    timeseries(
        13,
        "TTFT (prefill)",
        "Time to first token. p50 / p99 / average.",
        0,
        10,
        12,
        8,
        [
            target(hq("vllm:time_to_first_token_seconds", 0.5), "p50", "A"),
            target(hq("vllm:time_to_first_token_seconds", 0.99), "p99", "B"),
            target(hist_avg("vllm:time_to_first_token_seconds"), "avg", "C"),
        ],
        "s",
    ),
    timeseries(
        14,
        "ITL / TPOT (decode)",
        "Inter-token latency and time-per-output-token.",
        12,
        10,
        12,
        8,
        [
            target(hq("vllm:inter_token_latency_seconds", 0.5), "ITL p50", "A"),
            target(hq("vllm:inter_token_latency_seconds", 0.99), "ITL p99", "B"),
            target(
                hq("vllm:request_time_per_output_token_seconds", 0.5),
                "TPOT p50",
                "C",
            ),
            target(
                hq("vllm:request_time_per_output_token_seconds", 0.99),
                "TPOT p99",
                "D",
            ),
        ],
        "s",
    ),
    timeseries(
        15,
        "Request prefill vs decode time",
        "Per-request prefill and decode duration.",
        0,
        18,
        12,
        8,
        [
            target(hq("vllm:request_prefill_time_seconds", 0.5), "prefill p50", "A"),
            target(hq("vllm:request_prefill_time_seconds", 0.99), "prefill p99", "B"),
            target(hq("vllm:request_decode_time_seconds", 0.5), "decode p50", "C"),
            target(hq("vllm:request_decode_time_seconds", 0.99), "decode p99", "D"),
        ],
        "s",
    ),
    timeseries(
        16,
        "E2E + queue",
        "End-to-end request latency and queue wait.",
        12,
        18,
        12,
        8,
        [
            target(hq("vllm:e2e_request_latency_seconds", 0.5), "e2e p50", "A"),
            target(hq("vllm:e2e_request_latency_seconds", 0.99), "e2e p99", "B"),
            target(hq("vllm:request_queue_time_seconds", 0.5), "queue p50", "C"),
            target(hq("vllm:request_queue_time_seconds", 0.99), "queue p99", "D"),
        ],
        "s",
    ),
    row(102, "Tokens in / out", 26),
    timeseries(
        17,
        "Token throughput",
        "Prompt (input) vs generation (output) tokens per second.",
        0,
        27,
        12,
        8,
        [
            target(rate_c("vllm:prompt_tokens_total"), "prompt tok/s", "A"),
            target(rate_c("vllm:generation_tokens_total"), "gen tok/s", "B"),
        ],
        "ops",
    ),
    timeseries(
        18,
        "Request token sizes",
        "Prompt and generation length per finished request.",
        12,
        27,
        12,
        8,
        [
            target(hq("vllm:request_prompt_tokens", 0.5), "prompt p50", "A"),
            target(hq("vllm:request_prompt_tokens", 0.99), "prompt p99", "B"),
            target(hq("vllm:request_generation_tokens", 0.5), "gen p50", "C"),
            target(hq("vllm:request_generation_tokens", 0.99), "gen p99", "D"),
        ],
        "short",
    ),
    row(103, "KV / cache", 35),
    timeseries(
        19,
        "KV usage + queue",
        "KV cache fraction plus running/waiting request counts.",
        0,
        36,
        12,
        8,
        [
            target(KV_PCT, "KV usage", "A"),
            target(gauge("vllm:num_requests_running"), "running", "B"),
            target(gauge("vllm:num_requests_waiting"), "waiting", "C"),
        ],
        "short",
    ),
    timeseries(
        20,
        "Prefix cache",
        "Prefix-cache hit rate and query/hit rates.",
        12,
        36,
        12,
        8,
        [
            target(PREFIX_HIT, "hit rate", "A"),
            target(rate_c("vllm:prefix_cache_queries_total"), "queries/s", "B"),
            target(rate_c("vllm:prefix_cache_hits_total"), "hits/s", "C"),
        ],
        "short",
    ),
    row(104, "DSpark / spec decode", 44),
    timeseries(
        21,
        "Acceptance rate + mean length",
        "accepted/draft tokens; mean length = 1 + accepted/drafts.",
        0,
        45,
        12,
        8,
        [
            target(ACCEPT, "accept rate", "A"),
            target(MEAN_LEN, "mean accept length", "B"),
        ],
        "short",
    ),
    timeseries(
        22,
        "Draft / accepted tokens",
        "Spec-decode draft and accept counters as rates.",
        12,
        45,
        12,
        8,
        [
            target(rate_c("vllm:spec_decode_num_drafts_total"), "drafts/s", "A"),
            target(
                rate_c("vllm:spec_decode_num_draft_tokens_total"),
                "draft tok/s",
                "B",
            ),
            target(
                rate_c("vllm:spec_decode_num_accepted_tokens_total"),
                "accepted tok/s",
                "C",
            ),
        ],
        "ops",
    ),
    timeseries(
        23,
        "Acceptance by draft position",
        "accepted_tokens_per_pos / drafts. Position 0 is the first draft token.",
        0,
        53,
        24,
        8,
        [
            target(
                (
                    "sum by (position) "
                    f"(rate(vllm:spec_decode_num_accepted_tokens_per_pos_total{{{M}}}"
                    "[$__rate_interval])) / "
                    f"clamp_min({rate_c('vllm:spec_decode_num_drafts_total')}, 1e-9)"
                ),
                "pos {{position}}",
                "A",
            ),
        ],
        "percentunit",
    ),
]

# Override units on mixed KV panel: Grafana can't easily dual-axis in this
# compact schema; keep KV as percentunit via override.
panels[next(i for i, p in enumerate(panels) if p.get("id") == 19)][
    "fieldConfig"
]["overrides"] = [
    {
        "matcher": {"id": "byName", "options": "KV usage"},
        "properties": [{"id": "unit", "value": "percentunit"}],
    }
]
panels[next(i for i, p in enumerate(panels) if p.get("id") == 21)][
    "fieldConfig"
]["overrides"] = [
    {
        "matcher": {"id": "byName", "options": "accept rate"},
        "properties": [{"id": "unit", "value": "percentunit"}],
    }
]
panels[next(i for i, p in enumerate(panels) if p.get("id") == 20)][
    "fieldConfig"
]["overrides"] = [
    {
        "matcher": {"id": "byName", "options": "hit rate"},
        "properties": [{"id": "unit", "value": "percentunit"}],
    }
]

dashboard = {
    "annotations": {
        "list": [
            {
                "builtIn": 1,
                "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                "enable": True,
                "hide": True,
                "iconColor": "rgba(0, 211, 255, 1)",
                "name": "Annotations & Alerts",
                "type": "dashboard",
            }
        ]
    },
    "description": (
        "One-page vLLM serving view: prefill/decode, DSpark acceptance, "
        "KV / prefix cache, and token in/out. Model defaults to All."
    ),
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,
    "links": [],
    "panels": panels,
    "preload": False,
    "refresh": "5s",
    "schemaVersion": 40,
    "tags": ["vllm", "serving", "dspark", "spec-decode"],
    "templating": {
        "list": [
            {
                "current": {"text": "Prometheus", "value": "prometheus"},
                "label": "datasource",
                "name": "DS_PROMETHEUS",
                "query": "prometheus",
                "refresh": 1,
                "type": "datasource",
            },
            {
                "allValue": ".*",
                "current": {"text": "All", "value": "$__all"},
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "definition": (
                    "label_values(vllm:num_requests_running, model_name)"
                ),
                "includeAll": True,
                "label": "model",
                "multi": True,
                "name": "model",
                "query": (
                    "label_values(vllm:num_requests_running, model_name)"
                ),
                "refresh": 2,
                "sort": 1,
                "type": "query",
            },
        ]
    },
    "time": {"from": "now-15m", "to": "now"},
    "timepicker": {},
    "timezone": "browser",
    "title": "vLLM Serving / DSpark / KV",
    "uid": "vllm-serving-dspark",
    "version": 1,
}

out = Path(__file__).with_name("serving.json")
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
