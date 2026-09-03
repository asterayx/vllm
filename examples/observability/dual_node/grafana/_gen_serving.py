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
    w: int = 6,
    h: int = 5,
    decimals: int | None = 1,
    color: str | None = None,
    steps: list | None = None,
) -> dict:
    defaults: dict = {
        "thresholds": {
            "mode": "absolute",
            "steps": steps
            or [
                {"color": color or "green", "value": None},
            ],
        },
        "unit": unit,
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    if color:
        defaults["color"] = {"mode": "fixed", "fixedColor": color}
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
            "textMode": "value",
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
    overrides: list | None = None,
    axis_label: str | None = None,
    legend_calcs: list | None = None,
) -> dict:
    custom: dict = {
        "drawStyle": "line",
        "fillOpacity": 10,
        "lineWidth": 2,
        "showPoints": "never",
        "spanNulls": True,
    }
    if axis_label:
        custom["axisLabel"] = axis_label
        custom["axisPlacement"] = "left"
    return {
        "datasource": DS,
        "description": desc,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": custom,
                "unit": unit,
            },
            "overrides": overrides or [],
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": pid,
        "options": {
            "legend": {
                "calcs": legend_calcs or ["mean", "max"],
                "displayMode": "table",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi"},
        },
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def right_axis(name: str, label: str) -> dict:
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [
            {"id": "custom.axisPlacement", "value": "right"},
            {"id": "custom.axisLabel", "value": label},
        ],
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
PREFILL_TPS = rate_c("vllm:prompt_tokens_total")
DECODE_TPS = rate_c("vllm:generation_tokens_total")

panels = [
    stat(
        1,
        "Decode throughput",
        "rate(generation_tokens) — tokens emitted per second.",
        0,
        0,
        DECODE_TPS,
        "decode",
        "ops",
        color="green",
    ),
    stat(
        2,
        "Prefill throughput",
        "rate(prompt_tokens) — prompt tokens consumed per second.",
        6,
        0,
        PREFILL_TPS,
        "prefill",
        "ops",
        color="blue",
    ),
    stat(
        3,
        "Requests running",
        "Requests in RUNNING.",
        12,
        0,
        gauge("vllm:num_requests_running"),
        "running",
        "short",
        decimals=0,
        color="purple",
    ),
    stat(
        4,
        "Requests waiting",
        "Requests in WAITING.",
        18,
        0,
        gauge("vllm:num_requests_waiting"),
        "waiting",
        "short",
        decimals=0,
        color="green",
        steps=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 4},
            {"color": "red", "value": 16},
        ],
    ),
    timeseries(
        5,
        "Throughput (tokens/sec)",
        "Prefill (left axis) vs decode (right axis). "
        "Prefill bursts are much larger than decode tok/s.",
        0,
        5,
        12,
        9,
        [
            target(DECODE_TPS, "decode", "A"),
            target(PREFILL_TPS, "prefill", "B"),
        ],
        "ops",
        axis_label="prefill tok/s",
        overrides=[right_axis("decode", "decode tok/s")],
    ),
    timeseries(
        6,
        "End-to-end request latency (wall time)",
        "Finished-request e2e latency.",
        12,
        5,
        12,
        9,
        [
            target(hq("vllm:e2e_request_latency_seconds", 0.5), "p50", "A"),
            target(hq("vllm:e2e_request_latency_seconds", 0.95), "p95", "B"),
            target(hq("vllm:e2e_request_latency_seconds", 0.99), "p99", "C"),
        ],
        "s",
    ),
    timeseries(
        7,
        "Time to first token (TTFT)",
        "Prefill-dominated time to first token.",
        0,
        14,
        12,
        8,
        [
            target(hq("vllm:time_to_first_token_seconds", 0.5), "p50", "A"),
            target(hq("vllm:time_to_first_token_seconds", 0.95), "p95", "B"),
            target(hq("vllm:time_to_first_token_seconds", 0.99), "p99", "C"),
        ],
        "s",
    ),
    timeseries(
        8,
        "Inter-token latency (ITL)",
        "Decode inter-token latency.",
        12,
        14,
        12,
        8,
        [
            target(hq("vllm:inter_token_latency_seconds", 0.5), "p50", "A"),
            target(hq("vllm:inter_token_latency_seconds", 0.95), "p95", "B"),
        ],
        "s",
    ),
    timeseries(
        9,
        "Request phase wall time (avg)",
        "Average queue / prefill / decode / e2e per finished request.",
        0,
        22,
        12,
        8,
        [
            target(hist_avg("vllm:request_queue_time_seconds"), "queue", "A"),
            target(hist_avg("vllm:request_prefill_time_seconds"), "prefill", "B"),
            target(hist_avg("vllm:request_decode_time_seconds"), "decode", "C"),
            target(hist_avg("vllm:e2e_request_latency_seconds"), "e2e", "D"),
        ],
        "s",
    ),
    timeseries(
        10,
        "DFlash2 speculative-decode acceptance",
        "accept rate = accepted/draft tokens (left). "
        "mean length = 1 + accepted/drafts (right, includes bonus token).",
        12,
        22,
        12,
        8,
        [
            target(ACCEPT, "accept rate", "A"),
            target(MEAN_LEN, "mean accept length", "B"),
        ],
        "short",
        overrides=[
            {
                "matcher": {"id": "byName", "options": "accept rate"},
                "properties": [
                    {"id": "unit", "value": "percentunit"},
                    {"id": "custom.axisPlacement", "value": "left"},
                    {"id": "custom.axisLabel", "value": "accept rate"},
                ],
            },
            right_axis("mean accept length", "mean length"),
        ],
    ),
    row(100, "KV / tokens / draft positions", 30),
    timeseries(
        11,
        "KV cache + queue",
        "KV block usage (left) and running/waiting counts (right).",
        0,
        31,
        12,
        8,
        [
            target(KV_PCT, "KV usage", "A"),
            target(gauge("vllm:num_requests_running"), "running", "B"),
            target(gauge("vllm:num_requests_waiting"), "waiting", "C"),
        ],
        "short",
        overrides=[
            {
                "matcher": {"id": "byName", "options": "KV usage"},
                "properties": [
                    {"id": "unit", "value": "percentunit"},
                    {"id": "custom.axisPlacement", "value": "left"},
                    {"id": "custom.axisLabel", "value": "KV"},
                ],
            },
            {
                "matcher": {"id": "byRegexp", "options": "running|waiting"},
                "properties": [
                    {"id": "custom.axisPlacement", "value": "right"},
                    {"id": "custom.axisLabel", "value": "requests"},
                ],
            },
        ],
    ),
    timeseries(
        12,
        "Prefix cache",
        "Hit rate (left) and query/hit rates (right).",
        12,
        31,
        12,
        8,
        [
            target(PREFIX_HIT, "hit rate", "A"),
            target(rate_c("vllm:prefix_cache_queries_total"), "queries/s", "B"),
            target(rate_c("vllm:prefix_cache_hits_total"), "hits/s", "C"),
        ],
        "short",
        overrides=[
            {
                "matcher": {"id": "byName", "options": "hit rate"},
                "properties": [
                    {"id": "unit", "value": "percentunit"},
                    {"id": "custom.axisPlacement", "value": "left"},
                    {"id": "custom.axisLabel", "value": "hit rate"},
                ],
            },
            {
                "matcher": {"id": "byRegexp", "options": "queries/s|hits/s"},
                "properties": [
                    {"id": "custom.axisPlacement", "value": "right"},
                    {"id": "custom.axisLabel", "value": "ops/s"},
                ],
            },
        ],
    ),
    timeseries(
        13,
        "Request token sizes",
        "Prompt and generation length per finished request.",
        0,
        39,
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
    timeseries(
        14,
        "Draft / accepted tokens",
        "Spec-decode draft and accept counters as rates.",
        12,
        39,
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
        15,
        "Acceptance by draft position",
        "accepted_tokens_per_pos / drafts. Position 0 is the first draft token.",
        0,
        47,
        24,
        8,
        [
            target(
                (
                    "sum by (position) "
                    f"(rate(vllm:spec_decode_num_accepted_tokens_per_pos_total{{{M}}}"
                    "[$__rate_interval])) / on() group_left() "
                    f"clamp_min({rate_c('vllm:spec_decode_num_drafts_total')}, 1e-9)"
                ),
                "pos {{position}}",
                "A",
            ),
        ],
        "percentunit",
    ),
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
        "Spark serving view: decode/prefill throughput (dual axis), "
        "TTFT/ITL/e2e, DFlash2 acceptance, KV, and token sizes. "
        "Model defaults to All."
    ),
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,
    "links": [],
    "panels": panels,
    "preload": False,
    "refresh": "5s",
    "schemaVersion": 40,
    "tags": ["vllm", "serving", "dspark", "dflash2"],
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
    "title": "Spark vLLM — Serving Performance",
    "uid": "vllm-serving-dspark",
    "version": 2,
}

out = Path(__file__).with_name("serving.json")
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
