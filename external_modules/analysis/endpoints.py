"""Analysis API - endpoints for statistics, parameter analysis, and reporting."""

from __future__ import annotations

import json as python_json
import logging
import os
import sys
import time as time_module
from pathlib import Path
from typing import Any

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from flask import Blueprint, jsonify, request

from external_modules.database_structure.comparisons_table import (
    get_all_comparisons,
    get_total_comparisons,
)
from external_modules.database_structure.images_table import get_all_images
from external_modules.analysis.helpers import distribute
from shared.graph import crystal_graph
from shared.tasks import (
    start_task,
    get_task_status,
    set_task_output,
    cancel_task as _cancel_task,
)


import time

analysis_bp = Blueprint("analysis_v2", __name__, url_prefix="/api/v2/analysis")
logger = logging.getLogger(__name__)


@analysis_bp.route("/stats", methods=["GET"])
def get_stats():
    _start = time.perf_counter()
    _start = time.perf_counter()
    all_images = get_all_images()
    total = len(all_images)
    if total == 0:
        result = jsonify(
            {
                "total_images": 0,
                "total_comparisons": 0,
                "mu_buckets": {},
                "sigma_buckets": {},
                "score_buckets": {},
                "comp_buckets": {},
                "chain_buckets": {},
                "avg_sigma": 0,
                "top_images": [],
                "bottom_images": [],
                "least_compared": [],
                "graph_stats": {},
            }
        )
        return result

    mus = [float(img.get("rating_mu", 25.0)) for img in all_images]
    sigmas = [float(img.get("rating_sigma", 25.0 / 3.0)) for img in all_images]
    scores = [float(img.get("score", 0.5)) for img in all_images]
    comps = [int(img.get("comparison_count", 0)) for img in all_images]

    chain_lengths: list[int] = []
    for img in all_images:
        try:
            cl = crystal_graph.get_node_chain_length(img["filename"])
            chain_lengths.append(cl if cl is not None else 0)
        except Exception:
            chain_lengths.append(0)

    graph_stats = crystal_graph.get_graph_stats()

    sorted_by_score = sorted(all_images, key=lambda x: float(x.get("score", 0.5)))
    sorted_by_comp = sorted(all_images, key=lambda x: int(x.get("comparison_count", 0)))

    result = jsonify(
        {
            "total_images": total,
            "total_comparisons": get_total_comparisons(),
            "total_chains": graph_stats.get("total_chains", 0),
            "mu_buckets": distribute(
                mus,
                [
                    ("under_5", 5),
                    ("5_to_10", 10),
                    ("10_to_15", 15),
                    ("15_to_20", 20),
                    ("20_to_25", 25),
                    ("25_to_30", 30),
                    ("30_to_35", 35),
                    ("35_to_40", 40),
                    ("40_to_45", 45),
                    ("over_45", 99999),
                ],
            ),
            "sigma_buckets": distribute(
                sigmas,
                [
                    ("under_1", 1),
                    ("1_to_2", 2),
                    ("2_to_3", 3),
                    ("3_to_4", 4),
                    ("4_to_5", 5),
                    ("5_to_6", 6),
                    ("6_to_8", 8),
                    ("8_to_10", 10),
                    ("10_to_15", 15),
                    ("over_15", 99999),
                ],
            ),
            "score_buckets": distribute(
                scores,
                [
                    ("under_0.1", 0.1),
                    ("0.1_to_0.2", 0.2),
                    ("0.2_to_0.3", 0.3),
                    ("0.3_to_0.4", 0.4),
                    ("0.4_to_0.5", 0.5),
                    ("0.5_to_0.6", 0.6),
                    ("0.6_to_0.7", 0.7),
                    ("0.7_to_0.8", 0.8),
                    ("0.8_to_0.9", 0.9),
                    ("over_0.9", 99999),
                ],
            ),
            "comp_buckets": distribute(
                [float(c) for c in comps],
                [
                    ("zero", 1),
                    ("1", 2),
                    ("2", 3),
                    ("3_to_4", 5),
                    ("5_to_7", 8),
                    ("8_to_10", 11),
                    ("11_to_15", 16),
                    ("16_to_20", 21),
                    ("21_to_30", 31),
                    ("over_30", 99999),
                ],
            ),
            "chain_buckets": distribute(
                [float(c) for c in chain_lengths],
                [
                    ("zero", 1),
                    ("1", 2),
                    ("2", 3),
                    ("3_to_4", 5),
                    ("5_to_7", 8),
                    ("8_to_10", 11),
                    ("11_to_15", 16),
                    ("16_to_20", 21),
                    ("21_to_30", 31),
                    ("over_30", 99999),
                ],
            ),
            "avg_sigma": round(sum(sigmas) / total, 2) if total else 0,
            "top_images": [
                {
                    "filename": img["filename"],
                    "score": round(float(img.get("score", 0.5)), 4),
                }
                for img in reversed(sorted_by_score[-20:])
            ],
            "bottom_images": [
                {
                    "filename": img["filename"],
                    "score": round(float(img.get("score", 0.5)), 4),
                }
                for img in sorted_by_score[:20]
            ],
            "least_compared": [
                {
                    "filename": img["filename"],
                    "comparison_count": int(img.get("comparison_count", 0)),
                }
                for img in sorted_by_comp[:50]
            ],
            "graph_stats": graph_stats,
        }
    )

    return result


@analysis_bp.route("/analyze-parameters", methods=["POST"])
def analyze_parameters():
    _start = time.perf_counter()
    _start = time.perf_counter()

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        from shared.io import load_single_jsonl
        from shared.paths import vectors_file, text_data_file, scores_file

        vectors_raw = load_single_jsonl(vectors_file)
        scores_raw = load_single_jsonl(scores_file)
        text_data = load_single_jsonl(text_data_file)

        vector_dicts = []
        for i, v in enumerate(vectors_raw):
            entry: dict[str, Any] = {}
            if isinstance(v, dict):
                entry.update(v)
            score = scores_raw[i] if i < len(scores_raw) else 0.5
            if isinstance(score, dict):
                entry["score"] = score.get("score", 0.5)
            else:
                entry["score"] = float(score)
            vector_dicts.append(entry)

        from shared.training.parameter_analysis import ParameterAnalyzer
        from shared.paths import vectors_file

        report_dir = os.path.dirname(vectors_file)
        analyzer = ParameterAnalyzer(vector_dicts, text_data, output_dir=report_dir)
        analyzer.analyze_all()

        report_files = []
        for fname in [
            "term_correlations.json",
            "sampler_stats.json",
            "scheduler_stats.json",
        ]:
            fpath = os.path.join(report_dir, fname)
            if os.path.isfile(fpath):
                report_files.append(fpath)

        set_task_output(
            tid,
            {
                "status": "done",
                "result": {
                    "type": "parameter_analysis",
                    "report_files": report_files,
                    "categories_analyzed": len(report_files),
                },
            },
        )

    _, body = start_task(_run, task_prefix="params", args=())

    return jsonify(body)


@analysis_bp.route("/analyze-matrix", methods=["POST"])
def analyze_matrix():
    _start = time.perf_counter()
    _start = time.perf_counter()

    def _run(tid):
        _start = time.perf_counter()
        _start = time.perf_counter()
        from shared.training.matrix_analysis import MatrixAnalyzer
        from shared.io import load_single_jsonl
        from shared.paths import vectors_file, scores_file, text_data_file

        scores_raw = load_single_jsonl(scores_file)
        scores = [
            float(s["score"]) if isinstance(s, dict) else float(s) for s in scores_raw
        ]
        text_data = load_single_jsonl(text_data_file)

        analyzer = MatrixAnalyzer(scores, text_data)
        analyzer.build_matrix()
        analyzer.calculate_statistics()
        matrix_path = os.path.join(
            os.path.dirname(vectors_file), "matrix_analysis.json"
        )
        analyzer.export_to_json(matrix_path)
        summary = analyzer.get_matrix_summary()
        set_task_output(
            tid, {"status": "done", "result": {"summary": summary, "path": matrix_path}}
        )

    _, body = start_task(_run, task_prefix="matrix", args=())

    return jsonify(body)


@analysis_bp.route("/report-file", methods=["GET"])
def get_report_file():
    _start = time.perf_counter()
    _start = time.perf_counter()
    path = request.args.get("path", "")
    max_items = request.args.get("max", 200, type=int)
    max_items = max(10, min(max_items, 5000))
    if not path:
        result = jsonify({"error": "Missing path parameter"}), 400

        result = result

        return result
    if not os.path.isabs(path):
        result = jsonify({"error": "Path must be absolute"}), 400

        result = result

        return result
    if not os.path.isfile(path):
        result = jsonify({"error": "File not found"}), 404

        result = result

        return result
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".json",):
        result = jsonify({"error": "Only .json files are supported"}), 400

        result = result

        return result
    try:
        filename = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "{":
                items = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(python_json.loads(line))
                        except python_json.JSONDecodeError:
                            pass
                    if len(items) >= max_items:
                        break
                total_count = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
                result = jsonify(
                    {
                        "status": "ok",
                        "filename": filename,
                        "format": "jsonl",
                        "content": items,
                        "total_lines": total_count,
                        "showing": min(total_count, max_items),
                    }
                )
                return result
            else:
                f.seek(0)
                content = python_json.load(f)
                result = jsonify(
                    {
                        "status": "ok",
                        "filename": filename,
                        "format": "json",
                        "content": content,
                    }
                )

                result = result

                return result
    except Exception as e:
        result = jsonify({"error": str(e)}), 500

        result = result

        return result


@analysis_bp.route("/task/<task_id>", methods=["GET"])
def get_task(task_id: str):
    since = request.args.get("since", 0, type=int)
    info = get_task_status(task_id, since=since)
    if info is None:
        result = jsonify({"error": "Task not found"}), 404

        return result
    result = jsonify(info)

    return result


@analysis_bp.route("/task/<task_id>/cancel", methods=["POST"])
def cancel_task(task_id: str):
    ok = _cancel_task(task_id)
    if not ok:
        result = jsonify({"error": "Task not found or already finished"}), 404

        return result
    result = jsonify({"status": "cancelled"})

    return result


def register_analysis_routes(app) -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    app.register_blueprint(analysis_bp)
