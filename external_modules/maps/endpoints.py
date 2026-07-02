"""Maps API - endpoints for chain visualizations (graph data)."""

from flask import Blueprint, jsonify, request

from ..database_structure.images_table import get_all_images
from ...shared.graph.crystal_graph import crystal_graph
import time

maps_bp = Blueprint("maps_v2", __name__, url_prefix="/api/v2/maps")
from ...shared.logger import get_logger, ModuleLogger
logger: ModuleLogger = get_logger(__name__)


@maps_bp.route("/graph-data", methods=["GET"])
def get_graph_data():
    _start = time.perf_counter()
    _start = time.perf_counter()
    try:
        if crystal_graph.is_cache_stale():
            crystal_graph.rebuild_from_database()

        stats = crystal_graph.get_graph_stats()
        node_to_height: dict[str, int] = {}
        for proxy, _node_list in crystal_graph.get_all_chains():
            for node_proxy in proxy.nodes:
                filename = node_proxy.filename
                if (
                    filename not in node_to_height
                    or proxy.length > node_to_height[filename]
                ):
                    node_to_height[filename] = proxy.length

        img_dict = {img["filename"]: img for img in get_all_images()}
        nodes = []
        for node in crystal_graph.get_all_nodes():
            filename = node.filename
            img_data = img_dict.get(filename)
            comp = node.get_component()
            nodes.append(
                {
                    "id": filename,
                    "score": round(float(img_data["score"]), 4) if img_data else 0.5,
                    "height": node_to_height.get(filename, 0),
                    "component": comp.id if comp else None,
                    "comparison_count": (
                        int(img_data["comparison_count"]) if img_data else 0
                    ),
                    "is_top": node.is_top(),
                    "is_bottom": node.is_bottom(),
                }
            )

        edges = [
            {"source": winner.filename, "target": loser.filename, "weight": 1.0}
            for winner, loser in crystal_graph.get_all_links()
        ]
        all_components = crystal_graph.get_all_components()
        component_members = {
            comp.id: [n.filename for n in comp.nodes] for comp in all_components
        }
        chains = []
        for chain_proxy, _ in crystal_graph.get_all_chains():
            comp = chain_proxy.get_component()
            chains.append(
                {
                    "id": chain_proxy.id,
                    "component": comp.id if comp else None,
                    "nodes": [n.filename for n in chain_proxy.nodes],
                }
            )

        result = jsonify(
            {
                "nodes": nodes,
                "edges": edges,
                "components": component_members,
                "chains": chains,
                "stats": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "total_components": len(all_components),
                    "total_chains": stats.get("total_chains", 0),
                },
            }
        )
        return result
    except Exception as exc:
        logger.error("Error in get_graph_data: %s", exc, exc_info=True)
        result = jsonify({"error": str(exc)}), 500
        return result


def register_maps_routes(app) -> None:
    _start = time.perf_counter()
    _start = time.perf_counter()
    app.register_blueprint(maps_bp)
