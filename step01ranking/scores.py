from typing import Any, Dict, List, Tuple
from flask import jsonify
from pathlib import Path

from shared.io import atomic_write_json
from step01ranking.utils import get_json_path, load_metadata, image_root
from step01ranking.cache import disable


def _write_score(json_path: str, score: Any) -> Tuple[bool, str | None]:
    meta = load_metadata(json_path)#.replace(".json", ""))
    if not meta:
        return False, "Invalid metadata"

    ts = next(iter(meta.keys()))
    meta[ts]["score"] = score

    try:
        atomic_write_json(json_path, meta, indent=4)
    except Exception as e:
        return False, str(e)

    return True, None

def _normalize_items(data: Any) -> Tuple[List[Dict[str, Any]], Tuple[Any, int] | None]:
    if isinstance(data, dict) and "image" in data:
        return [data], None
    if isinstance(data, dict):
        return ([{"image": k, "score": v} for k, v in data.items()], None)
    if isinstance(data, list):
        return (data, None)
    return ([], (jsonify({"error": "Invalid payload, expected list or dict"}), 400))



def submit_scores_handler(data: Any):
    items, err_resp = _normalize_items(data)
    if err_resp:
        return err_resp
    
    # if not isinstance(data, (dict, list)):
    #     error_message={"error": "Invalid payload"}
    #     print("error message",error_message)
    #     return jsonify(error_message), 400
    root=image_root()
    root_path = Path(root)
    
    # items = (
    #     [{"image": k, "score": v} for k, v in data.items()]
    #     if isinstance(data, dict)
    #     else data
    # )

    result = {"ok": [], "errors": []}

    for item in items:
        if not isinstance(item, dict) or "image" not in item or "score" not in item:
            result["errors"].append(item)
            continue

        img = (str(root_path)+"\\"+item["image"]).replace("\\", "/")
        score=item["score"]
        
        json_path: str = get_json_path(img)
        print ("json path",json_path)
        ok, err = _write_score(json_path, score)
        #ok, err = _write_score(get_json_path(img), item["score"])
        
        disable(img)
        if ok:
            result["ok"].append(img)
        else:
            result["errors"].append({"image": img, "error": err})
    #print (result, items)
    return jsonify(result), (200 if not result["errors"] else 207)
