from typing import Dict, List
from time import time

cache_list: Dict[str, bool] = {}
last_time_served = -1.0


def clear_cache():
    global cache_list
    cache_list = {}
    # print("Cache cleared.")


def in_cache(image_path: str) -> bool:
    global cache_list
    return image_path in cache_list


def add_to_cache(image_path: str):
    global cache_list
    cache_list[image_path] = True
    # print(f"Added to cache: {image_path}")


def disable_from_cache(image_path: str):
    global cache_list
    if image_path in cache_list:
        cache_list[image_path] = False


def remove_from_cache(image_path: str):
    global cache_list
    if image_path in cache_list:
        del cache_list[image_path]
        # print(f"Removed from cache: {image_path}")


def fast_serve():
    global last_time_served
    return last_time_served > (time() - 10)


def get_cache(fast: bool = False) -> List[str]:
    global cache_list, last_time_served
    if not fast:
        last_time_served = time()
    return [img for img, status in cache_list.items() if status]


def total_cached_items() -> int:
    global cache_list
    return len(cache_list)
