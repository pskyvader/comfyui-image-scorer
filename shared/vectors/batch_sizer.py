from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import torch
import logging

from ..loaders.model_loader import model_loader
from ..config import config
from ..io import load_json, atomic_write_json
from ..paths import vectors_size_file
logger = logging.getLogger(__name__)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    batch_size: int
    delta_memory: int
    timestamp: float


@dataclass
class ProfileData:
    model_name: str
    device_name: str
    device_id: str
    total_memory: int
    model_memory_bytes: int
    fixed_overhead: int | None = None
    pixel_cost: float | None = None
    r_squared: float | None = None
    history: dict[str, list[HistoryEntry]] = field(default_factory=dict)


class BatchSizer:
    _SAFETY_FRACTION: float = 0.9

    def __init__(self) -> None:
        _start = time.perf_counter()
        _start = time.perf_counter()
        self._active: ProfileData | None = None
        self._ready: bool = False
        logger.debug("__init__ took %.4fs", time.perf_counter() - _start)

    def _ensure_session_profiled(self) -> None:
        _start = time.perf_counter()
        _start = time.perf_counter()
        if self._ready:
            logger.debug("_ensure_session_profiled took %.4fs", time.perf_counter() - _start)
            logger.debug("_ensure_session_profiled took %.4fs", time.perf_counter() - _start)
            return

        data, err = load_json(vectors_size_file, expect=dict, default={})
        profiles_data = data.get("profiles", []) if isinstance(data, dict) else []

        profiles: list[ProfileData] = []
        for pd in profiles_data:
            hist: dict[str, list[HistoryEntry]] = {}
            for key, entries in pd.get("history", {}).items():
                hist[key] = [HistoryEntry(**e) for e in entries]
            others = {k: v for k, v in pd.items() if k != "history"}
            profiles.append(ProfileData(**others, history=hist))

        vision_config = config["prepare"]["vision_model"]
        model_name: str = vision_config["name"]
        device_id: str = vision_config["device"]
        device_name = torch.cuda.get_device_name(device_id)
        total_mem = int(torch.cuda.get_device_properties(device_id).total_memory)

        for p in profiles:
            if p.model_name == model_name and p.device_name == device_name:
                self._active = p
                break

        if self._active is None:
            self._active = ProfileData(
                model_name=model_name,
                device_name=device_name,
                device_id=device_id,
                total_memory=total_mem,
                model_memory_bytes=0,
            )

        if model_loader.vision_model is not None:
            self._active.model_memory_bytes = int(
                torch.cuda.memory_allocated(self._active.device_id)
            )

        self._ready = True

    @staticmethod
def _resolution_key(int, height: int) -> str:
        a, b = (width, height) if width <= height else (height, width)
        result = f"{a}x{b}"
        logger.debug("_resolution_key took %.4fs", time.perf_counter() - _start)
        return result

def get(int, height: int, rebuild: bool = False) -> int:
        self._ensure_session_profiled()
        profile = self._active
        assert profile is not None

        key = self._resolution_key(width, height)

        if key in profile.history and not rebuild:
            result = max(e.batch_size for e in profile.history[key])
            logger.debug("get took %.4fs", time.perf_counter() - _start)
            return result

        result = self._profile_new_resolution(width, height, rebuild)
        logger.debug("get took %.4fs", time.perf_counter() - _start)
        return result

    def _profile_new_resolution(
        _start = time.perf_counter()
        _start = time.perf_counter()
        self, width: int, height: int, rebuild: bool = False
        logger.debug("_profile_new_resolution took %.4fs", time.perf_counter() - _start)
    ) -> int:
        profile = self._active
        assert profile is not None
        key = self._resolution_key(width, height)

        if key not in profile.history:
            profile.history[key] = []

        torch.cuda.set_per_process_memory_fraction(0.99, 0)

        model, _, _ = model_loader.load_vision_model()
        profile.model_memory_bytes = int(torch.cuda.memory_allocated(profile.device_id))

        device_id = profile.device_id
        available = (
            int(profile.total_memory * self._SAFETY_FRACTION)
            - profile.model_memory_bytes
        )

        if rebuild and profile.history[key]:
            best = max(e.batch_size for e in profile.history[key])

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_id)

            batch_tensor = torch.zeros((best, 3, height, width), device=device_id)
            try:
                with torch.inference_mode():
                    model.eval()
                    model(batch_tensor)
                    torch.cuda.synchronize(device_id)

                peak = int(torch.cuda.max_memory_allocated(device_id))
                delta = peak - profile.model_memory_bytes

                profile.history[key].append(
                    HistoryEntry(
                        batch_size=best,
                        delta_memory=delta,
                        timestamp=time.time(),
                    )
                )

                if peak < int(profile.total_memory * self._SAFETY_FRACTION):
                    self._fit_model()
                    self._save_cache()
                    return best

            except Exception:
                pass
            finally:
                del batch_tensor
                torch.cuda.empty_cache()

            low, high = 1, best - 1
        else:
            high = 200
            if profile.pixel_cost is not None:
                per_image = profile.pixel_cost * width * height * 3
                fixed = profile.fixed_overhead or 0
                if per_image > 0:
                    high = max(1, int((available - fixed) / per_image) * 2)

        low, last_success = 1, 0

        while low <= high:
            mid = (low + high) // 2
            logger.debug(
                f"Profiling batch size {mid} for resolution {width}x{height} (rebuild={rebuild})"
            )
            if mid == last_success:
                break

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_id)

            batch_tensor = torch.zeros((mid, 3, height, width), device=device_id)
            try:
                with torch.inference_mode():
                    model.eval()
                    model(batch_tensor)
                    torch.cuda.synchronize(device_id)

                peak = int(torch.cuda.max_memory_allocated(device_id))
                delta = peak - profile.model_memory_bytes

                profile.history[key].append(
                    HistoryEntry(
                        batch_size=mid,
                        delta_memory=delta,
                        timestamp=time.time(),
                    )
                )

                if peak < int(profile.total_memory * self._SAFETY_FRACTION):
                    last_success = mid
                    low = mid + 1
                else:
                    high = mid - 1
            except Exception:
                high = mid - 1
            finally:
                del batch_tensor
                torch.cuda.empty_cache()

        result = max(last_success, 1)

        old_fixed = profile.fixed_overhead
        old_pixel = profile.pixel_cost
        self._fit_model()
        if (
            old_fixed is not None
            and old_pixel is not None
            and profile.fixed_overhead is not None
            and profile.pixel_cost is not None
        ):
            fixed_shift = abs(profile.fixed_overhead - old_fixed) / max(
                abs(old_fixed), 1
            )
            pixel_shift = abs(profile.pixel_cost - old_pixel) / max(
                abs(old_pixel), 1e-12
            )
            if fixed_shift > 0.1 or pixel_shift > 0.1:
                print(
                    f"Warning: model parameters shifted significantly after rebuild "
                    f"(fixed: {old_fixed} -> {profile.fixed_overhead}, "
                    f"pixel: {old_pixel} -> {profile.pixel_cost})"
                )

        self._save_cache()
        return result

    def _fit_model(self) -> None:
        _start = time.perf_counter()
        _start = time.perf_counter()
        profile = self._active
        assert profile is not None

        X: list[float] = []
        Y: list[float] = []

        for res_key, entries in profile.history.items():
            w_str, h_str = res_key.split("x")
            w, h = int(w_str), int(h_str)
            for e in entries:
                X.append(float(e.batch_size * w * h * 3))
                Y.append(float(e.delta_memory))

        n = len(X)
        if n < 2:
            profile.fixed_overhead = None
            profile.pixel_cost = None
            profile.r_squared = None
            logger.debug("_fit_model took %.4fs", time.perf_counter() - _start)
            logger.debug("_fit_model took %.4fs", time.perf_counter() - _start)
            return

        sum_x = sum(X)
        sum_y = sum(Y)
        sum_xx = sum(x * x for x in X)
        sum_xy = sum(x * y for x, y in zip(X, Y))

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            profile.fixed_overhead = None
            profile.pixel_cost = None
            profile.r_squared = None
            logger.debug("_fit_model took %.4fs", time.perf_counter() - _start)
            logger.debug("_fit_model took %.4fs", time.perf_counter() - _start)
            return

        pixel_cost = (n * sum_xy - sum_x * sum_y) / denom
        fixed_overhead = (sum_y - pixel_cost * sum_x) / n

        mean_y = sum_y / n
        ss_res = sum((y - (fixed_overhead + pixel_cost * x)) ** 2 for x, y in zip(X, Y))
        ss_tot = sum((y - mean_y) ** 2 for y in Y)

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        profile.fixed_overhead = int(round(fixed_overhead))
        profile.pixel_cost = pixel_cost
        profile.r_squared = r_squared

    def _save_cache(self) -> None:
        _start = time.perf_counter()
        _start = time.perf_counter()
        profile = self._active
        if profile is None:
            logger.debug("_save_cache took %.4fs", time.perf_counter() - _start)
            logger.debug("_save_cache took %.4fs", time.perf_counter() - _start)
            return

        data, err = load_json(vectors_size_file, expect=dict, default={})
        if not isinstance(data, dict):
            data = {}
        if "profiles" not in data:
            data["profiles"] = []

        hist_dict: dict[str, list[dict[str, Any]]] = {}
        for res_key, entries in profile.history.items():
            hist_dict[res_key] = [
                {
                    "batch_size": e.batch_size,
                    "delta_memory": e.delta_memory,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ]

        profile_data: dict[str, Any] = {
            "model_name": profile.model_name,
            "device_name": profile.device_name,
            "device_id": profile.device_id,
            "total_memory": profile.total_memory,
            "model_memory_bytes": profile.model_memory_bytes,
            "fixed_overhead": profile.fixed_overhead,
            "pixel_cost": profile.pixel_cost,
            "r_squared": profile.r_squared,
            "history": hist_dict,
        }

        existing = data["profiles"]
        for i, p in enumerate(existing):
            if (
                p.get("model_name") == profile.model_name
                and p.get("device_name") == profile.device_name
            ):
                existing[i] = profile_data
                break
        else:
            existing.append(profile_data)

        atomic_write_json(vectors_size_file, data, indent=4)
