from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import torch

from ..io import atomic_write_json, load_json
from ..logger import get_logger

from ..config import config
from ..loaders.model_loader import model_loader
from ..paths import vectors_size_file

logger = get_logger(__name__)


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

    def __init__(self, model_key: str) -> None:
        self._model_key = model_key
        self._active: ProfileData | None = None
        self._ready: bool = False

    def _ensure_session_profiled(self) -> None:
        _start = time.perf_counter()
        if self._ready:
            return

        data, _ = load_json(vectors_size_file, expect=dict)
        profiles_data = (
            data["profiles"] if isinstance(data, dict) and "profiles" in data else []
        )

        profiles: list[ProfileData] = []
        for profile_data in profiles_data:
            history_data = profile_data["history"] if "history" in profile_data else {}
            history = {
                key: [HistoryEntry(**entry) for entry in entries]
                for key, entries in history_data.items()
            }
            profile_fields = {
                key: value for key, value in profile_data.items() if key != "history"
            }
            profiles.append(ProfileData(**profile_fields, history=history))

        vision_models = config["prepare"]["vision_models"]
        model_cfg = vision_models[self._model_key]
        if not model_cfg:
            model_cfg = next(iter(vision_models.values()), {})
        model_name: str = model_cfg["name"]
        device_id: str = model_cfg["device"]
        device_name = torch.cuda.get_device_name(device_id)
        total_mem = int(torch.cuda.get_device_properties(device_id).total_memory)

        self._active = None
        for profile in profiles:
            if profile.model_name == model_name and profile.device_name == device_name:
                self._active = profile
                break

        if self._active is None:
            self._active = ProfileData(
                model_name=model_name,
                device_name=device_name,
                device_id=device_id,
                total_memory=total_mem,
                model_memory_bytes=0,
            )

        if model_loader.vision_model_cache:
            self._active.model_memory_bytes = int(
                torch.cuda.memory_allocated(self._active.device_id)
            )

        self._ready = True

    @staticmethod
    def _resolution_key(width: int, height: int) -> str:
        _start = time.perf_counter()
        a, b = (width, height) if width <= height else (height, width)
        result = f"{a}x{b}"

        return result

    def get(self, width: int, height: int, rebuild: bool) -> int:
        _start = time.perf_counter()
        self._ensure_session_profiled()
        profile = self._active
        assert profile is not None

        key = self._resolution_key(width, height)
        if key in profile.history and not rebuild:
            result = max(entry.batch_size for entry in profile.history[key])

            return result

        result = self._profile_new_resolution(width, height, rebuild)

        return result

    def _profile_new_resolution(
        self, width: int, height: int, rebuild: bool
    ) -> int:
        _start = time.perf_counter()
        profile = self._active
        assert profile is not None

        key = self._resolution_key(width, height)
        if key not in profile.history:
            profile.history[key] = []

        torch.cuda.set_per_process_memory_fraction(0.99, 0)
        model, _, _, _ = model_loader.load_vision_model(model_key=self._model_key)
        profile.model_memory_bytes = int(torch.cuda.memory_allocated(profile.device_id))

        device_id = profile.device_id
        available = (
            int(profile.total_memory * self._SAFETY_FRACTION)
            - profile.model_memory_bytes
        )

        if rebuild and profile.history[key]:
            best = max(entry.batch_size for entry in profile.history[key])
            result = self._evaluate_candidate(
                model=model,
                profile=profile,
                key=key,
                candidate=best,
                width=width,
                height=height,
                device_id=device_id,
            )
            if result is not None:

                return result
            low, high = 1, best - 1
        else:
            high = 500
            if profile.pixel_cost is not None:
                per_image = profile.pixel_cost * width * height * 3
                fixed = profile.fixed_overhead or 0
                if per_image > 0:
                    high = max(1, int((available - fixed) / per_image) * 2)
            low = 1

        last_success = 0
        while low <= high:
            mid: int = (low + high) // 2
            if mid == last_success:
                break

            result = self._evaluate_candidate(
                model=model,
                profile=profile,
                key=key,
                candidate=mid,
                width=width,
                height=height,
                device_id=device_id,
            )
            if result is None:
                high = mid - 1
            else:
                last_success = result
                low = mid + 1

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
                logger.warning(
                    "model parameters shifted significantly after rebuild "
                    f"(fixed: {old_fixed} -> {profile.fixed_overhead}, "
                    f"pixel: {old_pixel} -> {profile.pixel_cost})"
                )

        self._save_cache()

        return result

    def _evaluate_candidate(
        self,
        *,
        model: Any,
        profile: ProfileData,
        key: str,
        candidate: int,
        width: int,
        height: int,
        device_id: str,
    ) -> int | None:
        result = None
        logger.debug(
            f"Evaluating batch size {candidate} for resolution {width}x{height}"
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_id)
        batch_tensor = torch.zeros((candidate, 3, height, width), device=device_id)
        try:
            with torch.inference_mode():
                model.eval()
                model(batch_tensor)
                torch.cuda.synchronize(device_id)

            peak = int(torch.cuda.max_memory_allocated(device_id))
            delta = peak - profile.model_memory_bytes
            profile.history[key].append(
                HistoryEntry(
                    batch_size=candidate,
                    delta_memory=delta,
                    timestamp=time.time(),
                )
            )
            threshold = int(profile.total_memory * self._SAFETY_FRACTION)
            result = candidate if peak < threshold else None
        except Exception as e:
            logger.warning(
                f"batch size {candidate} for resolution {width}x{height} failed with error: {e}"
            )
        finally:
            del batch_tensor
            torch.cuda.empty_cache()
        return result

    def _fit_model(self) -> None:
        _start = time.perf_counter()
        profile = self._active
        assert profile is not None

        x_values: list[float] = []
        y_values: list[float] = []
        for res_key, entries in profile.history.items():
            width_str, height_str = res_key.split("x")
            width = int(width_str)
            height = int(height_str)
            for entry in entries:
                x_values.append(float(entry.batch_size * width * height * 3))
                y_values.append(float(entry.delta_memory))

        count = len(x_values)
        if count < 2:
            profile.fixed_overhead = None
            profile.pixel_cost = None
            profile.r_squared = None

            return

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xx = sum(value * value for value in x_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        denom = count * sum_xx - sum_x * sum_x
        if denom == 0:
            profile.fixed_overhead = None
            profile.pixel_cost = None
            profile.r_squared = None

            return

        pixel_cost = (count * sum_xy - sum_x * sum_y) / denom
        fixed_overhead = (sum_y - pixel_cost * sum_x) / count
        mean_y = sum_y / count
        ss_res = sum(
            (y - (fixed_overhead + pixel_cost * x)) ** 2
            for x, y in zip(x_values, y_values)
        )
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        profile.fixed_overhead = int(round(fixed_overhead))
        profile.pixel_cost = pixel_cost
        profile.r_squared = r_squared

    def _save_cache(self) -> None:
        _start = time.perf_counter()
        profile = self._active
        if profile is None:

            return

        data, _ = load_json(vectors_size_file, expect=dict)
        if not isinstance(data, dict):
            data = {}
        if "profiles" not in data:
            data["profiles"] = []

        history_payload: dict[str, list[dict[str, Any]]] = {}
        for res_key, entries in profile.history.items():
            history_payload[res_key] = [
                {
                    "batch_size": entry.batch_size,
                    "delta_memory": entry.delta_memory,
                    "timestamp": entry.timestamp,
                }
                for entry in entries
            ]

        profile_payload: dict[str, Any] = {
            "model_name": profile.model_name,
            "device_name": profile.device_name,
            "device_id": profile.device_id,
            "total_memory": profile.total_memory,
            "model_memory_bytes": profile.model_memory_bytes,
            "fixed_overhead": profile.fixed_overhead,
            "pixel_cost": profile.pixel_cost,
            "r_squared": profile.r_squared,
            "history": history_payload,
        }

        existing_profiles = data["profiles"]
        for index, existing in enumerate(existing_profiles):
            if (
                existing["model_name"] == profile.model_name
                and existing["device_name"] == profile.device_name
            ):
                existing_profiles[index] = profile_payload
                break
        else:
            existing_profiles.append(profile_payload)

        atomic_write_json(vectors_size_file, data, indent=4)
