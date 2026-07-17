from typing import Any, Iterator
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import os
import time

# print(f"package:{__package__}, name: {__name__}, file: {__file__}", flush=True)

from ..config import config
from .image_vector import ImageVector
from .map_vector import MapVector
from .number_vector import IntVector, FloatVector
from .embedding_vector import EmbeddingVector
from .position_vector import PositionVector
from .keypoint_vector import KeypointVector
from .person_map_vector import PersonMapVector

from ..paths import split_dir, scores_file
from ..io import load_single_jsonl, write_single_jsonl
from ..loaders.maps_loader import maps_list
from ...external_modules.comparison.algorithm.trueskill_rating import (
    public_score_from_rating,
    Rating,
)
from ...external_modules.database_structure.comparisons_table import get_all_comparisons
from ...external_modules.database_structure.images_table import get_image
from ...shared.logger import get_logger, ModuleLogger

logger: ModuleLogger = get_logger(__name__)

cache_split_data: dict[str, list[dict[str, Any]]] = {}


class VectorList:
    _IMAGE = "image"
    _INT = "int"
    _FLOAT = "float"
    _MAP = "map"
    _EMBEDDING = "embedding"
    _POSITION = "position"
    _KEYPOINT = "keypoint"
    _PERSON_MAP = "person_map"

    def __init__(
        self,
        raw_data: list[tuple[str, dict[str, Any], str, str]],
        # index_list: list[str],
        # vectors_list: list[list[float]],
        # scores_list: list[int],
        # text_list: list[dict[str, Any]],
        # add_new: bool,
        # merge_lists: bool = False,
        read_only: bool,
        # process_images: bool = True,
    ) -> None:

        # self.index_list = index_list
        # self.vectors_list = vectors_list
        # self.scores_list = scores_list
        # self.text_list = text_list
        self.image_paths: dict[str, str] = {}
        self.entries: dict[str, Any] = {}
        self.unique_ids: list[str] = []
        self.scores: dict[str, float] = {}
        self.vector_config = config["vector"]["vectors"]
        self.sorted_vectors: dict[str, Any] = {}
        # self.merge_lists = merge_lists
        self.read_only = read_only
        # self.process_images = process_images
        self.configure_sorted_vectors()
        # self.add_new_to_map = add_new
        self.add_new_to_map = not self.read_only

        # if self.merge_lists:
        #     self.split_vectors()

        if not self.read_only:
            self.load_split_files()
            self.load_split_scores()

        duplicated: list[str] = []
        for data in raw_data:
            image_path, entry, _timestamp, file_id = data

            if (
                file_id in self.unique_ids
                or file_id in self.entries
                or file_id in self.scores
                or file_id in self.image_paths
            ):
                duplicated.append(file_id)
                # logger.debug(f"Duplicate file_id found in entries: {file_id}")
            else:
                self.unique_ids.append(file_id)
            self.entries[file_id] = entry
            if not self.read_only:

                mu = float(entry["rating_mu"])
                sigma = float(entry["rating_sigma"])
                score: float = public_score_from_rating(Rating(mu=mu, sigma=sigma))
                self.scores[file_id] = score
                self.image_paths[file_id] = image_path
        if len(duplicated) > 0:
            logger.debug(
                f"Found {len(duplicated)} duplicated file_ids in raw_data entries. Sample duplicates: {duplicated[:5]}"
            )
        self.final_vector: list[list[float]] = []
        self.final_text_data: list[dict[str, Any]] = []
        self.final_comparison_data: list[dict[str, Any]] = []

    def configure_sorted_vectors(self) -> None:
        for current_type in self.vector_config:
            v_type = current_type["type"]
            name = current_type["name"]

            if v_type == self._MAP:
                vec = MapVector(name)
            elif v_type == self._INT:
                vec = IntVector(name, current_type["max_normalization"])
            elif v_type == self._FLOAT:
                vec = FloatVector(name, current_type["max_normalization"])
            elif v_type == self._EMBEDDING:
                slot_size = current_type["slot_size"]
                vec = EmbeddingVector(name, slot_size=slot_size)
            elif v_type == self._IMAGE:
                model_key = current_type["model_key"]
                slot_size = current_type["slot_size"]
                vec = ImageVector(name, model_key=model_key, slot_size=slot_size)
            elif v_type == self._POSITION:
                vec = PositionVector(name)
            elif v_type == self._KEYPOINT:
                vec = KeypointVector(name)
            elif v_type == self._PERSON_MAP:
                vec = PersonMapVector(name)
            else:
                raise ValueError(f"Unknown vector type: {v_type}")

            self.sorted_vectors[name] = {
                "vector": vec,
                **current_type,
            }

    def _exclude_present_entry(self, current_vector: Any) -> dict[str, Any]:

        new_entries: dict[str, Any] = {}
        current_list = set(current_vector.vector_list.keys())
        for file_id, entry in list(self.entries.items()):
            if file_id in current_list:
                continue
            new_entries[file_id] = entry

        return new_entries

    def _exclude_present_image_path(
        self, current_vector: ImageVector
    ) -> dict[str, Any]:

        new_paths: dict[str, str] = {}
        current_list = set(current_vector.path_list.keys())
        for file_id, entry in self.image_paths.items():
            if file_id in current_list:
                continue
            new_paths[file_id] = entry

        return new_paths

    def create_vectors(self) -> None:
        # split by data type
        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            alias = c["alias"]
            # print(f"Vector config for {v}: {c}")
            if c["type"] == self._MAP:
                map_vector: MapVector = c["vector"]
                # Category registration into the map is done during the text
                # (metadata) stage, not here; create_vectors only maps values.
                new_entries = self._exclude_present_entry(map_vector)
                map_vector.parse_value_list(new_entries, self.add_new_to_map, alias)
                map_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = map_vector
            elif c["type"] == self._INT:
                int_vector: IntVector = c["vector"]
                new_entries = self._exclude_present_entry(int_vector)
                int_vector.parse_value_list(new_entries, alias)
                int_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = int_vector
            elif c["type"] == self._FLOAT:
                float_vector: FloatVector = c["vector"]
                new_entries = self._exclude_present_entry(float_vector)
                float_vector.parse_value_list(new_entries, alias)
                float_vector.create_vector_list()
                if len(float_vector.vector_list) != len(float_vector.value_list):
                    raise ValueError(
                        f"vector length ({len(float_vector.vector_list)}) mismatch with value length ({len(float_vector.value_list)})"
                    )
                self.sorted_vectors[v]["vector"] = float_vector
            elif c["type"] == self._EMBEDDING:
                embedding_vector: EmbeddingVector = c["vector"]
                new_entries = self._exclude_present_entry(embedding_vector)
                embedding_vector.parse_value_list(new_entries, alias)
                embedding_vector.create_vector_list(batch_size=256)
                embedding_vector.create_text_list(batch_size=256)

                self.sorted_vectors[v]["vector"] = embedding_vector
            elif c["type"] == self._IMAGE:  # and self.process_images:
                image_vector: ImageVector = c["vector"]
                new_image_paths: dict[str, str] = self._exclude_present_image_path(
                    image_vector
                )
                result = (-1, -1)
                while isinstance(result, tuple):
                    # print(f"processing images with size: {result}...")
                    result = image_vector.create_vector_list_from_paths(
                        new_image_paths,
                        0.85,
                        rebuild_width=result[0],
                        rebuild_height=result[1],
                    )
                self.sorted_vectors[v]["vector"] = image_vector
            elif c["type"] == self._POSITION:
                position_vector: PositionVector = c["vector"]
                new_entries = self._exclude_present_entry(position_vector)
                position_vector.parse_value_list(new_entries, self.add_new_to_map, alias)
                position_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = position_vector
            elif c["type"] == self._KEYPOINT:
                keypoint_vector: KeypointVector = c["vector"]
                new_entries = self._exclude_present_entry(keypoint_vector)
                keypoint_vector.parse_value_list(new_entries, self.add_new_to_map, alias)
                keypoint_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = keypoint_vector
            elif c["type"] == self._PERSON_MAP:
                person_map_vector: PersonMapVector = c["vector"]
                new_entries = self._exclude_present_entry(person_map_vector)
                person_map_vector.parse_value_list(
                    new_entries, self.add_new_to_map, alias
                )
                person_map_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = person_map_vector

    def validate_and_convert(
        self, data: list[list[float]], name: str, target_size: int
    ) -> npt.NDArray[np.float32]:
        arr = np.array(data, dtype=np.float32)
        if arr.shape[1] != target_size:
            raise ValueError(
                f"Error in '{name}': Row length {arr.shape[1]} "
                f"does not match configured slot_size {target_size}"
            )
        return arr

    def filter_missing_vectors(self) -> None:
        logger.debug("filtering missing scores...")
        valid_ids: list[str] = self.unique_ids
        scores_list: set[str] = set(self.scores.keys())
        error_ids: dict[str, list[str]] = {}
        error_ids["scores"] = []
        for id in valid_ids:
            if id not in scores_list:
                error_ids["scores"].append(id)
            valid_ids = [id for id in valid_ids if id in scores_list]
        logger.info(
            f"valid vectors with present scores:{len(valid_ids)}, error vectors: {len(error_ids)}"
        )

        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            current_vector = c["vector"]
            vector_ids = set(current_vector.vector_list.keys())
            # logger.debug(
            #     f" for vector {c["name"]} initial valid_ids count: {len(valid_ids)}"
            # )
            error_ids[c["name"]] = []
            errors: list[str] = []
            for id in valid_ids:
                if id not in vector_ids:
                    errors.append(id)
                    # logger.debug(
                    #     f"ID '{id[:10]}...' missing from vector '{c['name']}', removing"
                    # )
            if len(errors) > 0:
                error_ids[c["name"]] = errors

            valid_ids = [id for id in valid_ids if id in vector_ids]
            # logger.debug(
            #     f"After filtering with vector '{c['name']}': valid_ids count: {len(valid_ids)}, error_ids count: {len(error_ids)}"
            # )

        logger.info(f"valid vectors:{len(valid_ids)}")
        if len(error_ids.items()) > 0:
            logger.debug(
                f"error vectors:{[(id,len(errors)) for id,errors in error_ids.items()]}"
            )
        self.unique_ids = valid_ids

    def join_vectors(self) -> list[list[float]]:
        clean_arrays: list[npt.NDArray[np.float32]] = []
        with tqdm(
            total=len(self.sorted_vectors),
            desc="joining vectors",
            unit="vectors",
            position=1,
        ) as pbar:
            for v in self.sorted_vectors:
                c = self.sorted_vectors[v]
                current_vector = c["vector"]
                valid_vectors: list[list[float]] = []
                for id in self.unique_ids:
                    valid_vectors.append(current_vector.vector_list[id])

                if len(valid_vectors) != len(self.unique_ids):
                    raise ValueError(
                        f"After validation, vector '{c['name']}' has {len(valid_vectors)} valid entries but expected {len(self.unique_ids)}. This should not happen."
                    )

                converted_vector = self.validate_and_convert(
                    (valid_vectors), c["name"], c["slot_size"]
                )
                clean_arrays.append(converted_vector)
                # logging.debug(
                #     f"Joined vector '{c['name']}' with shape {converted_vector.shape}, (original: {len(current_vector.vector_list)} entries)"
                # )
                pbar.update(1)

        # print(f"clean_arrays lengths: {[len(arr) for arr in clean_arrays]}")

        logger.info("assembling vectors...")
        self.final_vector = np.column_stack(clean_arrays).tolist()
        return self.final_vector

    def convert_text_list(
        self,
        clean_arrays: dict[str, Any],
        current_list: dict[str, str],
        name: str,
    ) -> dict[str, Any]:

        for id, value in current_list.items():
            if id not in clean_arrays:
                clean_arrays[id] = {}
            clean_arrays[id][name] = value
        return clean_arrays

    def join_text_data(self) -> list[dict[str, Any]]:
        if self.final_text_data:
            return self.final_text_data

        initial_arrays: dict[str, Any] = {}
        with tqdm(
            total=len(self.sorted_vectors),
            desc="joining text data",
            unit=" texts",
            position=1,
        ) as pbar:
            for v in self.sorted_vectors:
                c = self.sorted_vectors[v]
                current_vector = c["vector"]
                valid_texts: dict[str, str] = {}
                if c["type"] in [
                    self._MAP,
                    self._INT,
                    self._FLOAT,
                    self._POSITION,
                    self._KEYPOINT,
                    self._PERSON_MAP,
                ]:
                    current_list: dict[str, str] = current_vector.value_list
                elif c["type"] == self._EMBEDDING:
                    current_list: dict[str, str] = current_vector.text_list
                elif c["type"] == self._IMAGE:
                    continue
                else:
                    raise ValueError(
                        f"Unsupported column type for text data: {c['type']}"
                    )

                for id in self.unique_ids:
                    valid_texts[id] = current_list[id]

                initial_arrays = self.convert_text_list(
                    initial_arrays, valid_texts, c["name"]
                )
                pbar.update(1)

        clean_arrays: list[dict[str, Any]] = [
            {key: value} for key, value in initial_arrays.items()
        ]
        self.final_text_data = clean_arrays
        return self.final_text_data

    def join_comparison_data(self) -> list[dict[str, Any]]:
        if self.final_comparison_data:
            return self.final_comparison_data

        index_list = getattr(self, "index_list", self.unique_ids)
        index_lookup = {fid: idx for idx, fid in enumerate(index_list)}
        comparison_rows: list[dict[str, Any]] = []

        for row in get_all_comparisons():
            filename_a = row.get("filename_a")
            filename_b = row.get("filename_b")
            winner = row.get("winner")
            if filename_a is None or filename_b is None or winner is None:
                logger.warning(
                    f"Skipping comparison row {row.get('id')}: missing required "
                    f"filename_a/filename_b/winner (got a={filename_a!r}, "
                    f"b={filename_b!r}, winner={winner!r})"
                )
                continue

            filename_a = str(filename_a)
            filename_b = str(filename_b)
            winner = str(winner)

            if (
                filename_a not in index_lookup
                or filename_b not in index_lookup
                or winner not in index_lookup
            ):
                continue

            if winner == filename_a:
                loser = filename_b
                winner_side = "a"
            elif winner == filename_b:
                loser = filename_a
                winner_side = "b"
            else:
                continue

            score_a = self.scores.get(filename_a)
            score_b = self.scores.get(filename_b)
            score_diff = (
                abs(float(score_a) - float(score_b))
                if score_a is not None and score_b is not None
                else None
            )
            weight_value = row.get("weight", 1.0)
            weight = 1.0 if weight_value is None else float(weight_value)

            comparison_rows.append(
                {
                    "comparison_id": int(row.get("id", 0) or 0),
                    "filename_a": filename_a,
                    "filename_b": filename_b,
                    "winner": winner,
                    "loser": loser,
                    "winner_side": winner_side,
                    "index_a": index_lookup[filename_a],
                    "index_b": index_lookup[filename_b],
                    "winner_index": index_lookup[winner],
                    "loser_index": index_lookup[loser],
                    "score_a": float(score_a) if score_a is not None else None,
                    "score_b": float(score_b) if score_b is not None else None,
                    "score_diff": score_diff,
                    "weight": weight,
                    "transitive_depth": int(row.get("transitive_depth", 0) or 0),
                    "timestamp": str(row.get("timestamp", "")),
                }
            )

        self.final_comparison_data = comparison_rows
        self.comparisons_list = self.final_comparison_data
        return self.final_comparison_data

    def update_lists(self) -> None:
        logger.info("updating vector lists...")
        self.vectors_list = self.final_vector
        self.text_list = self.final_text_data
        self.index_list: list[str] = self.unique_ids
        self.scores_list: list[float] = [self.scores[fid] for fid in self.index_list]
        self.comparisons_list = self.final_comparison_data

    def load_split_files(self) -> None:
        """Load split files back into each vector's ``.vector_list`` /
        ``.value_list`` / ``.text_list`` — the reverse of
        :meth:`export_split_files`.

        Returns the ordered list of unique IDs found in the splits.
        """
        _start = time.perf_counter()

        global cache_split_data
        invalid_entries: dict[str, list[Any]] = {}
        with tqdm(
            total=len(self.sorted_vectors),
            desc="loading split files",
            unit="vectors",
            position=1,
        ) as pbar:
            for v in self.sorted_vectors:
                c = self.sorted_vectors[v]
                name = c["name"]
                v_type = c["type"]
                current_vector = c["vector"]
                # logger.debug(f"vector: {v}", _start)

                split_path = os.path.join(split_dir, v_type, f"{name}.jsonl")
                if not os.path.exists(split_path):
                    logger.warning(f"Split file not found: {split_path}")
                    continue

                raw_vals: dict[str, Any] = {}
                vec_vals: dict[str, list[float]] = {}
                invalid: list[dict[str, Any]] = []
                if name in cache_split_data:
                    reader: Iterator[dict[str, Any]] = iter(cache_split_data[name])
                else:
                    reader: Iterator[dict[str, Any]] = load_single_jsonl(split_path)

                for obj in reader:
                    if obj["raw"] is not None and len(list(obj["vector"])) > 0:
                        self.unique_ids.append(obj["id"])
                        raw_vals[obj["id"]] = obj["raw"]
                        vec_vals[obj["id"]] = obj["vector"]
                        # Re-register categories from the stored raw value so the
                        # map vocabulary stays complete even when split files
                        # already covered these entries (the text stage may not
                        # have registered them on a prior run).
                        if v_type in (self._MAP, self._PERSON_MAP):
                            maps_list.register_value(name, obj["raw"])
                    else:
                        invalid.append(obj)

                if invalid:
                    invalid_entries[name] = invalid

                current_vector.vector_list = vec_vals

                if v_type in [
                    self._MAP,
                    self._INT,
                    self._FLOAT,
                    self._POSITION,
                    self._KEYPOINT,
                    self._PERSON_MAP,
                ]:
                    current_vector.value_list = raw_vals
                elif v_type == self._EMBEDDING:
                    current_vector.text_list = raw_vals
                elif v_type == self._IMAGE:
                    current_vector.path_list = raw_vals
                pbar.update(1)
            # return unique_ids

        # replace adding only uniques each step for just transform to set and then list again, once
        logger.debug(f"before unique:{len(self.unique_ids)}")
        self.unique_ids = list(set(self.unique_ids))
        logger.debug(f"after unique:{len(self.unique_ids)}")

        if len(invalid_entries.items()) > 0:

            logger.debug(
                f"invalid ids: {[(name,len(value)) for name,value in invalid_entries.items()]}"
            )
            example = list(invalid_entries.items())[0]
            logger.debug(
                f"example clip skip: {example}, conditions: {"raw ok" if example[1][0]["raw"] else "raw missing"} , {"vector ok" if example[1][0]["vector"] else "vector missing"}"
            )

    def load_split_scores(self):
        scores_list = list(load_single_jsonl(scores_file))
        if len(scores_list) == len(self.unique_ids):
            self.scores = {
                fid: score for fid, score in zip(self.unique_ids, scores_list)
            }
            return

        logger.warning(
            f"Scores list length {len(scores_list)} does not match unique IDs length {len(self.unique_ids)}. Attempting to rebuild scores mapping from DB..."
        )

        for ids in self.unique_ids:
            if not ids in self.scores:
                row = get_image(ids)
                if row is not None:
                    mu = float(row["rating_mu"])
                    sigma = float(row["rating_sigma"])
                    score = public_score_from_rating(Rating(mu=mu, sigma=sigma))
                    self.scores[ids] = score
                else:
                    logger.warning(f"No DB record for ID: {ids}")

    def export_split_files(self) -> None:
        global cache_split_data
        logger.info("Exporting split data files...")
        with tqdm(
            total=len(self.sorted_vectors),
            desc="exporting splits",
            unit="vectors",
            position=1,
        ) as pbar:
            for v in self.sorted_vectors:
                c = self.sorted_vectors[v]
                name = c["name"]
                v_type = c["type"]
                current_vector = c["vector"]
                raw_values: dict[str, Any] = {}

                if v_type in [
                    self._MAP,
                    self._INT,
                    self._FLOAT,
                    self._POSITION,
                    self._KEYPOINT,
                    self._PERSON_MAP,
                ]:
                    raw_values = current_vector.value_list
                elif v_type == self._EMBEDDING:
                    raw_values = current_vector.text_list
                elif v_type == self._IMAGE:
                    raw_values = {id: id for id in current_vector.vector_list.keys()}
                else:
                    raise ValueError(f"Unknown vector type: {v_type}")

                vector_values_len = len(current_vector.vector_list.values())
                if vector_values_len != len(raw_values):
                    raise ValueError(
                        f"Length mismatch in vector '{name}' of type '{v_type}'. "
                        f"raw values: {len(raw_values)}, vector values: {vector_values_len}"
                    )

                out_dir = os.path.join(split_dir, v_type)
                os.makedirs(out_dir, exist_ok=True)

                out_file = os.path.join(out_dir, f"{name}.jsonl")

                split_data: list[dict[str, Any]] = []

                # with jsonlines.open(out_file, mode="w") as writer:
                for uid in current_vector.vector_list.keys():
                    raw_val = raw_values[uid]
                    vec_val = current_vector.vector_list[uid]
                    split_data.append({"id": uid, "raw": raw_val, "vector": vec_val})

                cache_split_data[name] = split_data
                write_single_jsonl(out_file, split_data, mode="w")
                pbar.update(1)
