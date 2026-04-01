from typing import Any
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ..config import config
from .image_vector import ImageVector
from .map_vector import MapVector
from .number_vector import IntVector, FloatVector
from .embedding_vector import EmbeddingVector


class VectorList:
    _IMAGE = "image"
    _INT = "int"
    _FLOAT = "float"
    _MAP = "map"
    _EMBEDDING = "embedding"

    def __init__(
        self,
        raw_data: list[tuple[str, dict[str, Any], str, str]],
        index_list: list[str],
        vectors_list: list[list[float]],
        scores_list: list[int],
        text_list: list[dict[str, Any]],
        add_new: bool,
        merge_lists: bool = False,
        read_only: bool = False,
        process_images: bool = True,
    ) -> None:

        self.add_new_to_map = add_new
        self.index_list = index_list
        self.vectors_list = vectors_list
        self.scores_list = scores_list
        self.text_list = text_list
        self.image_paths: list[str] = []
        self.entries: list[Any] = []
        self.unique_ids: list[str] = []
        self.scores: list[int] = []
        self.vector_config = config["vector"]["vectors"]
        self.sorted_vectors: dict[str, Any] = {}
        self.merge_lists = merge_lists
        self.read_only = read_only
        self.process_images = process_images
        self.configure_sorted_vectors()

        if self.merge_lists:
            self.split_vectors()

        for data in raw_data:
            image_path, entry, timestamp, file_id = data
            self.entries.append(entry)
            if not self.read_only:
                unique_id = f"{file_id}#{timestamp}"
                self.unique_ids.append(unique_id)
                current_score = entry["score"]
                score_modifier = entry.get("score_modifier", 0)

                self.scores.append(current_score + (score_modifier * 0.1))
                self.image_paths.append(image_path)

        self.final_vector: list[list[float]] = []
        self.final_text_data: list[dict[str, Any]] = []

    def configure_sorted_vectors(self) -> None:
        image_type: list[dict[str, Any]] = []
        map_type: list[dict[str, Any]] = []
        int_type: list[dict[str, Any]] = []
        float_type: list[dict[str, Any]] = []
        embedding_type: list[dict[str, Any]] = []

        for c in self.vector_config:
            if c["type"] == self._IMAGE:
                image_type.append(c)
            elif c["type"] == self._INT:
                int_type.append(c)
            elif c["type"] == self._FLOAT:
                float_type.append(c)
            elif c["type"] == self._MAP:
                map_type.append(c)
            elif c["type"] == self._EMBEDDING:
                embedding_type.append(c)

        for current_type in map_type:
            self.sorted_vectors[current_type["name"]] = {
                "vector": MapVector(current_type["name"]),
                **current_type,
            }
        for current_type in int_type:
            self.sorted_vectors[current_type["name"]] = {
                "vector": IntVector(
                    current_type["name"], current_type["max_normalization"]
                ),
                **current_type,
            }
        for current_type in float_type:
            self.sorted_vectors[current_type["name"]] = {
                "vector": FloatVector(
                    current_type["name"], current_type["max_normalization"]
                ),
                **current_type,
            }
        for current_type in embedding_type:
            self.sorted_vectors[current_type["name"]] = {
                "vector": EmbeddingVector(current_type["name"]),
                **current_type,
            }
        for current_type in image_type:
            self.sorted_vectors[current_type["name"]] = {
                "vector": ImageVector(current_type["name"]),
                **current_type,
            }

    def create_vectors(self) -> None:
        # split by data type
        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            alias=c.get("alias", None)
            # print(f"Vector config for {v}: {c}")
            if c["type"] == self._MAP:
                map_vector: MapVector = c["vector"]
                map_vector.parse_value_list(self.entries, self.add_new_to_map,alias)
                map_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = map_vector
            elif c["type"] == self._INT:
                int_vector:IntVector = c["vector"]
                int_vector.parse_value_list(self.entries,alias)
                int_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = int_vector
            elif c["type"] == self._FLOAT:
                float_vector:FloatVector = c["vector"]
                float_vector.parse_value_list(self.entries,alias)
                float_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = float_vector
            elif c["type"] == self._EMBEDDING:
                embedding_vector :EmbeddingVector= c["vector"]
                embedding_vector.parse_value_list(self.entries,alias)
                embedding_vector.create_vector_list(batch_size=256)
                embedding_vector.create_text_list(batch_size=256)

                self.sorted_vectors[v]["vector"] = embedding_vector
            elif c["type"] == self._IMAGE and self.process_images:
                image_vector:ImageVector = c["vector"]
                image_vector.path_list = self.image_paths
                result = (-1, -1)
                while isinstance(result, tuple):
                    result = image_vector.create_vector_list_from_paths(
                        rebuild_width=result[0], rebuild_height=result[1]
                    )
                self.sorted_vectors[v]["vector"] = image_vector

    def validate_and_convert(
        self, data: list[list[str]], name: str, target_size: int
    ) -> npt.NDArray[np.float32]:
        try:
            return np.array(data, dtype=np.float32)
        except ValueError:
            arr_safe = np.array(data, dtype=object)
            lengths = np.vectorize(len)(arr_safe)
            bad_indices = np.where(lengths != target_size)[0]
            raise ValueError(
                f"Error in '{name}': Row lengths are inconsistent.\n"
                f"Expected: {target_size}\n"
                f"First 5 Mismatched Indices: {bad_indices[:5]}\n"
                f"Actual lengths at those indices: {lengths[bad_indices[:5]]}"
            )

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
                converted_vector = self.validate_and_convert(
                    current_vector.vector_list, c["name"], c["slot_size"]
                )
                clean_arrays.append(converted_vector)
                pbar.update(1)

        print("assembling vectors...")
        self.final_vector = np.column_stack(clean_arrays).tolist()
        #self._update_lists()
        return self.final_vector

    def convert_text_list(
        self,
        clean_arrays: list[dict[str, Any]],
        current_vector: Any,
        name: str,
        column_type: str,
    ) -> list[dict[str, Any]]:

        if column_type in [self._MAP, self._INT, self._FLOAT]:
            result = current_vector.value_list
        elif column_type == self._EMBEDDING:
            result = current_vector.text_list
        else:
            return clean_arrays
        if len(clean_arrays) == 0:
            while len(clean_arrays) < len(result):
                clean_arrays.append({})
        elif len(result) != len(clean_arrays):
            print(result)
            print(clean_arrays)
            raise ValueError(
                f"the number of elements doesn't match. real: {len(result)}, expected: {len(clean_arrays)}"
            )

        for i in range(len(result)):
            clean_arrays[i][name] = result[i]
        return clean_arrays

    def join_text_data(self) -> list[dict[str, Any]]:
        clean_arrays: list[dict[str, Any]] = []
        with tqdm(
            total=len(self.sorted_vectors),
            desc="joining text data",
            unit=" texts",
            position=1,
        ) as pbar:
            for v in self.sorted_vectors:
                c = self.sorted_vectors[v]
                current_vector = c["vector"]
                clean_arrays = self.convert_text_list(
                    clean_arrays, current_vector, c["name"], c["type"]
                )
                pbar.update(1)

        print("assembling text data...")
        self.final_text_data = clean_arrays
        #self._update_lists()
        return self.final_text_data

    def update_lists(self) -> None:
        print("updating vector lists...")
        if not self.merge_lists:
            self.vectors_list.extend(self.final_vector)
        else:
            self.vectors_list = self.final_vector
        self.text_list.extend(self.final_text_data)
        self.index_list.extend(self.unique_ids)
        self.scores_list.extend(self.scores)

    def fix_row(self, row: list[float], expected_total: int) -> list[float]:
        difference = expected_total - len(row)
        if difference < 0:
            raise ValueError(
                f"Initial row has more values than expected. Expected total: {expected_total}, actual total: {len(row)}"
            )
        vector_length = 0
        last_map_index = 0
        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            previous_length = vector_length
            vector_length += c["slot_size"]
            previous_vector = row[:previous_length]
            subrow = row[previous_length:vector_length]
            remaining_vector = row[vector_length:]
            if c["type"] == self._MAP:
                last_map_index = vector_length
                # map vectors are one hot,
                # find if subrow has more than 1 non zero slot
                # search for index of second non zero slot
                non_zero_indices = [i for i, x in enumerate(subrow) if x != 0]
                if len(non_zero_indices) > 1:
                    second_index = non_zero_indices[1]
                    # split the row at the second index,
                    previous_subrow = subrow
                    remaining_vector = subrow[second_index:] + remaining_vector
                    subrow = subrow[:second_index]
                    slots_to_add = c["slot_size"] - len(subrow)
                    subrow += [0.0] * slots_to_add
                    row = previous_vector + subrow + remaining_vector
                    difference -= slots_to_add
                    if difference == 0:
                        return row
                    if difference < 0:
                        raise ValueError(
                            f"Row part {c["name"]} has more values than expected after fixing map vector.{subrow} ",
                            f"Expected part: {c["slot_size"]}, actual part: {len(subrow)}",
                            f"Expected total: {expected_total}, actual total: {len(row)}",
                            f"second non zero index: {second_index}, non zero indices: {non_zero_indices}",
                            f"previous subrow: {previous_subrow}",
                            f"slots added to subrow: {slots_to_add}, difference after fixing: {difference}",
                        )
        difference = expected_total - len(row)
        if difference == 0:
            return row
        # add remaining difference as zeros to the end of the row, or if there is a map vector, add to the end of the last map vector
        if last_map_index > 0:
            row = row[:last_map_index] + [0.0] * difference + row[last_map_index:]
            return row

        row.extend([0.0] * difference)
        return row

    def split_vectors(self):
        if not self.vectors_list:
            return
        sizes = [c["slot_size"] for c in self.sorted_vectors.values()]
        expected_total = sum(sizes)

        # ---- Fast path ----
        try:
            matrix = np.array(self.vectors_list)
            if matrix.ndim != 2 or matrix.shape[1] != expected_total:
                raise ValueError("Shape mismatch")
        except Exception:
            for i, row in enumerate(self.vectors_list):
                if len(row) != expected_total:
                    self.vectors_list[i] = self.fix_row(row, expected_total)

            matrix = np.array(self.vectors_list)

        # ---- Continue normally ----
        indices = np.cumsum(sizes)[:-1]
        segments = np.split(matrix, indices, axis=1)

        for (v, _), segment in zip(self.sorted_vectors.items(), segments):
            converted_vector = segment.tolist()
            self.sorted_vectors[v]["vector"].vector_list = converted_vector
