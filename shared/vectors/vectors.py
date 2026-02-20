from typing import List, Dict, Any, Tuple
import numpy as np
from ..config import config
from .image_vector import ImageVector
from .map_vector import MapVector
from .number_vector import IntVector, FloatVector
from .embedding_vector import EmbeddingVector


class VectorList:
    def __init__(
        self,
        raw_data: List[Tuple[str, Dict[str, Any], str, str]],
        index_list: List[str],
        vectors_list: List[List[float]],
        scores_list: List[int],
        add_new: bool,
        merge_lists: bool = False,
    ) -> None:

        self._IMAGE = "image"
        self._INT = "int"
        self._FLOAT = "float"
        self._MAP = "map"
        self._EMBEDDING = "embedding"
        self.add_new = add_new
        self.index_list = index_list
        self.vectors_list = vectors_list
        self.scores_list = scores_list
        self.image_paths: List[str] = []
        self.entries: List[Any] = []
        self.timestamps: List[Any] = []
        self.file_ids: List[str] = []
        self.unique_ids: List[str] = []
        self.scores: List[int] = []
        self.vector_config = config["vector"]["vectors"]
        self.sorted_vectors: Dict[str, Any] = {}
        self.configure_sorted_vectors()
        self.merge_lists = merge_lists
        if self.merge_lists:
            self.split_vectors()

        for data in raw_data:
            image_path, entry, timestamp, file_id = data
            unique_id = f"{file_id}#{timestamp}"
            self.unique_ids.append(unique_id)
            self.scores.append(entry["score"])

            self.image_paths.append(image_path)
            self.entries.append(entry)
            self.timestamps.append(timestamp)
            self.file_ids.append(file_id)

        self.image_vectors: Dict[str, ImageVector] = {}
        self.map_vectors: Dict[str, MapVector] = {}
        self.int_vectors: Dict[str, IntVector] = {}
        self.float_vectors: Dict[str, FloatVector] = {}
        self.embedding_vectors: Dict[str, EmbeddingVector] = {}
        self.final_vector: List[List[float]] = []


    def configure_sorted_vectors(self):
        image_type: List[Dict[str, Any]] = []
        map_type: List[Dict[str, Any]] = []
        int_type: List[Dict[str, Any]] = []
        float_type: List[Dict[str, Any]] = []
        embedding_type: List[Dict[str, Any]] = []

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

    def create_vectors(self):
        # split by data type
        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            #print(f"Vector config for {v}: {c}")
            if c["type"] == self._MAP:
                current_vector = c["vector"]
                current_vector.parse_value_list(self.entries, self.add_new)
                current_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = current_vector
            elif c["type"] == self._INT:
                current_vector = c["vector"]
                current_vector.parse_value_list(self.entries)
                current_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = current_vector
            elif c["type"] == self._FLOAT:
                current_vector = c["vector"]
                current_vector.parse_value_list(self.entries)
                current_vector.create_vector_list()
                self.sorted_vectors[v]["vector"] = current_vector
            elif c["type"] == self._EMBEDDING:
                current_vector = c["vector"]
                current_vector.parse_value_list(self.entries)
                current_vector.create_vector_list(batch_size=256)
                self.sorted_vectors[v]["vector"] = current_vector
            elif c["type"] == self._IMAGE:
                current_vector = c["vector"]
                current_vector.path_list = self.image_paths
                current_vector.create_vector_list_from_paths(max_batch_size=16)
                self.sorted_vectors[v]["vector"] = current_vector

    def validate_and_convert(self, data: List[List[str]], name: str, target_size: int):
        try:
            return np.array(data, dtype=float)
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

    def join_vectors(self):
        clean_arrays: list[np.ndarray] = []
        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            current_vector = c["vector"]
            converted_vector = self.validate_and_convert(
                current_vector.vector_list, c["name"], c["slot_size"]
            )
            clean_arrays.append(converted_vector)

        self.final_vector = np.column_stack(clean_arrays).tolist()
        self._update_lists()
        return self.final_vector

    def _update_lists(self):
        if not self.merge_lists:
            self.vectors_list.extend(self.final_vector)
        else:
            self.vectors_list = self.final_vector
        self.index_list.extend(self.unique_ids)
        self.scores_list.extend(self.scores)


        
    def fix_row(self, row: List[float], expected_total: int) -> List[float]:
        difference = expected_total - len(row)
        if difference<0:
            raise ValueError(
                f"Initial row has more values than expected. Expected total: {expected_total}, actual total: {len(row)}"
            )
        vector_length=0
        last_map_index=0
        for v in self.sorted_vectors:
            c = self.sorted_vectors[v]
            previous_length = vector_length
            vector_length+=c["slot_size"]
            previous_vector=row[:previous_length]
            subrow = row[previous_length:vector_length]
            remaining_vector=row[vector_length:]
            if c["type"] == self._MAP:
                last_map_index=vector_length
                #map vectors are one hot, 
                #find if subrow has more than 1 non zero slot
                #search for index of second non zero slot
                non_zero_indices = [i for i, x in enumerate(subrow) if x != 0]
                if len(non_zero_indices) > 1:
                    second_index = non_zero_indices[1]
                    #split the row at the second index, 
                    previous_subrow = subrow
                    remaining_vector = subrow[second_index:] + remaining_vector
                    subrow=subrow[:second_index]
                    slots_to_add = c["slot_size"] - len(subrow)
                    subrow+=[0.0]*slots_to_add
                    row = previous_vector + subrow + remaining_vector
                    difference-=slots_to_add
                    if difference == 0:
                        return row
                    if difference < 0:
                        raise ValueError(
                            f"Row part {c["name"]} has more values than expected after fixing map vector.{subrow} ",
                            f"Expected part: {c["slot_size"]}, actual part: {len(subrow)}",
                            f"Expected total: {expected_total}, actual total: {len(row)}",
                            f"second non zero index: {second_index}, non zero indices: {non_zero_indices}",
                            f"previous subrow: {previous_subrow}",
                            f"slots added to subrow: {slots_to_add}, difference after fixing: {difference}"

                        )
        difference = expected_total - len(row)
        if difference==0:
            return row
        #add remaining difference as zeros to the end of the row, or if there is a map vector, add to the end of the last map vector
        if last_map_index>0:
            row = row[:last_map_index] + [0.0]*difference + row[last_map_index:]
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

        for (v, config), segment in zip(self.sorted_vectors.items(), segments):
            converted_vector = segment.tolist()
            self.sorted_vectors[v]["vector"].vector_list = converted_vector
