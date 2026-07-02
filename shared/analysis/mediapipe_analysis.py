import logging
import os
from typing import Any

import numpy as np
from numpy import typing as npt
from PIL import Image

from ..paths import mediapipe_models_dir

import mediapipe as mp

_mediapipe_loaded = False
_mp = None
_FaceDetector = None
_FaceDetectorOptions = None
_PoseLandmarker = None
_PoseLandmarkerOptions = None
_HandLandmarker = None
_HandLandmarkerOptions = None
_RunningMode = None
_BaseOptions = None


def _ensure_mediapipe():
    global _mediapipe_loaded, _mp
    global _FaceDetector, _FaceDetectorOptions
    global _PoseLandmarker, _PoseLandmarkerOptions
    global _HandLandmarker, _HandLandmarkerOptions, _RunningMode, _BaseOptions
    if _mediapipe_loaded:
        return
    _mp = mp
    _FaceDetector = mp.tasks.vision.FaceDetector
    _FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    _PoseLandmarker = mp.tasks.vision.PoseLandmarker
    _PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    _HandLandmarker = mp.tasks.vision.HandLandmarker
    _HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    _RunningMode = mp.tasks.vision.RunningMode
    _BaseOptions = mp.tasks.BaseOptions
    _mediapipe_loaded = True


class MediaPipeAnalyzer:
    """
    Detects faces, body pose, and hands using MediaPipe.

    Internal structure of body_pose (132 floats per person):
        Pose landmarks (MediaPipe Pose 33 landmarks, each [x, y, z, visibility]):
        Index mapping:
           0: nose                   1: left_eye_inner
           2: left_eye               3: left_eye_outer
           4: right_eye_inner        5: right_eye
           6: right_eye_outer        7: left_ear
           8: right_ear              9: mouth_left
          10: mouth_right           11: left_shoulder
          12: right_shoulder        13: left_elbow
          14: right_elbow           15: left_wrist
          16: right_wrist           17: left_pinky
          18: right_pinky           19: left_index
          20: right_index           21: left_thumb
          22: right_thumb           23: left_hip
          24: right_hip             25: left_knee
          26: right_knee            27: left_ankle
          28: right_ankle           29: left_heel
          30: right_heel            31: left_foot_index
          32: right_foot_index

    Internal structure of face_bbox (5 floats per face):
       [0]: x_min (relative), [1]: y_min (relative),
       [2]: width (relative), [3]: height (relative),
       [4]: detection confidence

    Internal structure of left_hand / right_hand (63 floats per hand):
       21 hand landmarks (MediaPipe Hands), each [x, y, z].
    """

    def __init__(self) -> None:
        self._face_detector: Any = None
        self._pose_landmarker: Any = None
        self._hand_landmarker: Any = None

    def _image_to_rgb(self, img: Image.Image) -> npt.NDArray[np.uint8]:
        return np.asarray(img.convert("RGB"))

    def _get_face_detector(self) -> Any:
        _ensure_mediapipe()
        if self._face_detector is None:
            model_path = os.path.join(mediapipe_models_dir, "face_detection.tflite")
            options = _FaceDetectorOptions(
                base_options=_BaseOptions(model_asset_path=model_path),
                running_mode=_RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            self._face_detector = _FaceDetector.create_from_options(options)
        return self._face_detector

    def _get_pose_landmarker(self) -> Any:
        _ensure_mediapipe()
        if self._pose_landmarker is None:
            model_path = os.path.join(mediapipe_models_dir, "pose_landmarker.task")
            options = _PoseLandmarkerOptions(
                base_options=_BaseOptions(model_asset_path=model_path),
                running_mode=_RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
            )
            self._pose_landmarker = _PoseLandmarker.create_from_options(options)
        return self._pose_landmarker

    def _get_hand_landmarker(self) -> Any:
        _ensure_mediapipe()
        if self._hand_landmarker is None:
            model_path = os.path.join(mediapipe_models_dir, "hand_landmarker.task")
            options = _HandLandmarkerOptions(
                base_options=_BaseOptions(model_asset_path=model_path),
                running_mode=_RunningMode.IMAGE,
                num_hands=4,
                min_hand_detection_confidence=0.5,
            )
            self._hand_landmarker = _HandLandmarker.create_from_options(options)
        return self._hand_landmarker

    def analyze(self, img: Image.Image) -> dict[str, Any]:
        _ensure_mediapipe()
        rgb = self._image_to_rgb(img)
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)

        face_detector = self._get_face_detector()
        face_result = face_detector.detect(mp_image)

        faces: list[list[float]] = []
        if face_result.detections:
            for detection in face_result.detections:
                bbox = detection.bounding_box
                x = bbox.origin_x / rgb.shape[1]
                y = bbox.origin_y / rgb.shape[0]
                w = bbox.width / rgb.shape[1]
                h = bbox.height / rgb.shape[0]
                conf = detection.categories[0].score
                faces.append([x, y, w, h, conf])

        pose_landmarker = self._get_pose_landmarker()
        pose_result = pose_landmarker.detect(mp_image)

        poses: list[list[float]] = []
        if pose_result.pose_landmarks:
            for landmarks in pose_result.pose_landmarks:
                person: list[float] = []
                for lm in landmarks:
                    person.extend([lm.x, lm.y, lm.z, 1.0])
                poses.append(person)

        hand_landmarker = self._get_hand_landmarker()
        hand_result = hand_landmarker.detect(mp_image)

        left_hands: list[list[float]] = []
        right_hands: list[list[float]] = []
        if hand_result.hand_landmarks and hand_result.handedness:
            for landmarks, handedness in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                label = handedness[0].category_name
                hand: list[float] = []
                for lm in landmarks:
                    hand.extend([lm.x, lm.y, lm.z])
                if label == "Left":
                    left_hands.append(hand)
                else:
                    right_hands.append(hand)

        return {
            "face_bbox": faces,
            "body_pose": poses,
            "left_hand": left_hands,
            "right_hand": right_hands,
        }
