from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from enum import ReprEnum, unique
from types import MappingProxyType
from typing import Final

import colex
import structlog
import keyboard

logger = structlog.get_logger()
logger.info("Importing fat modules")

import cv2  # noqa: E402 Logging
import mediapipe as mp  # noqa: E402 Logging
from mediapipe.tasks.python import vision  # noqa: E402 Logging

logger.info("Done loading fat modules")

from charz import (  # noqa: E402 Logging
    Engine,
    Screen,
    Clock,
    Scene,
    Sprite,
    Vec2,
    group,
)


type Char = str  # Length of 1
type NonNegative[T] = T
type LandmarkIndex = NonNegative[int]
type HandPointIndexes = list[int]
type HandLandmarkerResult = vision.HandLandmarkerResult  # type: ignore
type HandLandmarker = vision.HandLandmarker  # type: ignore


MODEL_PATH: Final = Path(__file__).parent.parent / "hand_landmarker.task"
ACTIVATION_DISTANCE: Final[float] = 0.05
CONNECTION_PAIRS: Final[list[tuple[LandmarkIndex, LandmarkIndex]]] = [
    *(
        (idx, idx + 1)
        for finger in range(5)
        for idx in range(finger * 4 + 1, finger * 4 + 4)
    ),
    (0, 1),
]
"""_Start_ and _end_ points of connectors, between **hand landmarks**, as _indexes_."""


class Sentinel(type):
    __new__: None


class HandSegment(metaclass=Sentinel): ...


@group(HandSegment)
class HandPixel(Sprite): ...


@unique
class PixelType(tuple[colex.ColorValue, Char], ReprEnum):
    LANDMARK = (colex.GRAY, "X")
    CONNECTOR = (colex.DIM_GRAY, "x")
    FINGER_TIP = (colex.GRAY, "#")

    @property
    def color(self) -> colex.ColorValue:
        return self[0]

    @property
    def char(self) -> Char:
        return self[1]


HAND_INDEX_TO_PIXEL_TYPE: Final = MappingProxyType[LandmarkIndex, PixelType](
    {
        4: PixelType.FINGER_TIP,
        8: PixelType.FINGER_TIP,
        12: PixelType.FINGER_TIP,
        16: PixelType.FINGER_TIP,
        20: PixelType.FINGER_TIP,
    }
)


class App(Engine):
    clock = Clock(fps=24)
    screen = Screen(auto_resize=True)

    def update(self) -> None:
        if keyboard.is_pressed("q") or keyboard.is_pressed("space"):
            self.is_running = False

    def clear_all_hands(self) -> None:
        for node in Scene.current.get_group_members(HandSegment, type_hint=Sprite):
            node.queue_free()

    def draw_connectors(self, points: list[Vec2]):
        for landmark_index_start, landmark_index_end in CONNECTION_PAIRS:
            start_point = points[landmark_index_start]
            end_point = points[landmark_index_end]
            start_screen_point = Vec2(
                start_point.x * self.screen.width,
                start_point.y * self.screen.height,
            )
            end_screen_point = Vec2(
                end_point.x * self.screen.width,
                end_point.y * self.screen.height,
            )

            total_step_length = start_screen_point.distance_to(end_screen_point)
            direction = start_screen_point.direction_to(end_screen_point)
            angle = direction.angle()
            # Points: [Start, End> | [Start, End]
            for step in range(int(total_step_length)):
                line_segment_location = start_screen_point + direction * step
                HandPixel(
                    position=line_segment_location,
                    rotation=angle,
                    texture=[PixelType.CONNECTOR.char],
                    color=PixelType.CONNECTOR.color,
                )
            # Check if length is float number
            if int(total_step_length) != total_step_length:
                # Point: End
                HandPixel(
                    position=start_screen_point,
                    rotation=angle,
                    texture=[PixelType.CONNECTOR.char],
                    color=PixelType.CONNECTOR.color,
                )

    def draw_hand(self, points: list[Vec2]) -> None:
        self.draw_connectors(points)

        for idx, point in enumerate(points):
            screen_point = point * self.screen.size
            pixel = HAND_INDEX_TO_PIXEL_TYPE.get(idx, PixelType.LANDMARK)
            HandPixel(
                position=screen_point,
                texture=[pixel.char],
                color=pixel.color,
            )

    def on_detection_result(
        self,
        result: HandLandmarkerResult,
        _output_image: mp.Image,
        _timestamp_ms: int,
    ) -> None:
        self.clear_all_hands()

        if not result.hand_landmarks or not result.handedness:
            return

        for hand_landmarks in result.hand_landmarks:
            self.draw_hand(
                [Vec2(landmark.x, landmark.y) for landmark in hand_landmarks]
            )


def collect_and_send_for_detection(
    app: App,
    capture: cv2.VideoCapture,
    landmarker: HandLandmarker,
) -> None:
    if not capture.isOpened():
        app.is_running = False
        return

    success, frame = capture.read()
    if not success:
        logger.warn("Quitting because of unsuccessful video capture")
        app.is_running = False
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    flipped_frame = cv2.flip(rgb_frame, 1)  # Flip horizontally
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_frame)
    timestamp_ms = time.time() * 1000
    landmarker.detect_async(image, int(timestamp_ms))


def main() -> None:
    app = App()
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=app.on_detection_result,
        num_hands=4,  # Max num of hands
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        logger.info("Opening video capture")
        capture = cv2.VideoCapture(index=0)
        logger.info("Done opening video capture")
        app.frame_tasks[30] = partial(
            collect_and_send_for_detection,
            capture=capture,
            landmarker=landmarker,
        )
        app.run()
        capture.release()


if __name__ == "__main__":
    main()
