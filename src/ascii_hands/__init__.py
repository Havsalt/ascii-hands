from __future__ import annotations

from math import pi as PI


import colex
import keyboard
import structlog

# Logging - Placed far up to log laoding of "fat modules"
logger = structlog.get_logger()
logger.info("Importing fat modules")

import cv2
from mediapipe.python.solutions.hands import Hands, HandLandmark

logger.info("Done")

from charz import (
    Engine,
    Screen,
    Clock,
    Scene,
    Group,
    Sprite,
    Label,
    ColliderComponent,
    Hitbox,
    Vec2,
    Vec2i,
)


ACTIVATION_DISTANCE: float = 0.05
CONNECTION_PAIRS: list[tuple[int, int]] = [
    *(
        (index, index + 1)
        for finger in range(5)
        for index in range(finger * 4 + 1, finger * 4 + 4)
    ),
    (0, 1),
]


class ShortLived:
    """Mixin class used as a `tag`.

    Nodes subclassing from this base is meant to only live for `1 display frame`.
    By using `isinstance(<Node>, ShortLived)` you can filter out
    and `queue_free` these nodes.
    """


class DebugMarker(ShortLived, Sprite):
    z_index = 10
    color = colex.RED
    texture = ["X"]
    # visible = False  # TEST


class HandLandmarkVisualPoint(Sprite, ShortLived):
    color = colex.GRAY
    texture = ["X"]


class HandLandmarkVisualConnector(Sprite, ShortLived):
    color = colex.DIM_GRAY
    texture = ["x"]


class FingerTipVisualPoint(Sprite, ShortLived):
    color = colex.GRAY
    texture = ["#"]


def size_of(texture: list[str], /) -> Vec2i:
    return Vec2i(
        len(max(texture, key=len)),
        len(texture),
    )


class GravityBox(Sprite, ColliderComponent):
    _GRAVITY: float = 0.1
    _speed_y: float = -1
    color = colex.PINK
    centered = True
    texture = [
        "#######",
        "#######",
        "#######",
        "#######",
    ]
    texture = [
        "################################",
        "################################",
        "################################",
        "################################",
        "################################",
    ]
    hitbox = Hitbox(size=size_of(texture), centered=True)

    def contains_point(self, point: Vec2) -> bool:
        # Add/subtract half of texture size since centered
        return (
            self.global_position - self.get_texture_size() / 2
            <= point
            <= self.global_position + self.get_texture_size() / 2
        )

    def update(self) -> None:
        # DEBUG
        for corner in self.get_corner_points():
            DebugMarker(position=corner)
        self.position.y += self._speed_y
        if self.is_colliding():
            self.position.y -= self._speed_y
            self._speed_y = 0
            while self.is_colliding():
                self.position.y -= 0.01
            return
        self._speed_y += self._GRAVITY


class App(Engine):
    _SYMBOLS = ("―", "\\", "|", "/", "―", "\\", "|", "/")
    # _SYMBOLS = ["⇒", "⇘", "⇓", "⇙", "⇐", "⇖", "⇑", "⇗"]
    _FRAM_GRABS: int = 2  # 1+
    clock = Clock(fps=20)
    screen = Screen(auto_resize=True)

    def __init__(self) -> None:
        logger.info("Starting program...")
        # Mediapipe setup
        self._hands = Hands(max_num_hands=2)
        self._video_capture = cv2.VideoCapture(0)
        # Node label and boxes
        self._node_count = Label(position=Vec2.ONE, text="Nodes: ???")
        self._boxes = [
            GravityBox(position=Vec2(20, 5)),
            # GravityBox(position=Vec2(28, 5)),
            GravityBox(position=Vec2(36, 5)),
        ]

    def update(self) -> None:
        # Keep box inside screen
        for box in self._boxes:
            half_texture_height = box.get_texture_size().y / 2
            box.set_global_y(
                min(
                    self.screen.height - half_texture_height,
                    box.global_position.y + half_texture_height,
                )
            )
            box._speed_y = 0

        if not self._video_capture.isOpened():
            return
        main_success, image = self._video_capture.read()
        # FPS boost
        for _ in range(self._FRAM_GRABS - 1):
            success, _ = self._video_capture.read()
            if not success:
                break

        if not main_success:
            logger.warning("Ignoring empty camera frame")
            return

        for node in Scene.current.get_group_members(Group.NODE, type_hint=Sprite):
            if isinstance(node, ShortLived):
                node.queue_free()

        # Convert frame
        image = cv2.flip(image, 1)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hands
        results = self._hands.process(rgb_frame)
        if results.multi_hand_landmarks:  # type: ignore
            for hand in results.multi_hand_landmarks:  # type: ignore
                # Render landmarks and connections
                for index, connection_pair in enumerate(CONNECTION_PAIRS):
                    landmark_index_start, landmark_index_end = connection_pair
                    point1 = hand.landmark[landmark_index_start]
                    point2 = hand.landmark[landmark_index_end]
                    segment_index = (point1.z + point2.z) / 2
                    line_start = Vec2(
                        point1.x * self.screen.width,
                        point1.y * self.screen.height,
                    )
                    line_end = Vec2(
                        point2.x * self.screen.width,
                        point2.y * self.screen.height,
                    )
                    HandLandmarkVisualPoint(
                        position=line_start,
                        z_index=segment_index + 1,
                    )

                    total_step_length = line_start.distance_to(line_end)
                    direction = line_start.direction_to(line_end)
                    angle = direction.angle()
                    # Points: [Start, End> | [Start, End]
                    for step in range(int(total_step_length)):
                        line_segment_location = line_start + direction * step
                        HandLandmarkVisualConnector(
                            position=line_segment_location,
                            rotation=angle,
                            z_index=segment_index,
                        )
                    # Check if length is float number
                    if int(total_step_length) != total_step_length:
                        # Point: End
                        HandLandmarkVisualConnector(
                            position=line_start,
                            rotation=angle,
                            z_index=segment_index,
                        )
                # Render tip of fingers
                visual_finger_tips: list[FingerTipVisualPoint] = []
                for index, point in enumerate(hand.landmark):
                    if index not in (4, 8, 12, 16, 20):
                        continue
                    visual_finger_tips.append(
                        FingerTipVisualPoint(
                            position=Vec2(
                                point.x * self.screen.width,
                                point.y * self.screen.height,
                            )
                        )
                    )
                # Check for actions (whether index and thumb is close)
                thumb_position = Vec2(
                    hand.landmark[HandLandmark.THUMB_TIP].x,
                    hand.landmark[HandLandmark.THUMB_TIP].y,
                )
                index_position = Vec2(
                    hand.landmark[HandLandmark.INDEX_FINGER_TIP].x,
                    hand.landmark[HandLandmark.INDEX_FINGER_TIP].y,
                )
                mid_point_global = (
                    thumb_position.lerp(index_position, 0.50) * self.screen.size
                )
                # DEBUG
                DebugMarker(position=mid_point_global, texture=["!"])

                distance = thumb_position.distance_to(index_position)
                if distance < ACTIVATION_DISTANCE:
                    for box in self._boxes:
                        if box.contains_point(mid_point_global):
                            box.global_position = mid_point_global
                            box._speed_y = -1
                            box.global_rotation = (
                                thumb_position.direction_to(index_position)
                                .rotated(PI / 2)
                                .angle()
                            )
                            break

        # Show how many nodes that are in the current scene
        self._node_count.text = (
            f"Nodes: {len(Scene.current.get_group_members(Group.NODE))}"
        )

        # Exit controls
        if keyboard.is_pressed("q"):
            self.is_running = False
            logger.info("Terminating")
            return

    def run(self) -> None:  # Release video capture on exit
        super().run()
        self._video_capture.release()


def main() -> None:
    app = App()
    app.run()
