from __future__ import annotations

import colex
import keyboard
import cv2
from mediapipe.python.solutions.hands import Hands
from charz import Engine, Screen, Clock, Scene, Group, Sprite, Label, Vec2


class ShortLived:
    """Mixin class used as a `tag`.

    Nodes subclassing from this base is meant to only live for `1 display frame`.
    By using `isinstance(<Node>, ShortLived)` you can filter out
    and `queue_free` these nodes.
    """


class HandLandmarkVisualPoint(Sprite, ShortLived):
    color = colex.GRAY
    texture = ["X"]


class HandLandmarkVisualConnector(Sprite, ShortLived):
    color = colex.DIM_GRAY
    texture = ["x"]


class FingerTipVisualPoint(Sprite, ShortLived):
    color = colex.GRAY
    texture = ["#"]


CONNECTION_PAIRS: list[tuple[int, int]] = [
    *(
        (index, index + 1)
        for finger in range(5)
        for index in range(finger * 4 + 1, finger * 4 + 4)
    ),
    (0, 1),
]


class App(Engine):
    _SYMBOLS = ("―", "\\", "|", "/", "―", "\\", "|", "/")
    _FRAM_GRABS: int = 2  # 1+
    # _SYMBOLS = ["⇒", "⇘", "⇓", "⇙", "⇐", "⇖", "⇑", "⇗"]
    clock = Clock(fps=20)
    screen = Screen(auto_resize=True)

    def __init__(self) -> None:
        print("[Startup] Starting program...", end="\r")
        # Mediapipe setup
        self.hands = Hands(max_num_hands=2)
        self.video_capture = cv2.VideoCapture(0)
        self.node_count = Label(position=Vec2.ONE, text="Nodes: ???")

    def update(self) -> None:
        if not self.video_capture.isOpened():
            return
        main_success, image = self.video_capture.read()
        # FPS boost
        for _ in range(self._FRAM_GRABS - 1):
            success, _ = self.video_capture.read()
            if not success:
                break

        if not main_success:
            print("[Warning] Ignoring empty camera frame", end="\r")
            return

        for node in Scene.current.get_group_members(Group.NODE, type_hint=Sprite):
            if isinstance(node, ShortLived):
                node.queue_free()

        # Convert frame
        image = cv2.flip(image, 1)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hands
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:  # type: ignore
            for hand in results.multi_hand_landmarks:  # type: ignore
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

                for index, point in enumerate(hand.landmark):
                    if index not in (4, 8, 12, 16, 20):
                        continue
                    FingerTipVisualPoint(
                        position=Vec2(
                            point.x * self.screen.width,
                            point.y * self.screen.height,
                        )
                    )

        # Show how many nodes that are in the current scene
        self.node_count.text = (
            f"Nodes: {len(Scene.current.get_group_members(Group.NODE))}"
        )

        # Exit controls
        if keyboard.is_pressed("q"):
            self.is_running = False
            print("[Terminate] Terminating", end="\r")
            return

    def run(self) -> None:  # Release video capture on exit
        super().run()
        self.video_capture.release()


def main() -> None:
    app = App()
    app.run()
