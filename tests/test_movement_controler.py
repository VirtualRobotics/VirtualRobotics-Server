import pytest
import numpy as np
import cv2
from dataclasses import dataclass
from src.movement_controller import MovementController


# --- 1. MOCK CONFIGURATION ---
# Mock configuration classes to keep tests independent of config.py


@dataclass
class MockColors:
    # Standard HSV ranges for OpenCV
    TARGET_RED_LOWER_1 = np.array([0, 100, 100])
    TARGET_RED_UPPER_1 = np.array([10, 255, 255])
    TARGET_RED_LOWER_2 = np.array([170, 100, 100])
    TARGET_RED_UPPER_2 = np.array([180, 255, 255])

    WALL_BLUE_LOWER = np.array([100, 100, 100])
    WALL_BLUE_UPPER = np.array([140, 255, 255])


@dataclass
class MockCmd:
    MOVE_FORWARD = "MOVE 0,2\n"
    TURN_LEFT_5 = "ROTATE -5\n"
    TURN_RIGHT_5 = "ROTATE 5\n"
    TURN_LEFT_10 = "ROTATE -10\n"
    TURN_RIGHT_10 = "ROTATE 10\n"


@dataclass
class MockNavParams:
    MIN_TARGET_AREA = 10
    TARGET_ALIGN_TOLERANCE = 0.1
    ROI_CENTER_START = 0.4
    ROI_CENTER_END = 0.6
    BLOCKAGE_THRESHOLD = 0.5
    DEAD_END_SIDE_RATIO = 0.3
    WALL_VERY_CLOSE_RATIO = 0.8
    HEADING_ALIGNED_DIFF = 0.3
    CENTERING_DEADZONE = 0.05
    CENTERING_CORRECTION = 0.25
    OSCILLATION_LIMIT = 4


class MockConfig:
    Colors = MockColors()
    Cmd = MockCmd()
    Nav = MockNavParams()


# --- 2. FIXTURES & HELPERS ---


@pytest.fixture
def controller():
    """Create a fresh controller instance before each test."""
    return MovementController(MockConfig())


def create_image(color_bgr, position=None, size=(100, 100)):
    """
    Helper to create a 100x100 image.
    color_bgr: tuple (B, G, R)
    position: 'left', 'right', 'center', 'bottom', or None (full image)
    """
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    if position is None:
        return img  # Returns black image

    # Define rectangle coordinates based on scenario
    if position == "center_small":  # Small center target
        cv2.rectangle(img, (45, 45), (55, 55), color_bgr, -1)
    elif position == "right_small":  # Target on the right
        cv2.rectangle(img, (80, 45), (90, 55), color_bgr, -1)
    elif position == "left_small":  # Target on the left
        cv2.rectangle(img, (10, 45), (20, 55), color_bgr, -1)
    elif position == "left_wall":  # Wall on the left (bottom half)
        cv2.rectangle(img, (0, 50), (40, 100), color_bgr, -1)
    elif position == "right_wall":  # Wall on the right (bottom half)
        cv2.rectangle(img, (60, 50), (100, 100), color_bgr, -1)
    elif position == "dead_end":  # Wall across full width (bottom half)
        cv2.rectangle(img, (0, 50), (100, 100), color_bgr, -1)

    return img


# --- 3. TESTS ---


def test_empty_space_moves_forward(controller):
    """No target, no obstacles -> move forward."""
    img = create_image(None)
    cmd = controller.decide_command(img)
    assert cmd == MockConfig.Cmd.MOVE_FORWARD


def test_target_logic_priority(controller):
    """Red target should be detected and prioritized over empty space."""
    # Create red dot (BGR: 0, 0, 255) in center
    img = create_image((0, 0, 255), "center_small")
    cmd = controller.decide_command(img)
    assert cmd == MockConfig.Cmd.MOVE_FORWARD


def test_target_tracking_right(controller):
    """Red target on right -> turn right."""
    img = create_image((0, 0, 255), "right_small")
    cmd = controller.decide_command(img)
    assert cmd == MockConfig.Cmd.TURN_RIGHT_5


def test_target_tracking_left(controller):
    """Red target on left -> turn left."""
    img = create_image((0, 0, 255), "left_small")
    cmd = controller.decide_command(img)
    assert cmd == MockConfig.Cmd.TURN_LEFT_5


def test_obstacle_avoidance_left_wall(controller):
    """Blue wall on left -> escape right."""
    # Blue in BGR is (255, 0, 0)
    img = create_image((255, 0, 0), "left_wall")

    # Optional: check internal analysis state
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    analysis = controller._analyze_obstacles(hsv, 100, 100)
    assert analysis.left_ratio > analysis.right_ratio

    cmd = controller.decide_command(img)
    # Expect right correction (centering or avoidance)
    assert cmd in [MockConfig.Cmd.TURN_RIGHT_5, MockConfig.Cmd.TURN_RIGHT_10]


def test_obstacle_avoidance_right_wall(controller):
    """Blue wall on right -> escape left."""
    img = create_image((255, 0, 0), "right_wall")
    cmd = controller.decide_command(img)
    assert cmd in [MockConfig.Cmd.TURN_LEFT_5, MockConfig.Cmd.TURN_LEFT_10]


def test_dead_end_behavior(controller):
    """Dead end -> sharp turn."""
    img = create_image((255, 0, 0), "dead_end")

    cmd = controller.decide_command(img)

    # Dead-end forces sharp turn (TURN_10)
    assert cmd in [MockConfig.Cmd.TURN_LEFT_10, MockConfig.Cmd.TURN_RIGHT_10]

    # Check if maneuver flag is set
    assert controller._is_turning_maneuver is True


def test_target_overrides_obstacles(controller):
    """TARGET (Red) overrides WALL (Blue)."""
    img = create_image((255, 0, 0), "left_wall")

    # Manually draw red target on the right
    cv2.rectangle(img, (80, 45), (90, 55), (0, 0, 255), -1)

    cmd = controller.decide_command(img)

    # Normally left wall triggers right turn via avoidance logic.
    # But since target is found on the right, it should return Target Logic command.
    assert cmd == MockConfig.Cmd.TURN_RIGHT_5
