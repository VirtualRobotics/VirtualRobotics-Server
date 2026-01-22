import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectionConfig:
    HOST: str = "127.0.0.1"
    PORT: int = 5000


class RobotCommands:
    TURN_RIGHT_10 = "ROTATE 10\n"
    TURN_LEFT_10 = "ROTATE -10\n"
    TURN_LEFT_5 = "ROTATE -5\n"
    TURN_RIGHT_5 = "ROTATE 5\n"

    MOVE_FORWARD = "MOVE 0,2\n"
    MOVE_BACK = "MOVE -0,1\n"
    STOP = "MOVE 0\n"


@dataclass(frozen=True)
class NavParams:
    # Basic Limits
    STOP_THRESHOLD: int = 1500
    TOO_CLOSE_THRESHOLD: int = 2000
    STEER_DEADZONE: float = 0.09

    # Target Logic (Red)
    MIN_TARGET_AREA: int = 50
    TARGET_ALIGN_TOLERANCE: float = 0.1

    # Obstacle Detection (Blue ROI)
    ROI_CENTER_START: float = 0.4
    ROI_CENTER_END: float = 0.6

    BLOCKAGE_THRESHOLD: float = 0.5  # >50% pixels means blocked
    DEAD_END_SIDE_RATIO: float = 0.30  # Side walls threshold

    # Wall Hugging / Avoidance
    WALL_CLOSE_RATIO: float = 0.6
    WALL_VERY_CLOSE_RATIO: float = 0.8

    # Movement Logic
    HEADING_ALIGNED_DIFF: float = 0.3  # Path is considered straight
    CENTERING_DEADZONE: float = 0.05  # Ignore small asymmetries
    CENTERING_CORRECTION: float = 0.25  # Threshold to trigger centering turn
    OSCILLATION_LIMIT: int = 4


class ColorConfig:
    TARGET_RED_LOWER_1 = np.array([0, 100, 80])
    TARGET_RED_UPPER_1 = np.array([10, 255, 255])

    TARGET_RED_LOWER_2 = np.array([170, 100, 80])
    TARGET_RED_UPPER_2 = np.array([180, 255, 255])

    WALL_BLUE_LOWER = np.array([95, 80, 60])
    WALL_BLUE_UPPER = np.array([135, 255, 255])


class Config:
    Net = ConnectionConfig()
    Cmd = RobotCommands
    Nav = NavParams()
    Colors = ColorConfig
