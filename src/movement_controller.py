import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ObstacleAnalysis:
    left_ratio: float
    right_ratio: float
    center_blocked: bool
    diff: float
    blue_ratio: float
    is_dead_end: bool


class MovementController:
    def __init__(self, config: Any):
        self.cfg = config

        # Internal state
        self._diff_history = deque(maxlen=5)
        self._oscillation_counter = 0
        self._last_cmd_side = 0
        self._last_turn_direction = 0  # 0: None, -1: Left, 1: Right
        self._is_turning_maneuver = False

        # Pre-allocate morphological kernel for optimization
        self._morph_kernel = np.ones((3, 3), np.uint8)

    def reset_state(self):
        self._last_turn_direction = 0
        self._is_turning_maneuver = False
        self._oscillation_counter = 0
        self._last_cmd_side = 0
        self._diff_history.clear()

    def decide_command(self, img: np.ndarray) -> str:
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Target Logic (Red)
        target_cmd = self._process_target_logic(hsv, w)
        if target_cmd:
            return target_cmd

        # 2. Obstacle Logic (Blue)
        analysis = self._analyze_obstacles(hsv, h, w)

        # 3. Update State
        self._update_oscillation_state(analysis.diff)
        self._update_turn_direction(analysis)

        # 4. Resolve Movement
        return self._resolve_movement_logic(analysis)

    def _process_target_logic(self, hsv: np.ndarray, w: int) -> Optional[str]:
        mask1 = self._create_mask(
            hsv, self.cfg.Colors.TARGET_RED_LOWER_1, self.cfg.Colors.TARGET_RED_UPPER_1
        )
        mask2 = self._create_mask(
            hsv, self.cfg.Colors.TARGET_RED_LOWER_2, self.cfg.Colors.TARGET_RED_UPPER_2
        )
        red_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) <= self.cfg.Nav.MIN_TARGET_AREA:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        offset_x = (cx / w) - 0.5

        self._is_turning_maneuver = False
        self._oscillation_counter = 0

        tol = self.cfg.Nav.TARGET_ALIGN_TOLERANCE

        if offset_x > tol:
            self._last_turn_direction = 1
            return self.cfg.Cmd.TURN_RIGHT_5
        elif offset_x < -tol:
            self._last_turn_direction = -1
            return self.cfg.Cmd.TURN_LEFT_5

        return self.cfg.Cmd.MOVE_FORWARD

    def _analyze_obstacles(self, hsv: np.ndarray, h: int, w: int) -> ObstacleAnalysis:
        blue_mask = self._create_mask(
            hsv, self.cfg.Colors.WALL_BLUE_LOWER, self.cfg.Colors.WALL_BLUE_UPPER
        )

        # Analyze bottom 50% of the image
        y0 = int(h * 0.50)
        roi = blue_mask[y0:h, :]

        total_pixels = float(roi.size) or 1.0
        blue_ratio = cv2.countNonZero(roi) / total_pixels

        mid_w = w // 2
        left_roi = roi[:, :mid_w]
        right_roi = roi[:, mid_w:]

        left_ratio = cv2.countNonZero(left_roi) / float(left_roi.size or 1)
        right_ratio = cv2.countNonZero(right_roi) / float(right_roi.size or 1)

        raw_diff = right_ratio - left_ratio
        self._diff_history.append(raw_diff)
        avg_diff = sum(self._diff_history) / len(self._diff_history)

        # Smoothing differential to reduce jitter
        center_start = int(w * self.cfg.Nav.ROI_CENTER_START)
        center_end = int(w * self.cfg.Nav.ROI_CENTER_END)
        center_roi = roi[:, center_start:center_end]
        center_blocked = (
            cv2.countNonZero(center_roi) / float(center_roi.size or 1)
        ) > self.cfg.Nav.BLOCKAGE_THRESHOLD

        side_threshold = self.cfg.Nav.DEAD_END_SIDE_RATIO

        # Dead end condition: Center blocked AND significant walls on both sides
        is_dead_end = center_blocked and (
            left_ratio > side_threshold and right_ratio > side_threshold
        )

        return ObstacleAnalysis(
            left_ratio=left_ratio,
            right_ratio=right_ratio,
            center_blocked=center_blocked,
            diff=avg_diff,
            blue_ratio=blue_ratio,
            is_dead_end=is_dead_end,
        )

    def _update_oscillation_state(self, diff: float):
        current_side = -1 if diff > 0.1 else (1 if diff < -0.1 else 0)

        if (
            current_side != 0
            and self._last_cmd_side != 0
            and current_side != self._last_cmd_side
        ):
            self._oscillation_counter += 1
        else:
            self._oscillation_counter = max(0, self._oscillation_counter - 1)

        self._last_cmd_side = current_side

    def _update_turn_direction(self, analysis: ObstacleAnalysis):
        if not analysis.is_dead_end:
            deadzone = self.cfg.Nav.CENTERING_DEADZONE
            if analysis.left_ratio > analysis.right_ratio + deadzone:
                self._last_turn_direction = 1
            elif analysis.right_ratio > analysis.left_ratio + deadzone:
                self._last_turn_direction = -1

        # If stuck in a dead end without history, force a default direction (Right)
        if analysis.is_dead_end and self._last_turn_direction == 0:
            self._last_turn_direction = 1

    def _resolve_movement_logic(self, metrics: ObstacleAnalysis) -> str:
        # A. Clear path
        path_clear = not metrics.center_blocked
        heading_aligned = abs(metrics.diff) < self.cfg.Nav.HEADING_ALIGNED_DIFF

        if path_clear and heading_aligned:
            self._is_turning_maneuver = False
            return self.cfg.Cmd.MOVE_FORWARD

        # B. Escape Dead End
        if metrics.is_dead_end:
            self._is_turning_maneuver = True
            return (
                self.cfg.Cmd.TURN_LEFT_10
                if self._last_turn_direction == -1
                else self.cfg.Cmd.TURN_RIGHT_10
            )

        # C. Oscillation Damping (ignore jitter in open space)
        if (
            self._oscillation_counter > self.cfg.Nav.OSCILLATION_LIMIT
            and not metrics.center_blocked
        ):
            self._oscillation_counter = 0
            self._is_turning_maneuver = False
            return self.cfg.Cmd.MOVE_FORWARD

        # D. Centering (Minor corrections in corridors)
        if not metrics.center_blocked and metrics.blue_ratio > 0.05:
            correction = self.cfg.Nav.CENTERING_CORRECTION
            if metrics.diff > correction:
                return self.cfg.Cmd.TURN_LEFT_5
            if metrics.diff < -correction:
                return self.cfg.Cmd.TURN_RIGHT_5

        # E. Standard Avoidance (Corner or blocked path)
        is_wall_too_close = metrics.blue_ratio >= self.cfg.Nav.WALL_VERY_CLOSE_RATIO

        if metrics.center_blocked or self._is_turning_maneuver or is_wall_too_close:
            self._is_turning_maneuver = True
            return (
                self.cfg.Cmd.TURN_LEFT_10
                if self._last_turn_direction == -1
                else self.cfg.Cmd.TURN_RIGHT_10
            )

        return self.cfg.Cmd.MOVE_FORWARD

    def _create_mask(
        self, hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=2)
        mask = cv2.dilate(mask, self._morph_kernel, iterations=1)
        return mask
