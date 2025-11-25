import numpy as np

HOST = "127.0.0.1"
PORT = 5000
IMAGE_PATH = r"C:/Users/oskar/AppData/LocalLow/DefaultCompany/VirtualRobotics\agent_frame.png"


TURN_RIGHT_20 = "ROTATE 20\n"
TURN_LEFT_20 = "ROTATE -20\n"
MOVE_FORWARD = "MOVE 1\n"
TURN_LEFT_10 = "ROTATE -10\n"
TURN_RIGHT_10 = "ROTATE 10\n"
STOP = "MOVE 0\n"

STOP_THRESHOLD = 2000

LOWER_RED1 = np.array([0, 100, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 100, 80])
UPPER_RED2 = np.array([180, 255, 255])