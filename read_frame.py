import time
import cv2
import os

IMAGE_PATH = r"C:/Users/oskar/AppData/LocalLow/DefaultCompany/VirtualRobotics\agent_frame.png"


def main():
    if not os.path.exists(IMAGE_PATH):
        print("Plik jeszcze nie istnieje, odpal najpierw Play w Unity...")

    last_mtime = 0

    while True:
        if os.path.exists(IMAGE_PATH):
            mtime = os.path.getmtime(IMAGE_PATH)
            if mtime != last_mtime:
                last_mtime = mtime
                img = cv2.imread(IMAGE_PATH)
                if img is None:
                    print("Nie udało się wczytać obrazu")
                else:
                    print("Nowa klatka:", img.shape)
                    cv2.imshow("Agent View", img)
                    cv2.waitKey(1)
        time.sleep(0.1)  # nie miel grubo dysku


if __name__ == "__main__":
    main()
