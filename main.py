import socket
from src.config import Config
from src.server import handle_client


def main(debug: bool = False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        s.bind((Config.Net.HOST, Config.Net.PORT))
        s.listen(1)

        print(f"[PY] Server listening on {Config.Net.HOST}:{Config.Net.PORT}")

        try:
            while True:
                connection, address = s.accept()
                handle_client(connection, address, debug=debug)
        except KeyboardInterrupt:
            print("\n[PY] Server stopping...")


if __name__ == "__main__":
    main()
