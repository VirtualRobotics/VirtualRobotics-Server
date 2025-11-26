import socket
from src.config import HOST, PORT
from src.api.server import handle_client

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[PY] Serwer nasłuchuje na {HOST}:{PORT}")
    while True:
        connection, address = s.accept()
        handle_client(connection, address, debug=True)