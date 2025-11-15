import socket

HOST = "127.0.0.1"
PORT = 5000

def handle_client(connection, address):
    print(f"[PY] Połączono z {address}")
    with connection:
        while True:
            data = connection.recv(1024)
            if not data:
                print("[PY] Klient się rozłączył")
                break

            text = data.decode("utf-8").strip()
            print(f"[PY] Odebrano z Unity: {text}")

            response = "ROTATE 10\n"
            connection.sendall(response.encode("utf-8"))
            print(f"[PY] Wysłano do Unity: {response.strip()}")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[PY] Serwer nasłuchuje na {HOST}:{PORT}")
        while True:
            connection, address = s.accept()
            handle_client(connection, address)

if __name__ == "__main__":
    main()
