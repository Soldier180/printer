
import socket
import struct
import json


config = json.load(open('config.json'))

HOST, PORT = config["latepanda_ip"], int(config["width_server_port"])
data = b'\7e'

# Создать сокет (SOCK_STREAM означает TCP сокет)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Подключиться к серверу и отправить данные
    sock.connect((HOST, PORT))
    sock.sendall(data)

    # Получение данных с сервера и завершение работы
    received = sock.recv(1024)
    v = struct.unpack('<f', received)

print("Sent:     {}".format(data))
print("Received:", v[0])