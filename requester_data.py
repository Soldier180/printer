# #!/usr/bin/env python3
#
# import socket
#
# HOST = '127.0.0.1'  # The server's hostname or IP address
# PORT = 12346        # The port used by the server
#
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     s.sendall(b'5')
#     data = s.recv(1024)
#
# print('Received', repr(data))

import socket
import sys
import struct

HOST, PORT = "localhost", 9999
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