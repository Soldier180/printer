# #!/usr/bin/env python3
# import  socketserver
# import socket
#
# HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
# PORT = 12346        # Port to listen on (non-privileged ports are > 1023)
#
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print('Connected by', addr)
#         while True:
#             data = conn.recv(1024)
#             print(data)
#             if not data:
#                 continue
#             conn.sendall(data)

import socketserver
from socketserver import TCPServer, BaseRequestHandler
import numpy as np
import struct
from threading import Thread, Event







def get_rand():
    return np.random.rand() * 50

class Server(TCPServer, Thread):
    def __init__(self, server_address, handler, method, parent=None):
        TCPServer.__init__(self, server_address, handler, bind_and_activate=True)
        Thread.__init__(self, parent)
        self._stop = Event()
        self.get_method = method

    def stop(self):
        self._stop.set()
        self.shutdown()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        self.serve_forever()



class MyTCPHandler(BaseRequestHandler):
    def handle(self):
        # self.request - это TCP - сокет, подключённый к клиенту
        #b'\x7e\xdd'
        self.data = self.request.recv(1024)
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        v = self.server.get_method()

        #v = rnd.get_random()
        print("value", v)
        self.request.sendall(bytearray(struct.pack("<f", v)))



if __name__ == "__main__":
    HOST, PORT = "localhost", 9999

    # Создать серверный биндинг localhost на порту 9999
    server = Server(("localhost", 9999), MyTCPHandler, get_rand)
    server.start()
    #server.serve_forever()
    # with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
    #     # Активировать сервер; это будет продолжаться до тех пор, пока вы не прервёте
    #     # программу с помощью Ctrl-C
    #     server.serve_forever()