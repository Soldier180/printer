#file:server.py
from json_cl_server import Server

host = 'LOCALHOST'
port = 5556

server = Server(host, port)

while True:
    server.accept()
    data = server.recv()
    server.send({"response":data})

server.close()