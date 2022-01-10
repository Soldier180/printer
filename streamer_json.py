#file:client.py
from json_cl_server import Client
import time

host = '172.16.234.76'
port = 5556

i=1
while True:
    client = Client()
    client.connect(host, port).send({'test':i})
    i+=1
    response = client.recv()
    print(response)
    client.close()
    time.sleep(1)