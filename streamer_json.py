import socket
import json

HOST = '172.16.234.76'  # The server's hostname or IP address
PORT = 5556         # The port used by the server

def json_message(direction):
    local_ip = socket.gethostbyname(socket.gethostname())
    data = {
        'sender': local_ip,
        'instruction': direction
    }

    json_data = json.dumps(data, sort_keys=False, indent=2)
    print("data %s" % json_data)

    send_message(json_data + ";")

    return json_data



def send_message(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data.encode())
        data = s.recv(1024)

    print('Received', repr(data))

json_message("SOME_DIRECTION")