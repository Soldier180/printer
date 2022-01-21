#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import json

MAX_DGRAM = 2**16

def dump_buffer(s):
    """ Emptying buffer frame """
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        print(seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            print("finish emptying buffer")
            break
config = json.load(open('config.json'))
def main():
    """ Getting image udp frame &
    concate before decode and output image """
    
    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((config["video_stream_config"]["ip"], int(config["video_stream_config"]["port"])))
    dat = b''
    dump_buffer(s)

    while True:
        try:
            seg, addr = s.recvfrom(MAX_DGRAM)
            if struct.unpack("B", seg[0:1])[0] > 1:
                dat += seg[1:]
            else:
                dat += seg[1:]
                img = cv2.imdecode(np.frombuffer(dat, dtype=np.uint8), 1)
                if img is not None:
                    cv2.imshow('Q - Quit', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                dat = b''
        except Exception as e:
            break
            cv2.destroyAllWindows()
            s.close()


    # cap.release()
    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()
