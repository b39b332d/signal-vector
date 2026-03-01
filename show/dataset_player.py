import matplotlib.pyplot as plt
import sys,os
sys.path.insert(0,'.')
import numpy as np
from data_parser import data_align 
import cv2
import time


def play(ds_path):
    fs=30
    window_length = fs*10
    data_align_handler = data_align.GetDataWrapper(ds_path,fs,window_length)
    from data_parser.video_parser import VideoParser
    from mjpeg_streamer import MjpegServer, Stream
    server = MjpegServer("0.0.0.0", 28080)
    vp = VideoParser(False)
    stream =None
    i=0
    for img in data_align_handler.read_video():
        print(i)
        i+=1
        if stream is None:
            
            stream = Stream("my_camera", size=(320, int(320/img.shape[1]*img.shape[0])), quality=20, fps=10)

            server.add_stream(stream)
            server.start()
        stream.set_frame(img[:,:,::-1])
        _,_,frame = vp.process_frame(img,True)
        cv2.imshow("0",frame)

        cv2.waitKey(1)
        if i%10 ==0:
            cv2.imwrite(f"./vid_save_frames/img_{i}.jpg",frame)
        # plt.savefig("./out.png")
    

if __name__ == "__main__":
    if len(sys.argv) != 1:
        #plt.ion()
        param = sys.argv[1]
        play(param)
    else:
        play(r"/tank/数据集/MMPD/subject17/p17_8.mat")
        # play(r"/tank/数据集/MMPD/subject15/p15_7.mat")