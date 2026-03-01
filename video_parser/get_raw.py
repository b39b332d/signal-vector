import sys,os
sys.path.insert(0,'.')
from data_parser import data_align

def parse_video(file):
    video_parser_handler = data_align.GetDataWrapper(file)
    if "right_Video-0" in file:
        video_parser_handler.process_video(show = False,autoskip=False,small_roi=True)#,vid_roi=((300,150),(450,300)))
    else:
        video_parser_handler.process_video(show = False,autoskip=True)#,vid_roi=((300,150),(450,300)))

    return



if __name__ == "__main__":
    if len(sys.argv) != 1:
        param = sys.argv[1] 
        parse_video(param)
    else:
        # parse_video(r"/tank/数据集/ECG-Fitness/视频版/12/01/c920-1.avi")

        #parse_video(r"../rec/1734256199/MF- USB Camera3 @vid_0bda_Video-0.avi")
        # parse_video(r"../rec/1734363177/MF- USB Camera3 @vid_0bda_Video-0.avi")
        parse_video(r"/tank/数据集/PhyRecorder/202412/1734427486/right_Video-0.avi")
        # parse_video(r"/tank/数据集/PURE/01-01/01-01.json")
        # parse_video(r"/tank/数据集/iBVP/p21_d/p21_d_bvp.csv")
        # parse_video(r"/tank/数据集/ECG-Fitness/视频版/01/01/c920-1.avi")
        # parse_video(r"/tank/数据集/UBFC/UBFC2/subject37/vid.avi")
        # parse_video(r"/tank/数据集/LGI_PPGI/id2/angelo/angelo_rotation/cv_camera_sensor_stream_handler.avi")
        # parse_video(r"/tank/数据集/iBVP/p10_a/p10_a_bvp.csv")
        # parse_video(r"/tank/在读/wanbingbing/rec/1729409989_ycq3m1/2736_vid.avi")