
import sys,os
sys.path.insert(0,'.')
from data_parser import video_parser

def parse_video(file):
    video_parser_handler = video_parser.VideoParserWrapper(file)
    full_sigs,rois_sigs = video_parser_handler.process_video()
    video_parser_handler.save_raw(full_sigs,rois_sigs)
    return



if __name__ == "__main__":
    if len(sys.argv) != 1:
        param = sys.argv[1]
        parse_video(param)
    else:
        parse_video(r"/tank/数据集/PURE/01-01.json")
