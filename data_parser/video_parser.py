import os
import sys
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import math,time,sys
import cv2 as cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from matplotlib import colors

draw_colors = list(colors.CSS4_COLORS.keys())
def getRawSignal(file):
    #cap = cv2.VideoCapture(r"D:\Exp\opencv3\rec\1797089410_1.avi")
    #filename = r"D:\Exp\realsense\rec\1795852138.avi"
    filename = file
    fpath = filename.split(os.sep)
    out_fname = fpath[-3]+"_"+fpath[-2]+"out.npy"
    base=os.path.basename(filename)
    fname = os.path.splitext(base)[0]

    cap = cv2.VideoCapture(filename)


    count = 0
    fpose = []


    def rotation(image, angleInDegrees):
        h, w = image.shape[:2]
        img_c = (w / 2, h / 2)

        rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

        rad = math.radians(angleInDegrees)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot[0, 2] += ((b_w / 2) - img_c[0])
        rot[1, 2] += ((b_h / 2) - img_c[1])

        outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        return outImg


    #kf_face_p = [KalmanFilter1D(Q=0.001,R=2),KalmanFilter1D(Q=0.001,R=2),KalmanFilter1D(Q=0.001,R=2)]
    init_rot_angle =  0
    last_face_width = 0
    raw_sig = []
    mean_box = []
    t_now = time.time_ns()
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    n_loop = 0

    face_landmarks_ext_idx = [[66, 69], [104, 105], [68, 63], [71, 70],[35,226],[232,121], #473
                              [108,107],[151,9],[337,336],[299,296],#477
                              [452,350],[454,366],[234,93],[43,202],#481
                              [454,323],[296,299],[333,334],[298,293],[301,300],[446,265]]#487

    # face_landmarks = results.multi_face_landmarks[0].landmark
    rois = {
        "forehead": [67, 69, 468, 474, 475, 476,477,299,297,338,10,109],
        "nose_up":[474, 107,55,193,168,417, 285, 336, 476, 475],
        "noise_bridge":[456, 248, 195, 3, 236, 126, 47, 121, 232, 233, 245, 193, 168, 417, 465, 453, 452, 350, 277, 355 ],
        "noise":[371, 355, 456, 248, 195, 3, 236, 126, 142, 203,92, 165, 167, 164, 393, 391,322, 423 ],
        "left_forehead": [ 234, 127, 162, 21, 54, 103, 67,69, 468, 469, 470, 471, 156, 472, 31,111,116,227],
        "right_forehead": [ 454,356,389,251,284,332,297,299,483,484,485,486,383,487,261,340,345,447],
        "left_cheek": [205, 203, 142,126, 47,121,473, 231,230,229,228,31,111,116,227,  234, 480],
        "right_cheek":[425, 423, 371, 355,277, 350, 478, 451, 450, 449, 448, 261, 340,345,447,  454,482,366 ],

        "left_cheek_down": [480, 205, 203, 92, 186,57, 212, 192, 213, 132, 93],
        "chain": [106, 204, 211, 150, 149, 176, 148, 152, 377, 400, 378, 379, 431,424, 335, 406, 313, 18, 83, 182],
        "right_cheek_down":[366,425,423,322, 410,287,432,416,361,401,323,482],
        "left_chain": [212, 192,   132,58, 172, 136, 150, 211, 204, 106, 43,57],
        "right_chain":[432,416,361,288,397,365,379,431,424,335,273,287],
    }

    while True:
        ret, frame = cap.read()
        if not ret:# or cap.get(cv2.CAP_PROP_POS_FRAMES) == 200:
            np.save("./data/raw/"+out_fname,raw_sig)
            exit(0)
        #frame = rotation(frame,init_rot_angle)
        n_loop+=1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width = image.shape[1]
        height = image.shape[0]
        # [landmarks, info], with info->x_center ,y_center, r, g, b
        ldmks = np.zeros((468, 5), dtype=np.float32)
        ldmks[:, 0] = -1.0
        ldmks[:, 1] = -1.0
                ### face landmarks ###
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            face_landmarks = np.array([[p.x,p.y] for p in results.multi_face_landmarks[0].landmark])
            face_landmarks[face_landmarks>1]=1
            face_landmarks = (face_landmarks*[width,height]).astype(np.int32)
            min_pt = np.max([np.min(face_landmarks,0)-20,[0,0]],0)
            max_pt = np.min([np.max(face_landmarks,0)+20,[width,height]],0)

            face_landmarks -= min_pt
            face_landmarks_ext = np.mean(face_landmarks[face_landmarks_ext_idx], 1).astype(np.int32)
            face_landmarks = np.vstack([face_landmarks,face_landmarks_ext])

            face_image = image[min_pt[1]:max_pt[1],min_pt[0]:max_pt[0]]
            image_show = face_image.copy()

            sig_out = []
            for i, ldmks_idx in enumerate(rois.values()):
                ldmks = np.array([face_landmarks[ldmks_idx]])
                mask = np.zeros(image_show.shape[:2],dtype=np.uint8)
                if i>=8:
                    ldmks = cv2.convexHull(ldmks,returnPoints=True).swapaxes(0,1)
                mask = cv2.fillPoly(mask, ldmks, 255)
                image_show = cv2.addWeighted((cv2.bitwise_and(image_show, image_show, mask=255 - mask)*
                                             colors.to_rgb(draw_colors[0])).astype(np.uint8), 0.5, image_show, 0.5,
                                             0.0)
                m = np.array(cv2.mean(face_image, mask))
                m[3] = cv2.contourArea(ldmks)
                sig_out.append(m)

            raw_sig.append(sig_out)
            # scale = 1000/max(image_show.shape)
            # out_image = cv2.resize(image_show,None,fx=scale,fy=scale)
            # for i,ldmk in enumerate(face_landmarks):
            #     ldmk_t = (ldmk*scale).astype(np.int32)
            #     cv2.circle(out_image,ldmk_t,1,(128,0,128),1)
            #     cv2.putText(out_image,str(i),ldmk_t,cv2.FONT_HERSHEY_SIMPLEX,0.5,(12,0,128),1,cv2.LINE_AA)
            # cv2.imshow("",out_image)
            # cv2.waitKey(0)


        continue


if __name__ == "__main__":
    block = True
    if len(sys.argv) != 1:
        plt.ion()
        param = sys.argv[1]
        getRawSignal(param)
    else:
        #getRawSignal(r"E:\数据集压缩包备份\UBFC\UBFC2\subject9\vid.avi")
        getRawSignal(r"/tank/数据集/ECG-Fitness/视频版/07/03/c920-1.avi")
