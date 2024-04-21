
import sys,os
sys.path.insert(0,'.')
import sys
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import math,time,sys
import cv2 as cv2
import mediapipe as mp
from matplotlib import colors
import utils.skin_extraction_methods

draw_colors = list(colors.CSS4_COLORS.keys())
def getRawSignal(file):
    #cap = cv2.VideoCapture(r"D:\Exp\opencv3\rec\1797089410_1.avi")
    #filename = r"D:\Exp\realsense\rec\1795852138.avi"
    filename = file
    fpath = filename.split(os.sep)
    out_fname = fpath[-3]+"_"+fpath[-2]+"out.npy"
    bgr_fname = fpath[-3]+"_"+fpath[-2]+"bgr.npy"
    bgr_sig = []
    base=os.path.basename(filename)
    fname = os.path.splitext(base)[0]

    cap = cv2.VideoCapture(filename)

    count = 0
    fpose = []
    #kf_face_p = [KalmanFilter1D(Q=0.001,R=2),KalmanFilter1D(Q=0.001,R=2),KalmanFilter1D(Q=0.001,R=2)]
    init_rot_angle =  0
    last_face_width = 0
    raw_sig = []
    mean_box = []
    t_now = time.time_ns()

    while True:
        ret, frame = cap.read()
        if not ret:# or cap.get(cv2.CAP_PROP_POS_FRAMES) == 200:
            np.save("./data/ecgfitness/raw/"+out_fname,raw_sig)
            np.save("./data/ecgfitness/raw/"+bgr_fname,bgr_sig)
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
            skin_image,mask = se.extract_skin(image,results.multi_face_landmarks[0])
            mean_bgr = np.array(cv2.mean(skin_image, mask))[2::-1]
            mean_bgr = np.append(mean_bgr,np.sum(mask))
            bgr_sig.append(mean_bgr)

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
