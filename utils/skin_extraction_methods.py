import math
import numpy as np
import os
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
import mediapipe as mp


def bbox2_CPU(img):
    """
    Args:
        img (ndarray): ndarray with shape [rows, columns, rgb_channels].
    Returns:
        Four cropping coordinates (row, row, column, column) for removing black borders (RGB [O,O,O]) from img.
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    nzrows = np.nonzero(rows)
    nzcols = np.nonzero(cols)
    if nzrows[0].size == 0 or nzcols[0].size == 0:
        return -1, -1, -1, -1
    rmin, rmax = np.nonzero(rows)[0][[0, -1]]
    cmin, cmax = np.nonzero(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


class MagicLandmarks():
    """
    This class contains usefull lists of landmarks identification numbers.
    """
    high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    high_prio_nose = [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198,
                      209, 217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440, 456]
    high_prio_left_cheek = [36, 47, 50, 100, 101, 116, 117,
                            118, 119, 123, 126, 147, 187, 203, 205, 206, 207, 216]
    high_prio_right_cheek = [266, 280, 329, 330, 346, 347,
                             347, 348, 355, 371, 411, 423, 425, 426, 427, 436]

    mid_prio_forehead = [8, 9, 21, 68, 103, 251,
                         284, 297, 298, 301, 332, 333, 372, 383]
    mid_prio_nose = [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412]
    mid_prio_left_cheek = [34, 111, 137, 156, 177, 192, 213, 227, 234]
    mid_prio_right_cheek = [340, 345, 352, 361, 454]
    mid_prio_chin = [135, 138, 169, 170, 199, 208, 210, 211,
                     214, 262, 288, 416, 428, 430, 431, 432, 433, 434]
    mid_prio_mouth = [92, 164, 165, 167, 186, 212, 322, 391, 393, 410]
    # more specific areas
    forehead_left = [21, 71, 68, 54, 103, 104, 63, 70,
                     53, 52, 65, 107, 66, 108, 69, 67, 109, 105]
    forehead_center = [10, 151, 9, 8, 107, 336, 285, 55, 8]
    forehoead_right = [338, 337, 336, 296, 285, 295, 282,
                       334, 293, 301, 251, 298, 333, 299, 297, 332, 284]
    eye_right = [283, 300, 368, 353, 264, 372, 454, 340, 448,
                 450, 452, 464, 417, 441, 444, 282, 276, 446, 368]
    eye_left = [127, 234, 34, 139, 70, 53, 124,
                35, 111, 228, 230, 121, 244, 189, 222, 143]
    nose = [193, 417, 168, 188, 6, 412, 197, 174, 399, 456,
            195, 236, 131, 51, 281, 360, 440, 4, 220, 219, 305]
    mounth_up = [186, 92, 167, 393, 322, 410, 287, 39, 269, 61, 164]
    mounth_down = [43, 106, 83, 18, 406, 335, 273, 424, 313, 194, 204]
    chin = [204, 170, 140, 194, 201, 171, 175,
            200, 418, 396, 369, 421, 431, 379, 424]
    cheek_left_bottom = [215, 138, 135, 210, 212, 57, 216, 207, 192]
    cheek_right_bottom = [435, 427, 416, 364,
                          394, 422, 287, 410, 434, 436]
    cheek_left_top = [116, 111, 117, 118, 119, 100, 47, 126, 101, 123,
                      137, 177, 50, 36, 209, 129, 205, 147, 177, 215, 187, 207, 206, 203]
    cheek_right_top = [349, 348, 347, 346, 345, 447, 323,
                       280, 352, 330, 371, 358, 423, 426, 425, 427, 411, 376]
    # dense zones used for convex hull masks
    left_eye = [157,144, 145, 22, 23, 25, 154, 31, 160, 33, 46, 52, 53, 55, 56, 189, 190, 63, 65, 66, 70, 221, 222, 223, 225, 226, 228, 229, 230, 231, 232, 105, 233, 107, 243, 124]
    right_eye = [384, 385, 386, 259, 388, 261, 265, 398, 276, 282, 283, 285, 413, 293, 296, 300, 441, 442, 445, 446, 449, 451, 334, 463, 336, 464, 467, 339, 341, 342, 353, 381, 373, 249, 253, 255]
    mounth = [391, 393, 11, 269, 270, 271, 287, 164, 165, 37, 167, 40, 43, 181, 313, 314, 186, 57, 315, 61, 321, 73, 76, 335, 83, 85, 90, 106]
    # equispaced facial points - mouth and eyes are excluded.
    equispaced_facial_points = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, \
             58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, \
             118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, \
             210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, \
             284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, \
             346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]


class SkinExtractionConvexHull:
    """
        This class performs skin extraction on CPU/GPU using a Convex Hull segmentation obtained from facial landmarks.
    """
    def __init__(self,device='CPU'):
        """
        Args:
            device (str): This class can execute code on 'CPU' or 'GPU'.
        """
        self.device = device
    
    def extract_skin(self,image, face_landmarks):
        """
        This method extract the skin from an image using Convex Hull segmentation.

        Args:
            image (uint8 ndarray): ndarray with shape [rows, columns, rgb_channels].
            ldmks (float32 ndarray): landmarks used to create the Convex Hull; ldmks is a ndarray with shape [num_landmarks, xy_coordinates].

        Returns:
            Cropped skin-image and non-cropped skin-image; both are uint8 ndarray with shape [rows, columns, rgb_channels].
        """
        ldmks = np.zeros((468, 5), dtype=np.float32)
        ldmks[:, 0] = -1.0
        ldmks[:, 1] = -1.0
        ### face landmarks ###
        landmarks = [l for l in face_landmarks.landmark]
        for idx in range(len(landmarks)):
            landmark = landmarks[idx]
            if not ((landmark.HasField('visibility') and landmark.visibility < 0.5)
                    or (landmark.HasField('presence') and landmark.presence < 0.5)):
                coords = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image.shape[1], image.shape[0])
                if coords:
                    ldmks[idx, 0] = coords[1]
                    ldmks[idx, 1] = coords[0]
        ### skin extraction ###

        aviable_ldmks = ldmks[ldmks[:,0] >= 0][:,:2]        
        # face_mask convex hull 
        hull = ConvexHull(aviable_ldmks)
        verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
        img = Image.new('L', image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
        mask = np.expand_dims(mask,axis=0).T

        # left eye convex hull
        left_eye_ldmks = ldmks[MagicLandmarks.left_eye]
        aviable_ldmks = left_eye_ldmks[left_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            left_eye_mask = np.array(img)
            left_eye_mask = np.expand_dims(left_eye_mask,axis=0).T
        else:
            left_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # right eye convex hull
        right_eye_ldmks = ldmks[MagicLandmarks.right_eye]
        aviable_ldmks = right_eye_ldmks[right_eye_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            right_eye_mask = np.array(img)
            right_eye_mask = np.expand_dims(right_eye_mask,axis=0).T
        else:
            right_eye_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)

        # mounth convex hull
        mounth_ldmks = ldmks[MagicLandmarks.mounth]
        aviable_ldmks = mounth_ldmks[mounth_ldmks[:,0] >= 0][:,:2]
        if len(aviable_ldmks) > 3:
            hull = ConvexHull(aviable_ldmks)
            verts = [(aviable_ldmks[v,0], aviable_ldmks[v,1]) for v in hull.vertices]
            img = Image.new('L', image.shape[:2], 0)
            ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
            mounth_mask = np.array(img)
            mounth_mask = np.expand_dims(mounth_mask,axis=0).T
        else:
            mounth_mask = np.ones((image.shape[0], image.shape[1],1),dtype=np.uint8)
        mask = mask * (1 - left_eye_mask) * (1 - right_eye_mask) * (1 - mounth_mask)
        skin_image = image * mask

        # rmin, rmax, cmin, cmax = bbox2_CPU(skin_image)

        # cropped_skin_im = skin_image
        # if rmin >= 0 and rmax >= 0 and cmin >= 0 and cmax >= 0 and rmax-rmin >= 0 and cmax-cmin >= 0:
        #     cropped_skin_im = skin_image[int(rmin):int(rmax), int(cmin):int(cmax)]

        return skin_image,mask
