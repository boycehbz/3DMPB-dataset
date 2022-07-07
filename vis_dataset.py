import os
import json
import pickle as pkl
import numpy as np
import cv2
import math
import sys
from tqdm import tqdm
# from cmd_parse import parse_config
from ochumanApi.ochuman import OCHuman

def draw_bbox(img, bbox, thickness=3, color=(255, 0, 0)):
    canvas = img.copy()
    cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return canvas

def draw_skeleton(img, kpt, connection=None, colors=None, bbox=None):
    # kpt = np.array(kpt, dtype=np.int32).reshape(-1, 3)
    kpt = np.array(kpt)
    npart = kpt.shape[0]
    canvas = img.copy()

    # if npart==17: # coco
    #     part_names = ['nose', 
    #                   'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    #                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    #                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
    #                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle'] 
    #     visible_map = {2: 'vis', 
    #                    1: 'not_vis', 
    #                    0: 'missing'}
    #     map_visible = {value: key for key, value in visible_map.items()}
    #     if connection is None:
    #         connection = [[16, 14], [14, 12], [17, 15], 
    #                       [15, 13], [12, 13], [6, 12], 
    #                       [7, 13], [6, 7], [6, 8], 
    #                       [7, 9], [8, 10], [9, 11], 
    #                       [2, 3], [1, 2], [1, 3], 
    #                       [2, 4], [3, 5], [4, 6], [5, 7]]
    #     idxs_draw = np.where(kpt[:, :, 2] != map_visible['missing'])[0]
    # elif npart==19: # ochuman
    #     part_names = ["right_shoulder", "right_elbow", "right_wrist",
    #                  "left_shoulder", "left_elbow", "left_wrist",
    #                  "right_hip", "right_knee", "right_ankle",
    #                  "left_hip", "left_knee", "left_ankle",
    #                  "head", "neck"] + \
    #                  ['right_ear', 'left_ear', 'nose', 'right_eye', 'left_eye']
    #     visible_map = {0: 'missing', 
    #                    1: 'vis', 
    #                    2: 'self_occluded', 
    #                    3: 'others_occluded'}
    #     map_visible = {value: key for key, value in visible_map.items()}
    #     if connection is None:
    #         connection = [[16, 19], [13, 17], [4, 5],
    #                      [19, 17], [17, 14], [5, 6],
    #                      [17, 18], [14, 4], [1, 2],
    #                      [18, 15], [14, 1], [2, 3],
    #                      [4, 10], [1, 7], [10, 7],
    #                      [10, 11], [7, 8], [11, 12], [8, 9],
    #                      [16, 4], [15, 1]] # TODO
    #     idxs_draw = np.where(kpt[:, :, 2] != map_visible['missing'])[0]
        
    if npart==24: # smpl(24 joints + transl)
        part_names = ['Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
                      'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
                      'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
                      'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
                      'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
                      'Left_Finger', 'Right_Finger']
        visible_map = {0: 'missing', 
                       1: 'vis'}
        map_visible = {value: key for key, value in visible_map.items()}
        if connection is None:
            connection = [[15, 12], [12, 9], [9, 13], [9, 14], 
                        [16, 18], [18, 20], [20, 22], [13, 16],
                        [17, 19], [19, 21], [21, 23], [14, 17],
                        [9, 6], [6, 3], [3, 0], [0, 1], [0, 2],
                        [2, 5], [5, 8], [8, 11],
                        [1, 4], [4, 7], [7, 10]]                       
        idxs_draw = [i for i in range(kpt.shape[0])]
                
    if colors is None:
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], 
                 [255, 255, 0], [170, 255, 0], [85, 255, 0], 
                 [0, 255, 0], [0, 255, 85], [0, 255, 170], 
                 [0, 255, 255], [0, 170, 255], [0, 85, 255], 
                 [0, 0, 255], [85, 0, 255], [170, 0, 255],
                 [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    elif type(colors[0]) not in [list, tuple]:
        colors = [colors]
    
    # idxs_draw = np.where(vis != map_visible['missing'])[0]
    if len(idxs_draw)==0:
        return img
    
    if bbox is None:
        bbox = [np.min(kpt[idxs_draw, 0]), np.min(kpt[idxs_draw, 1]),
                np.max(kpt[idxs_draw, 0]), np.max(kpt[idxs_draw, 1])] # xyxy
    
    Rfactor = math.sqrt((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) / math.sqrt(img.shape[0] * img.shape[1])
    Rpoint = int(min(10, max(Rfactor*10, 4)))
    Rline = int(min(10, max(Rfactor*5, 2)))
    #print (Rfactor, Rpoint, Rline)
    
    for idx in idxs_draw:
        if kpt.shape[1] == 2:
            x, y = kpt[idx, :]
            cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
        else:
            x, y, v = kpt[idx, :]
            cv2.circle(canvas, (x, y), Rpoint, colors[idx%len(colors)], thickness=-1)
            
            if v==2:
                cv2.rectangle(canvas, (x-Rpoint-1, y-Rpoint-1), (x+Rpoint+1, y+Rpoint+1), 
                            colors[idx%len(colors)], 1)
            elif v==3:
                cv2.circle(canvas, (x, y), Rpoint+2, colors[idx%len(colors)], thickness=1)

    for idx in range(len(connection)):
        idx1, idx2 = connection[idx]
        if kpt.shape[1] == 2:
            y1, x1 = kpt[idx1]
            y2, x2 = kpt[idx2]
            v1 = 1
            v2 = 1
        else:
            y1, x1, v1 = kpt[idx1-1]
            y2, x2, v2 = kpt[idx2-1]
        if v1 == map_visible['missing'] or v2 == map_visible['missing']:
            continue
        mX = (x1+x2)/2.0
        mY = (y1+y2)/2.0
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), Rline), int(angle), 0, 360, 1)
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, colors[idx%len(colors)])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
    return canvas

    
def vis_kpt_2d(json_file, ImgDir, output_dir):

    # ochuman = OCHuman(AnnoFile=json_file, Filter= None)
    # image_ids = ochuman.getImgIds()
    # print ('Total images: %d'%len(image_ids))

    with open(json_file) as f:
        json_data = json.load(f)

    for img_id in tqdm(range(len(json_data['images']))):
        # data = ochuman.loadImgs(imgIds=[img_id])[0]
        data = json_data['images'][img_id]
        img = cv2.imread(os.path.join(ImgDir, data['img_file']))
        height, width = data['height'], data['width']

        colors = [[255, 0, 0], 
                [255, 255, 0],
                [0, 255, 0],
                [0, 255, 255], 
                [0, 0, 255], 
                [255, 0, 255]]

        for i in range(len(data['annotations'])):
            bbox_tmp = data['annotations'][i]['bbox']
            bbox = [i for j in bbox_tmp for i in j]  # 2*2->4*1
            img = draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
            # vis = data['annotations'][i]['vis']  #14
            kpt = data['annotations'][i]['smpl_joints_2d']
            if kpt is not None:
                img = draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)

        # cv2.imshow('img', img)
        # cv2.waitKey()
        img_file = os.path.join(output_dir, data['img_file'])
        os.makedirs(os.path.dirname(img_file), exist_ok=True)
        cv2.imwrite(img_file, img)
    print('Finish!')


def main(**args):
    json_file = args.pop('json_file')
    ImgDir = args.pop('ImgDir')
    output_dir = args.pop('output_dir')
    vis_kpt_2d(json_file, ImgDir, output_dir)

    

if __name__ == "__main__":
    import argparse
    sys.argv = ["", "--json_file", "3DMPB-dataset/3DMPB/3DMPB.json", "--ImgDir", "3DMPB-dataset/3DMPB/images", "--output_dir", "output"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, help='directory of dataset json file')
    parser.add_argument('--ImgDir', type=str, help='directory of dataset images')
    parser.add_argument('--output_dir', type=str, help='directory of output images')
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
    

    