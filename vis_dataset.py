'''
 @FileName    : vis_dataset.py
 @EditTime    : 2022-07-10 16:40:05
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import json
from matplotlib.pyplot import sca
import numpy as np
import torch
import cv2
import math
import sys
from tqdm import tqdm
from projection import joint_projection, surface_projection
import smplx
from smpl_torch_batch import SMPLModel
from render import Renderer

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
        
    if npart==14: # smpl(24 joints + transl)
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
            # connection = [[15, 12], [12, 9], [9, 13], [9, 14], 
            #             [16, 18], [18, 20], [20, 22], [13, 16],
            #             [17, 19], [19, 21], [21, 23], [14, 17],
            #             [9, 6], [6, 3], [3, 0], [0, 1], [0, 2],
            #             [2, 5], [5, 8], [8, 11],
            #             [1, 4], [4, 7], [7, 10]] 
            connection = [[13, 12], [12, 8], [8, 7], [7, 6], 
                        [12, 9], [9, 10], [10, 11], 
                        [12, 2], [2, 1], [1, 0], 
                        [12, 3], [3, 4], [4, 5]]                      
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

def draw_mask(img, mask, thickness=3, color=(255, 0, 0)):
    def _get_edge(mask, thickness=3):
        dtype = mask.dtype
        x=cv2.Sobel(np.float32(mask),cv2.CV_16S,1,0, ksize=thickness) 
        y=cv2.Sobel(np.float32(mask),cv2.CV_16S,0,1, ksize=thickness)
        absX=cv2.convertScaleAbs(x)
        absY=cv2.convertScaleAbs(y)  
        edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return edge.astype(dtype)
    
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    img = img.copy()
    canvas = np.zeros(img.shape, img.dtype) + color
    img[mask > 0] = img[mask > 0] * 0.8 + canvas[mask > 0] * 0.2
    edge = _get_edge(mask, thickness)
    img[edge > 0] = img[edge > 0] * 0.2 + canvas[edge > 0] * 0.8
    return img

    
def vis_kpt_2d(dataset_dir='./3DMPB', output='./output', **kwargs):
    json_file = os.path.join(dataset_dir, 'annot.json')
    with open(json_file) as f:
        annotations = json.load(f)
    with open('./3DMPB/3DMPB.json') as test:
        test_data = json.load(test)

    for data in annotations:

        img = cv2.imread(os.path.join(dataset_dir, data['img_file']))
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
            kpt = [j[:2] for j in data['annotations'][i]['lsp_joints_2d']]
            mask_file = data['annotations'][i]['mask_file']
            mask = cv2.imread(os.path.join(dataset_dir, mask_file))
            if kpt is not None:
                img = draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
            if mask is not None:
                img = draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])

        cv2.imshow('img', img)
        cv2.waitKey()
        # img_file = os.path.join(output_dir, 'vis_img',data['img_file'])
        # os.makedirs(os.path.dirname(img_file), exist_ok=True)
        # cv2.imwrite(img_file, img)
    print('Finish!')


def vis_smpl_3d(dataset_dir='./3DMPB', output='./output', **kwargs):
    json_file = os.path.join(dataset_dir, 'annot.json')
    with open(json_file) as f:
        annotations = json.load(f)

    smpl = SMPLModel(device=torch.device('cpu'), model_path='data/SMPL_NEUTRAL.pkl')

    for annot in annotations:
        
        img = cv2.imread(os.path.join(dataset_dir, annot['img_file']))
        render = Renderer(resolution=(img.shape[1], img.shape[0]))

        intri = np.array(annot['intri'], dtype=np.float32)
        extri = np.eye(4, dtype=np.float32)

        pose, shape, trans = [], [], []
        for person in annot['annotations']:
            pose.append(person['pose'])
            shape.append(person['betas'])
            trans.append(person['trans'])

        pose = torch.from_numpy(np.array(pose, dtype=np.float32))
        shape = torch.from_numpy(np.array(shape, dtype=np.float32))
        trans = torch.from_numpy(np.array(trans, dtype=np.float32))

        verts, joints = smpl(shape, pose, trans)

        img = render.render_multiperson(verts.detach().cpu().numpy(), smpl.faces, np.eye(3), np.zeros((3,)), intri.copy(), img.copy(), viz=True)


def main(**kwargs):
    if kwargs.get('vis_smpl'):
        vis_smpl_3d(**kwargs)
    else:
        vis_kpt_2d(**kwargs)
    

if __name__ == "__main__":
    import argparse
    sys.argv = ['', '--dataset_dir=3DMPB', '--output_dir=output']
    # sys.argv = ['', '--dataset_dir=3DMPB', '--output_dir=output', '--vis_smpl=True']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='directory of dataset')
    parser.add_argument('--vis_smpl', default=False, type=bool, help='')
    parser.add_argument('--output_dir', default='./output', type=str, help='directory of output images')
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
    

    