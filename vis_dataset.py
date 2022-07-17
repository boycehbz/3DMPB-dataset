'''
 @FileName    : vis_dataset.py
 @EditTime    : 2022-07-10 16:40:05
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import json
import torch
import cv2
import sys
import numpy as np
from utils.module_utils import *
from utils.smpl_torch_batch import SMPLModel
from utils.render import Renderer
    
def vis_kpt_2d(dataset_dir='./3DMPB', output_dir='./output', **kwargs):
    json_file = os.path.join(dataset_dir, 'annot.json')
    with open(json_file) as f:
        annotations = json.load(f)

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

        vis_img('img', img)
        output_path = os.path.join(output_dir, 'mask', data['img_file'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
    print('Finish!')


def vis_smpl_3d(dataset_dir='./3DMPB', output_dir='./output', **kwargs):
    json_file = os.path.join(dataset_dir, 'annot.json')
    with open(json_file) as f:
        annotations = json.load(f)

    smpl = SMPLModel(device=torch.device('cpu'), model_path='data/SMPL_NEUTRAL.pkl')
    render = Renderer(resolution=(annotations[0]['width'], annotations[0]['height']))

    for i, annot in enumerate(annotations):
        img = cv2.imread(os.path.join(dataset_dir, annot['img_file']))

        intri = np.array(annot['intri'], dtype=np.float32)

        pose, shape, trans = [], [], []
        for person in annot['annotations']:
            pose.append(person['pose'])
            shape.append(person['betas'])
            trans.append(person['trans'])

        pose = torch.from_numpy(np.array(pose, dtype=np.float32))
        shape = torch.from_numpy(np.array(shape, dtype=np.float32))
        trans = torch.from_numpy(np.array(trans, dtype=np.float32))

        verts, joints = smpl(shape, pose, trans)

        img = render.render_multiperson(verts.detach().cpu().numpy(), smpl.faces, np.eye(3), np.zeros((3,)), intri.copy(), img.copy(), viz=False)

        vis_img('img', img)
        output_path = os.path.join(output_dir, 'smpl', annot['img_file'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)


def main(vis_smpl=False, **kwargs):
    if vis_smpl:
        vis_smpl_3d(**kwargs)
    else:
        vis_kpt_2d(**kwargs)
    

if __name__ == "__main__":
    import argparse
    # sys.argv = ['', '--dataset_dir=3DMPB', '--output_dir=output']
    # sys.argv = ['', '--dataset_dir=3DMPB', '--output_dir=output', '--vis_smpl=True']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='directory of dataset')
    parser.add_argument('--vis_smpl', default=False, type=bool, help='')
    parser.add_argument('--output_dir', default='./output', type=str, help='directory of output images')
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
    

    