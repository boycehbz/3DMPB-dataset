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
from tqdm import tqdm

def save_json(out_path, data):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(data, f)


def estimate_translation_np(S, joints_2d, joints_conf, fx=5000, fy=5000, cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([fx, fy])
    # optical center
# center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def calc_aabb(ptSets):
    lt = np.array([ptSets[0][0], ptSets[0][1]])
    rb = lt.copy()
    for pt in ptSets:
        if pt[0] == 0 and pt[1] == 0:
            continue
        lt[0] = min(lt[0], pt[0])
        lt[1] = min(lt[1], pt[1])
        rb[0] = max(rb[0], pt[0])
        rb[1] = max(rb[1], pt[1])
    return lt, rb

def vis_smpl_3d(dataset_dir='./3DMPB', output='./output', **kwargs):
    json_file = os.path.join(dataset_dir, '3DMPB.json')
    with open(json_file) as f:
        annotations = json.load(f)

    smpl_path = 'data/SMPL_NEUTRAL.pkl'
    smpl_layer = smplx.create(smpl_path, 'smpl')

    smpl = SMPLModel(device=torch.device('cpu'), model_path=smpl_path)

    dataset = []

    for ant in tqdm(annotations['images'], total=len(annotations['images'])):
        data = {}

        data['img_file'] = 'images/' + ant['img_file']
        data['width'] = ant['width']
        data['height'] = ant['height']
        data['intri'] = ant['intri']
        extri = np.array(ant['extri'], dtype=np.float32)
        intri = np.array(ant['intri'], dtype=np.float32)

        frame_annot = []
        for pid in ant['annotations']:
            person_data = {}
            person_data['mask_file'] = 'masks/' + pid['mask_file']
            person_data['vis'] = pid['vis']


            origin_shape = np.array(pid['betas'], dtype=np.float32)
            origin_pose = np.array(pid['pose'], dtype=np.float32)
            origin_trans = np.array(pid['trans'], dtype=np.float32)
            origin_scale = pid['scale'][0]

            origin_shape = torch.from_numpy(origin_shape).reshape(-1, 10)
            origin_pose = torch.from_numpy(origin_pose).reshape(-1, 72)
            origin_trans = torch.from_numpy(origin_trans).reshape(-1, 3)

            origin_verts, origin_joints = smpl(origin_shape, origin_pose, origin_trans, origin_scale)
            origin_joints = origin_joints.detach().numpy()[0]
            origin_verts = origin_verts.detach().numpy()[0]

            img_path = os.path.join(dataset_dir, 'images', ant['img_file'])
            img = cv2.imread(img_path)

            proj_verts, _ = joint_projection(origin_verts, ant['extri'], ant['intri'], img, viz=False)
            lsp_joints_2d, _ = joint_projection(origin_joints, ant['extri'], ant['intri'], img, viz=False)

            lsp_joints_2d = np.insert(lsp_joints_2d, 2, 1, axis=1)

            # global_orient = pose[0,None,:]
            # body_pose = pose[1:,:]

            pose = np.array(pid['pose'], dtype=np.float32)
            global_rot = cv2.Rodrigues(pose[:3])[0]
            r = extri[:3,:3]
            global_rot = np.dot(r, global_rot)
            global_rot = cv2.Rodrigues(global_rot)[0].reshape(-1,)
            pose[:3] = global_rot

            shape = origin_shape
            pose = torch.from_numpy(pose).reshape(-1, 72)
            temp_trans = torch.zeros((1,3), dtype=torch.float32)

            temp_verts, temp_joints = smpl(shape, pose, temp_trans, 1)
            temp_joints = temp_joints.detach().numpy()[0]

            t = estimate_translation_np(temp_joints, lsp_joints_2d[:,:2], lsp_joints_2d[:,2], intri[0][0], intri[1][1], intri[0][2], intri[1][2])

            trans = torch.from_numpy(t).reshape(-1, 3).to(torch.float32)

            _, lsp_joints_3d = smpl(shape, pose, trans, 1)
            lsp_joints_3d = lsp_joints_3d.detach().numpy()[0]

            # joint_projection(lsp_joints_3d, np.eye(4), intri, img, viz=True)

            lt, rb = calc_aabb(proj_verts)
            bbox = np.array([lt, rb]).reshape(2,2)
            bbox = bbox.tolist()

            person_data['bbox'] = bbox
            person_data['lsp_joints_2d'] = lsp_joints_2d.tolist()
            person_data['lsp_joints_3d'] = lsp_joints_3d.tolist()
            person_data['betas'] = shape.detach().cpu().numpy().reshape(-1,).tolist()
            person_data['pose'] = pose.detach().cpu().numpy().reshape(-1,).tolist()
            person_data['trans'] = trans.detach().cpu().numpy().reshape(-1,).tolist()
            frame_annot.append(person_data)

        data['annotations'] = frame_annot
        dataset.append(data)
            # trans = trans / scale
            
            # extri[:3,3] = extri[:3,3] / scale
            # scale = scale / scale

    save_json(R'D:\BuzhenHuang_OpenSource\3DMPB-dataset\3DMPB\annot_scale1.json', dataset)



        # # get mesh and joint coordinates
        # with torch.no_grad():
        #     output = smpl_layer(betas=shape, body_pose=body_pose.view(1,-1), global_orient=global_orient, transl=trans)
        # mesh_cam = output.vertices[0].numpy()
        # joint_cam = output.joints[0].numpy()



        # surface_projection(verts, smpl.faces, joints, extri, ant['intri'], img, viz=True)
        # # joint_projection(joints, ant['extri'], ant['intri'], img, viz=True)

        # pass

def main(**kwargs):
    if kwargs.get('vis_smpl'):
        vis_smpl_3d(**kwargs)
    else:
        json_file = args.pop('json_file')
        ImgDir = args.pop('ImgDir')
        output_dir = args.pop('output_dir')
        vis_kpt_2d(json_file, ImgDir, output_dir)

    

if __name__ == "__main__":
    import argparse
    sys.argv = ["", "--dataset_dir", "3DMPB", "--output_dir", "output", "--vis_smpl", 'True']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='directory of dataset json file')
    parser.add_argument('--vis_smpl', type=bool, help='directory of dataset images')
    parser.add_argument('--output_dir', default='./output', type=str, help='directory of output images')
    args = parser.parse_args()
    args_dict = vars(args)

    main(**args_dict)
    

    