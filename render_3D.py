import cv2
import os
import json
import numpy as np
import torch
import torch.nn as nn
# from smplx.my_smpl_model import create_scale, SMPL
from smplx.smpl import SMPLModel, preprocess


# def load_model(num_people):

#     input_gender = 'neutral'
#     dtype = torch.float32
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')


#     model_params = dict(model_path='models/smpl/SMPL_NEUTRAL.pkl',
#                         joint_mapper=None,
#                         create_global_orient=True,
#                         create_body_pose=True,
#                         create_betas=True,
#                         create_left_hand_pose=False,
#                         create_right_hand_pose=False,
#                         create_expression=False,
#                         create_jaw_pose=False,
#                         create_leye_pose=False,
#                         create_reye_pose=False,
#                         create_transl=True, #set transl in multi-view task  --Buzhen Huang 07/31/2019
#                         create_scale=True,
#                         batch_size=1,
#                         dtype=dtype,)
#     models = []
#     # pose_embeddings = []
#     # # load vposer
#     # vposer = None
#     # if kwarg.get('use_vposer'):
#     #     vposer = load_vposer()
#     #     vposer = vposer.to(device=device)
#     #     vposer.eval()
#     # elif kwarg.get('use_motionprior'):
#     #     vposer = load_motionpriorHP()
#     #     vposer = vposer.to(device=device)
#     #     vposer.eval()

#     for idx in range(num_people):
#         #model = smplx.create_scale(gender=input_gender, **model_params)
#         model = create_scale(gender=input_gender, **model_params)
#         model = model.to(device=device)
#         models.append(model)

#         # pose_embedding = None
#         # # batch_size = frames
#         # pose_embedding = torch.zeros([dataset_obj.frames, 32],
#         #                             dtype=dtype, device=device,
#         #                             requires_grad=True)
#         # pose_embeddings.append(pose_embedding)


#     # setting['model'] = models
#     # setting['vposer'] = vposer
#     # setting['pose_embedding'] = pose_embeddings
#     # setting['frames'] = dataset_obj.frames
#     return models


def project_to_img(joints, verts, faces, extri, intri, image_path, img_folder, viz=False, path=None):
    exp = 1
    if len(verts) < 1:
        return
    if True:
        from render import Renderer
        # for v, (cam, img_path) in enumerate(zip(camera, image_path)):
        #     if v > 0 and exp:
        #         break
        # intri = np.eye(3)
        # rot = camera.rotation.detach().cpu().numpy()
        # trans = camera.translation.detach().cpu().numpy()
        # intri[0][0] = camera.focal_length_x.detach().cpu().numpy()
        # intri[1][1] = camera.focal_length_y.detach().cpu().numpy()
        # intri[:2,2] = camera.center.detach().cpu().numpy()
        # rot_mat = cv2.Rodrigues(rot)[0]
        rot = np.array(extri)[:3, :3]
        trans = np.array(extri)[:3, 3]
        rot_mat = cv2.Rodrigues(rot)[0]
        
        img = cv2.imread(os.path.join(img_folder, image_path))
        render = Renderer(resolution=(img.shape[1], img.shape[0]))
        img = render.render_multiperson(verts, faces, rot_mat.copy(), trans.copy(), intri.copy(), img.copy(), viz=False)

        # for i, gt_joint in enumerate(gt_joint_ids):
        #     # if i != 0 and i != 1:
        #     #     continue
        #     color = [(0,0,255),(0,255,255),(255,0,0),(255,255,0),(255,0,255),(148,148,255)]
        #     if gt_joint is not None and True:
        #         for p in gt_joint:
        #             cv2.circle(img, (int(p[0]),int(p[1])), 3, color[i], 10)
        img_out_file = os.path.join(path, image_path)
        if not os.path.exists(os.path.dirname(img_out_file)):
            os.makedirs(os.path.dirname(img_out_file))
        cv2.imwrite(img_out_file, img)
        render.renderer.delete()
        del render

def render(json_file, output_dir, save_meshes=False, save_images=False):
    import trimesh
    
    num_people = 4
    # models = []
    # models = load_model(num_people)
    # faces = models[0].faces
    model_path= './models/smpl/SMPL_NEUTRAL.pkl'
    pre_model = preprocess(model_path)            
    smpl = SMPLModel(pre_model)
    faces = smpl.get_faces()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    for i, image in enumerate(json_data['images']):
        img_file = image['img_file']
        img_id = image['image_id']
        extri = image['extri']
        intri = image['intri']
        
        curr_mesh_fn = os.path.join(output_dir, 'mesh', img_id)
        if not os.path.exists(curr_mesh_fn):
            os.makedirs(curr_mesh_fn)

        model_outputs = []
        meshes = []
        joints = []
        for idx in range(len(image['annotations'])):
            pose = np.array(image['annotations'][idx]['pose']).reshape(-1, 3)
            betas = np.array(image['annotations'][idx]['betas'])
            trans = np.array(image['annotations'][idx]['trans'])
            scale = np.array(image['annotations'][idx]['scale'])
            
            verts = smpl.set_params(beta=betas, pose=pose, trans=trans)

            # output_models = models[idx].forward(betas= betas, body_pose= pose[:, 3:], global_orient= pose[:, :3], transl= trans, scale= scale, return_vert)
            person_id = image['annotations'][idx]['person_id']
            mesh_fn = os.path.join(curr_mesh_fn, '%s_%s.obj' %(img_id, person_id))

            # # loadmodel->smpl初始姿态->加入pose->变形
            # model_output = models[idx](return_verts=True, betas= betas, body_pose= pose[:, 3:], global_orient= pose[:, :3], transl= trans, scale= scale)
            # model_outputs.append(model_output)
            # mesh = model_outputs[idx].vertices[i].detach().cpu().numpy()
            joints = smpl.get_joints()
            out_mesh = trimesh.Trimesh(verts, faces, process=False)
            out_mesh.export(mesh_fn)

            if save_images:
                meshes.append(verts)
                # meshes.append(model_outputs[idx].vertices[i])
                # joints.append(model_outputs[idx].joints[i])
        
        # save image
        if save_images:
            # img_p = [img_path[v][i] for v in range(len(data['img_path']))]
            # keyp_p = [keypoints[v][i] for v in range(len(data['img_path']))]
            img_p = os.path.join(os.path.dirname(json_file), 'images')
            img_ouput = os.path.join(output_dir, 'proj-images')
            project_to_img(joints, meshes, faces, extri, intri, img_file, img_p, viz=True, path=img_ouput)

def main():
    json_file = r'./3DMPB/3DMPB.json'   # norm-scale-t-trans original norm-scale-t
    output_dir = r'./output'
    render(json_file, output_dir, save_meshes=True, save_images=True)

if __name__ == "__main__":
    main()