#coding:utf-8
import cv2
import numpy as np
from copy import deepcopy

def surface_projection(vertices, faces, joint, extri, intri, image, viz=False):
    """
    @ vertices: N*3, mesh vertex
    @ faces: N*3, mesh face
    @ joint: N*3, joints
    @ extri: 4*4, camera extrinsic
    @ intri: 3*3, camera intrinsic
    @ image: RGB image
    @ viz: bool, visualization
    """
    im = deepcopy(image)
    h = im.shape[0]
    # homogeneous
    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))

    # projection
    out_point = np.dot(extri, temp_v)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1,0)
    
    # color
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)
    color = (255, 255, 255)
    
    # draw mesh
    for f in faces:
        point = out_point[f]
        im = cv2.polylines(im, [point], True, color, 1)
    
    # joints projection
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)

    # draw projected joints
    for i in range(len(out_point)):
        im = cv2.circle(im, tuple(out_point[i]), int(h/100), (255,0,0),-1)

    # visualization
    if viz:
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im/255.)
        cv2.waitKey()

    return out_point, im

def dist_joint_projection(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    
    Roughly, x = K*(R*X + t) + distortion
    
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    
    x = np.asarray(R*X + t)
    
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    
    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x

def joint_projection(joint, extri, intri, image, viz=False):

    im = deepcopy(image)

    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)

    if viz:
        for i in range(len(out_point)):
            im = cv2.circle(im, tuple(out_point[i]), 5, (0,0,255),-1)

        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im/255.)
        cv2.waitKey()

    return out_point, im

