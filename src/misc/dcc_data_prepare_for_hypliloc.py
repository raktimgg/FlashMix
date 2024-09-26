import copy
import os
import pickle
import sys

import cv2
import fpsample
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image

H, W = 32, 720
phi_u = 30.67/180*np.pi
phi_d = 10.67/180*np.pi

pcl_folder = '/data/raktim/Datasets/Mulran/DCC'
save_folder = '/home/raktim/DCC'

pickle_loc = '/data/raktim/Projects/APR/LPR_CRRL/src/utils/dcc_og.pkl'
with open(pickle_loc, 'rb') as file:
    data_dict_all = pickle.load(file)

data_dict1 = data_dict_all[0]
data_dict2 = data_dict_all[1]
data_dict1.extend(data_dict2)
data_dict = data_dict1

# config
offset = np.tile(np.array([0, 6, 12, 18]), 16)
H = 64
W = 1024

# Create a meshgrid for u and v
u = np.arange(H)
v = np.arange(W)
u_grid, v_grid = np.meshgrid(u, v, indexing='ij')

def scan2image(scan):
    # config
    offset = np.tile(np.array([0, 6, 12, 18]), 16)
    H = 64
    W = 1024

    # # xyz_image = np.zeros((H, W, 3))
    # range_img = np.zeros((H, W))
    # intensity_img = np.zeros((H, W))

    # print(offset.shape)

    # for u in range(H):
    #     for v in range(W):
    #         vv = (v + offset[u]) % W
    #         index = vv * H + u
    #         pt = scan[index, :3]
    #         intensity = scan[index, 3]

    #         range_img[u, v] = np.linalg.norm(pt)
    #         intensity_img[u, v] = intensity
    #         # xyz_image[u, v, :] = pt

    # Initialize images
    # range_img = np.zeros((H, W))
    # intensity_img = np.zeros((H, W))

    # Compute the shifted indices vv
    vv = (v_grid + offset[u_grid]) % W

    # Compute the flattened indices
    index = vv * H + u_grid

    # Extract the relevant points and intensities from the scan
    pts = scan[index, :3]
    intensities = scan[index, 3]

    # print(pts.shape)

    # Compute the range image
    range_img = np.linalg.norm(pts, axis=2)

    # Assign the intensity values to the intensity image
    intensity_img = intensities

    return range_img, intensity_img


# [1567496043059928133, 'DCC3/Ouster/1567496043059928133.bin', array([ 355631.7976, 4026722.127 ]), array([[ 4.85809606e-03, -9.99922468e-01, -1.14654679e-02,
#          3.55631798e+05],
#        [ 9.99945785e-01,  4.96317177e-03, -9.15394713e-03,
#          4.02672213e+06],
#        [ 9.21014249e-03, -1.14203755e-02,  9.99892368e-01,
#          1.93885258e+01],
#        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#          1.00000000e+00]])]

for idx, data_sample in tqdm.tqdm(enumerate(data_dict), total=len(data_dict)):
    rel_path = data_sample[1]
    # print(rel_path)
    if rel_path=='DCC3/Ouster/1567496784952532897.bin':
        continue
    seq = rel_path.split('/')[0]
    name = str(data_sample[0])
    pcl_file_loc = os.path.join(pcl_folder, rel_path)
    # load point cloud
    pc = np.fromfile(pcl_file_loc, dtype=np.float32)
    xyzd = np.reshape(pc, (-1, 4))
    xyzd_copy = copy.deepcopy(xyzd)
    # convert to spherical coordinates
    # xyzd = xyzd[np.where(np.sqrt(xyzd[:,0]**2+xyzd[:,1]**2+xyzd[:,2]**2)<80)]
    # xyzd = xyzd[np.where(np.sqrt(xyzd[:,0]**2+xyzd[:,1]**2+xyzd[:,2]**2)>0.1)]
    # xyzd = xyzd[np.where(xyzd[:,3]<70)]
    xyz = xyzd[:,:3]
    intens = xyzd[:,3]

    img_new2, img_new1 = scan2image(xyzd)

    # px, py, pz = xyz[:,0], xyz[:,1], xyz[:,2]
    # pd = np.sqrt(px**2 + py**2 + pz**2)
    # e = np.abs(phi_u) + np.abs(phi_d)
    # u = np.floor( ((np.arcsin(pz/pd) + phi_d)/e)*(H-1) ).astype(np.int32) + 1
    # v = np.floor((0.5*(np.arctan2(py, px)/np.pi))*(W-1)).astype(np.int32) + 180
    # img_new1 = np.zeros([H,W])
    # img_new2 = np.zeros([H,W])
    # pd1 = copy.deepcopy(pd)
    # # print(u,v)
    # img_new1[u,v] = intens
    # img_new2[u,v] = pd
    img_new = np.vstack([img_new1,img_new2])

    # save image
    data = img_new
    cmap = plt.get_cmap()
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())  # Normalize data for color mapping
    # Create color mapped version of the array
    color_mapped_img = cmap(norm(data))
    # Convert color mapped image to RGB (removing alpha channel)
    rgb_image = (color_mapped_img[:, :, :3] * 255).astype(np.uint8)  # Scale to 0-255 for PIL
    # Convert the RGB array to a Pillow Image
    img_pil = Image.fromarray(rgb_image)
    image_save_dir = os.path.join(save_folder, seq, 'projected_lidar_image')
    os.makedirs(image_save_dir, exist_ok=True)
    image_save_filename = os.path.join(image_save_dir, name+'.png')
    # print(image_save_filename)
    img_pil.save(image_save_filename)

    ### Farthest point sampling
    # xyz = xyzd[:,:3]
    # # xyz = farthest_point_sampling(xyz,4096)
    # if len(xyz)>4096:
    #     xyz_idx = fpsample.bucket_fps_kdtree_sampling(xyz,4096)
    #     xyz = xyz[xyz_idx]
    # else:
    #     num_to_pad = 4096 - xyz.shape[0]
    #     index = np.random.choice(xyz.shape[0], size=num_to_pad, replace=True)
    #     xyz = np.concatenate([xyz,xyz[index]])
    # lidar_save_dir = os.path.join(save_folder, seq, 'lidar_npy')
    # os.makedirs(lidar_save_dir, exist_ok=True)
    # lidar_save_filename = os.path.join(lidar_save_dir, name+'.npy')
    # np.save(lidar_save_filename, xyz)


    pose = data_sample[3]
    # print(data)
    pose_save_dir = os.path.join(save_folder, seq, 'poses')
    os.makedirs(pose_save_dir, exist_ok=True)
    pose_save_filename = os.path.join(pose_save_dir, name+'.txt')
    np.savetxt(pose_save_filename, pose)

