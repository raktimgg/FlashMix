import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import copy
import cv2
import tqdm
import fpsample

def load_velodyne_binary(velodyne_bin_path):
    """Decode a binary Velodyne example (of the form '<timestamp>.bin')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
    Returns:
        ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
    Notes:
        - The pre computed points are *NOT* motion compensated.
        - Converting a raw velodyne scan to pointcloud can be done using the
            `velodyne_ranges_intensities_angles_to_pointcloud` function.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    ptcld = data.reshape((4, -1))
    return ptcld

def farthest_point_sampling(points, num_samples):
    if num_samples > len(points):
        raise ValueError("num_samples must be less than or equal to the number of points")

    # Initialize an array to store the indices of the sampled points
    sampled_indices = np.zeros(num_samples, dtype=int)

    # Initialize a list to keep track of the distances to the nearest sampled point
    distances = np.full(len(points), np.inf)

    # Randomly select the first point
    first_index = np.random.randint(len(points))
    sampled_indices[0] = first_index

    # Update the distances from the first point
    current_distances = np.linalg.norm(points - points[first_index], axis=1)
    distances = np.minimum(distances, current_distances)

    # Iteratively select the rest of the points
    for i in range(1, num_samples):
        # Select the point with the maximum distance to the nearest sampled point
        next_index = np.argmax(distances)
        sampled_indices[i] = next_index

        # Update the distances from the newly selected point
        current_distances = np.linalg.norm(points - points[next_index], axis=1)
        distances = np.minimum(distances, current_distances)

    # Return the sampled points using the selected indices
    return points[sampled_indices]

H, W = 32, 720
phi_u = 30.67/180*np.pi
phi_d = 10.67/180*np.pi

pcl_folder = '/data/raktim/Datasets/vReLoc'
save_folder = '/home/raktim/vReLoc'
sequences = os.listdir(pcl_folder)
sequences = [seq for seq in sequences if seq[:3]=='seq']
remaining_seq = ['seq-03', 'seq-12', 'seq-15', 'seq-16', 'seq-05', 'seq-06', 'seq-07', 'seq-14']
for seq in sequences:
    if seq not in remaining_seq:
        continue
    print(seq)
    seq_dir = os.path.join(pcl_folder,seq)
    pcl_files = os.listdir(seq_dir)
    for pcl_file in tqdm.tqdm(pcl_files,total=len(pcl_files)):
        pcl_file_loc = os.path.join(seq_dir, pcl_file)
        if pcl_file_loc.endswith('.bin'):
            # load point cloud
            xyzd = load_velodyne_binary(pcl_file_loc)
            xyzd = xyzd.transpose()
            xyzd_copy = copy.deepcopy(xyzd)

            # convert to spherical coordinates
            xyzd = xyzd[np.where(np.sqrt(xyzd[:,0]**2+xyzd[:,1]**2+xyzd[:,2]**2)<80)]
            xyzd = xyzd[np.where(xyzd[:,3]<70)]
            xyz = xyzd[:,:3]
            intens = xyzd[:,3]
            px, py, pz = xyz[:,0], xyz[:,1], xyz[:,2]
            pd = np.sqrt(px**2 + py**2 + pz**2)
            e = np.abs(phi_u) + np.abs(phi_d)
            u = np.floor( ((np.arcsin(pz/pd) + phi_d)/e)*(H-1) ).astype(np.int32)+1
            v = np.floor((0.5*(np.arctan2(py, px)/np.pi))*(W-1)).astype(np.int32) + 180
            img_new1 = np.zeros([H,W])
            img_new2 = np.zeros([H,W])
            pd1 = copy.deepcopy(pd)
            img_new1[u,v] = intens
            img_new2[u,v] = pd
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
            image_save_filename = os.path.join(image_save_dir, pcl_file[:-3]+'png')
            img_pil.save(image_save_filename)

            xyz = xyzd[:,:3]
            # xyz = farthest_point_sampling(xyz,4096)
            xyz_idx = fpsample.bucket_fps_kdtree_sampling(xyz,4096)
            xyz = xyz[xyz_idx]
            lidar_save_dir = os.path.join(save_folder, seq, 'lidar_npy')
            os.makedirs(lidar_save_dir, exist_ok=True)
            lidar_save_filename = os.path.join(lidar_save_dir, pcl_file[:-3]+'npy')
            np.save(lidar_save_filename, xyz)
        if pcl_file_loc.endswith('.txt'):
            pose = np.loadtxt(pcl_file_loc, delimiter=',')
            pose_save_dir = os.path.join(save_folder, seq, 'poses')
            os.makedirs(pose_save_dir, exist_ok=True)
            pose_save_filename = os.path.join(pose_save_dir, pcl_file[:-3]+'txt')
            np.savetxt(pose_save_filename, pose)

