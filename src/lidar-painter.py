# -*- coding: utf-8 -*-

__version__ = "0.0.1"
__author__ = "Daniel Adi Nugroho"
__email__ = "dnugroho@gmail.com"
__status__ = "Alpha"
__date__ = "2023-10-25"
__copyright__ = "Copyright (c) 2023 Daniel Adi Nugroho"
__license__ = "GNU General Public License v3.0 (GPL-3.0)"

# Version History
# --------------

# 0.0.1 (2023-10-25)
# - Reorganized the code into a class structure
# - Clear separation between intrinsic and extrinsic parameters
# - Added tweak inputs for fine-tuning or boresighting
# - Added timing metrics for future optimizations
# - Found that the camera position and orientation are not correctly applied
# - dx, dy, dz correction is not absolute, but relative to the camera orientation
# - 180 degrees in omega is added for nadir-looking cameras to account for the mirroring effect
# - Perhaps this inconsistency can be rectified by adding larger principal point offsets?

"""
LiDAR Painter
==============

This script colorizes a LiDAR point cloud using an RGB image taken from a camera.
The colorization process involves projecting 3D points onto the 2D image plane.

Purpose:
--------
- Colorize LiDAR point clouds using RGB images
- Improve visualization and interpretation of LiDAR data

Requirements:
------------
- Python 3.8 or higher
- Required packages:
  - laspy
  - numpy
  - opencv-python

Input Formats:
-------------

Usage:
------
Run the script to launch the GUI:


Output:
-------
- A new LAS file with colorized points

Notes:
------
- This script assumes that the camera and LiDAR are calibrated and synchronized.
- The camera intrinsic parameters are required for the colorization process.
- The camera extrinsic parameters are needed to determine the camera position and orientation.

Acknowledgements:
----------------
- This script was written with the help of DeepSeek R1.


GNU GENERAL PUBLIC LICENSE
--------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import laspy
import cv2
import numpy as np
import time

class PointCloudColorizer:
    def __init__(self, focal_length, cx, cy, k1, k2, p1, p2, k3):
        """
        Initialize the PointCloudColorizer with camera intrinsic parameters.
        These parameters rarely change (e.g., when there are sensor changes).
        """
        self.focal_length = focal_length
        self.cx = cx
        self.cy = cy
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Radial: k1, k2, k3; Tangential: p1, p2

    def set_tweak_inputs(self, dx=0, dy=0, dz=0, d_omega=0, d_phi=0, d_kappa=0):
        """
        Set the tweak inputs for fine-tuning or boresighting purposes.
        These are adjustments to the camera position and orientation.
        """
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.d_omega = d_omega
        self.d_phi = d_phi
        self.d_kappa = d_kappa

    def set_main_inputs(self, point_cloud_path, image_path, camera_position, camera_orientation):
        """
        Set the main inputs that change frequently during typical operations.
        Apply the tweak inputs to the camera position and orientation before using them.
        """
        self.point_cloud_path = point_cloud_path
        self.image_path = image_path

        # Apply tweak inputs to camera position and orientation
        self.camera_position = camera_position + np.array([self.dx, self.dy, self.dz])
        self.camera_orientation = camera_orientation + np.array([self.d_omega, self.d_phi, self.d_kappa])

    def load_point_cloud(self):
        """
        Load the point cloud from the specified path.
        """
        las = laspy.read(self.point_cloud_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        return points

    def load_image(self):
        """
        Load the image from the specified path.
        """
        return cv2.imread(self.image_path)

    def calculate_principal_point(self, image):
        """
        Calculate the principal point based on the image dimensions and offsets (cx, cy).
        """
        return ((image.shape[1] / 2) - self.cx, (image.shape[0] / 2) - self.cy)

    def create_intrinsic_matrix(self, principal_point):
        """
        Create the intrinsic matrix (K) using the focal length and principal point.
        """
        return np.array([[self.focal_length, 0, principal_point[0]],
                         [0, self.focal_length, principal_point[1]],
                         [0, 0, 1]])

    def undistort_image(self, image, K):
        """
        Undistort the image using the camera's distortion coefficients.
        """
        return cv2.undistort(image, K, self.dist_coeffs)

    def rotation_matrix(self, omega, phi, kappa):
        """
        Create a rotation matrix from the given Euler angles (omega, phi, kappa).
        """
        omega, phi, kappa = np.radians(omega), np.radians(phi), np.radians(kappa)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(omega), -np.sin(omega)],
                       [0, np.sin(omega), np.cos(omega)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])
        Rz = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                       [np.sin(kappa), np.cos(kappa), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def create_extrinsic_matrix(self, R):
        """
        Create the extrinsic matrix (T) using the rotation matrix and adjusted camera position.
        """
        return np.hstack((R, -R @ self.camera_position.reshape(3, 1)))

    def project_points(self, points, P):
        """
        Project 3D points onto the 2D image plane using the projection matrix (P).
        """
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
        points_2d_hom = P @ points_hom.T
        points_2d = points_2d_hom[:2] / points_2d_hom[2]
        return points_2d.T

    def filter_points(self, points_2d, points, image_shape):
        """
        Filter points to ensure they are within the image bounds.
        """
        valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1]) & \
                       (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0])
        return points_2d[valid_points], points[valid_points]

    def calculate_depth(self, points, R):
        """
        Calculate the depth (Z coordinate in camera space) for each point.
        """
        points_cam = (R @ (points - self.camera_position).T).T
        return points_cam[:, 2]

    def sort_points_by_depth(self, points_2d, points, depth):
        """
        Sort points by depth (from closest to farthest).
        """
        sorted_indices = np.argsort(depth)
        return points_2d[sorted_indices], points[sorted_indices], depth[sorted_indices]

    def update_depth_buffer_and_colors(self, points_2d, depth, undistorted_image):
        """
        Update the depth buffer and assign colors to the points.
        """
        depth_buffer = np.full((undistorted_image.shape[0], undistorted_image.shape[1]), np.inf)
        colors = np.zeros((points_2d.shape[0], 3), dtype=np.uint8)  # Corrected shape: (N, 3)
        for i, (x, y) in enumerate(points_2d.astype(int)):
            if depth[i] < depth_buffer[y, x]:
                depth_buffer[y, x] = depth[i]
                colors[i] = undistorted_image[y, x]  # Assign RGB values
        return colors

    def save_colorized_point_cloud(self, points, colors, output_path):
        """
        Save the colorized point cloud to a new LAS file.
        """
        header = laspy.LasHeader(point_format=2, version="1.2")
        out_las = laspy.LasData(header)
        out_las.x = points[:, 0]
        out_las.y = points[:, 1]
        out_las.z = points[:, 2]
        out_las.red = colors[:, 2]
        out_las.green = colors[:, 1]
        out_las.blue = colors[:, 0]
        out_las.write(output_path)

    def run(self, output_path):
        """
        Run the entire process to colorize the point cloud.
        """
        start_time = time.time()
        first_start_time = time.time()

        # Load inputs
        points = self.load_point_cloud()
        print(f"Time to load point cloud: {time.time() - start_time:.2f} seconds")

        start_time = time.time()

        image = self.load_image()
        # Try flipping the image vertically
        #image = cv2.flip(image, 0)
        #print("Image flipped vertically")

        start_time = time.time()
        # Calculate intrinsic matrix and undistort image
        principal_point = self.calculate_principal_point(image)
        K = self.create_intrinsic_matrix(principal_point)
        undistorted_image = self.undistort_image(image, K)
        print(f"Time to undistort image: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # Create rotation matrix and extrinsic matrix
        R = self.rotation_matrix(*self.camera_orientation)
        T = self.create_extrinsic_matrix(R)
        P = K @ T

        # Project points and filter
        points_2d = self.project_points(points, P)
        print(f"Time to project points: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        points_2d, points = self.filter_points(points_2d, points, undistorted_image.shape)

        # Calculate depth and sort points
        depth = self.calculate_depth(points, R)
        points_2d, points, depth = self.sort_points_by_depth(points_2d, points, depth)
        print(f"Time to sort points: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # Update depth buffer and assign colors
        colors = self.update_depth_buffer_and_colors(points_2d, depth, undistorted_image)
        print(f"Time to update depth buffer and colors: {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        # Save the colorized point cloud
        self.save_colorized_point_cloud(points, colors, output_path)
        print(f"Time to save LAS file: {time.time() - start_time:.2f} seconds")

        print(f"Total execution time: {time.time() - first_start_time:.2f} seconds")


# Example usage
if __name__ == "__main__":
    # Camera intrinsic parameters (rarely change)
    #focal_length = 3710.76702
    focal_length = 2978.46482
    
    #cx = 0.0
    #cy = 0.0
    #cx = -6.66303
    #cy = 18.5645
    #cx = -6.66303 # these values are perfect
    #cy = -18.5645 # these values are perfect
    #k1 = -0.10661
    #k2 = 0.0011906
    #p1 = 0.000374906
    #2 = -0.000107769
    #k3 = -0.0184579


    cx = 0.0 # these values are perfect
    cy = 0.0 # these values are perfect
    k1 = -0.127207
    k2 = 0.0713991
    p1 = -0.0024243
    p2 = 7.65418e-05
    k3 = 0.0


    # Initialize the colorizer with intrinsic parameters
    colorizer = PointCloudColorizer(focal_length, cx, cy, k1, k2, p1, p2, k3)

    # Set tweak inputs (fine-tuning or boresighting)
    #dx = -0.6
    #dy = -1.6
    #dz = -0.7
    dx = 0.0
    dy = 0.0
    dz = 0.0
    d_omega = 20.5
    d_phi = 0.0
    d_kappa = 0.0

    colorizer.set_tweak_inputs(dx, dy, dz, d_omega, d_phi, d_kappa)

    # Set main inputs (change frequently)
    #point_cloud_path = '..//SAMPLES//warehouse//E5837_SAMPLE_MGA2020Z50_NOCOLOR.laz'
    point_cloud_path = '..//SAMPLES//building//building_pcloud_local_nocolor.laz'

    #image_path = 'DJI_20240823101350_0028_V.JPG'
    #image_path = '..//SAMPLES//warehouse//DJI_20240823101336_0017_V.JPG'
    #image_path = '..//SAMPLES//warehouse//DJI_20240823101401_0037_V.JPG'
    #image_path = '..//SAMPLES//warehouse//DJI_20240823101453_0073_V.JPG'
    image_path = '..//SAMPLES//building//IMG_4310.JPG'

    # Camera extrinsic parameters
    #camera_position = np.array([325948.3123836849117652, 6257893.2384495604783297, 133.7553913284237410])
    #camera_orientation = np.array([-0.9424025485542242, -0.5615344595192379, 78.8992250524786414])
    #camera_position = np.array([326073.2229147708858363, 6257869.1324370214715600, 133.6375558457994543])
    #camera_orientation = np.array([0.0115005606266411, -0.9799490897375966, 152.4919562799930759])
    #camera_position = np.array([325973.2287490139133297, 6257861.2956544291228056, 133.6659689489411846])
    #camera_orientation = np.array([0.2434398815702497, -0.4687796779942965, -101.3514141233089276])
    #camera_position = np.array([326039.9911073350231163, 6257798.1666464824229479, 133.4982608371374511])
    #camera_orientation = np.array([39.0045855240841632, 23.3652423038848873, 25.7762691870272995])
    camera_position = np.array([11.3217892717263506, 10.4768326740610629, -8.1597962947665721])
    camera_orientation = np.array([-99.9137241868528037, 39.1488434212446066, -173.9688624451934231])


    # set main inputs
    colorizer.set_main_inputs(point_cloud_path, image_path, camera_position, camera_orientation)

    # Run the colorization process
    output_path = '..//SAMPLES//building//colorized_point_cloud_xxx.laz'
    colorizer.run(output_path)