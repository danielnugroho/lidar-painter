import numpy as np
import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
import math
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy.spatial.transform import Rotation as R

# Set default matplotlib style to have black background
plt.style.use('dark_background')

def load_point_cloud(file_path):
    # [Same as before, no changes]
    print(f"Loading point cloud from {file_path}...")
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
    else:
        colors = np.ones((len(points), 3))
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounds = (min_bound[0], min_bound[1], min_bound[2], max_bound[0], max_bound[1], max_bound[2])
    print(f"Point cloud loaded with {len(points)} points")
    print(f"Bounds: {bounds}")
    return points, colors, bounds

def generate_camera_positions(bounds, num_cameras=10, max_distance_factor=1.0):
    # [Same as before, no changes]
    print(f"Generating {num_cameras} camera positions...")
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
    extents = [max_x - min_x, max_y - min_y, max_z - min_z]
    max_extent = max(extents)
    max_distance = max_extent * max_distance_factor
    camera_positions = []
    for i in range(num_cameras):
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        distance = random.uniform(0.5 * max_extent, max_distance)
        x = center[0] + distance * math.sin(phi) * math.cos(theta)
        y = center[1] + distance * math.sin(phi) * math.sin(theta)
        z = center[2] + distance * math.cos(phi)
        position = [x, y, z]
        target = center
        up = [0, 0, 1]
        camera_positions.append((position, target, up))
    print(f"Generated {len(camera_positions)} camera positions")
    return camera_positions

def dji_phantom_4_rtk_camera_params():
    # [Same as before, no changes]
    params = {
        'width': 5472, 'height': 3648, 'sensor_width': 13.2, 'sensor_height': 8.8,
        'focal_length': 8.8, 'pixel_size': 0.00241, 'principal_point_x': 5472 / 2,
        'principal_point_y': 3648 / 2, 'k1': -0.0012, 'k2': 0.0006, 'k3': -0.0001,
        'p1': 0.0001, 'p2': 0.0001, 'fov_x': None, 'fov_y': None
    }
    params['fov_x'] = 2 * math.atan(params['sensor_width'] / (2 * params['focal_length']))
    params['fov_y'] = 2 * math.atan(params['sensor_height'] / (2 * params['focal_length']))
    return params

def calculate_camera_intrinsics(camera_params):
    # [Same as before, no changes]
    fx = camera_params['focal_length'] / camera_params['pixel_size']
    fy = camera_params['focal_length'] / camera_params['pixel_size']
    cx = camera_params['principal_point_x']
    cy = camera_params['principal_point_y']
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic

def calculate_camera_extrinsics(position, target, up):
    # [Same as before, no changes]
    position = np.array(position)
    target = np.array(target)
    up = np.array(up)
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    rotation = np.array([[right[0], right[1], right[2], 0],
                         [up[0], up[1], up[2], 0],
                         [-forward[0], -forward[1], -forward[2], 0],
                         [0, 0, 0, 1]])
    translation = np.array([[1, 0, 0, -position[0]],
                           [0, 1, 0, -position[1]],
                           [0, 0, 1, -position[2]],
                           [0, 0, 0, 1]])
    extrinsic = np.matmul(rotation, translation)
    return extrinsic

def transform_points(points, extrinsic):
    # [Same as before, no changes]
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = np.dot(points_homogeneous, extrinsic.T)
    return points_transformed[:, :3]

def project_points(points, intrinsic):
    # [Same as before, no changes]
    points_normalized = points.copy()
    mask = points[:, 2] != 0
    points_normalized[mask, 0] = points[mask, 0] / points[mask, 2]
    points_normalized[mask, 1] = points[mask, 1] / points[mask, 2]
    points_normalized[mask, 2] = 1.0
    pixels = np.dot(points_normalized, intrinsic.T)
    return pixels[:, :2]

def generate_grid_lines(bounds, position, grid_spacing=10):
    # [Same as before, no changes]
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    camera_x, camera_y, camera_z = position
    center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
    
    min_x = np.floor(min_x / grid_spacing) * grid_spacing
    min_y = np.floor(min_y / grid_spacing) * grid_spacing
    min_z = np.floor(min_z / grid_spacing) * grid_spacing
    max_x = np.ceil(max_x / grid_spacing) * grid_spacing
    max_y = np.ceil(max_y / grid_spacing) * grid_spacing
    max_z = np.ceil(max_z / grid_spacing) * grid_spacing
    
    grid_lines = []
    
    include_xy_min_z = camera_z > center[2]
    include_xy_max_z = camera_z <= center[2]
    include_xz_min_y = camera_y > center[1]
    include_xz_max_y = camera_y <= center[1]
    include_yz_min_x = camera_x > center[0]
    include_yz_max_x = camera_x <= center[0]
    
    if include_xy_min_z:
        z = min_z
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            grid_lines.append(([x, min_y, z], [x, max_y, z]))
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            grid_lines.append(([min_x, y, z], [max_x, y, z]))
    if include_xy_max_z:
        z = max_z
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            grid_lines.append(([x, min_y, z], [x, max_y, z]))
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            grid_lines.append(([min_x, y, z], [max_x, y, z]))
    
    if include_xz_min_y:
        y = min_y
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            grid_lines.append(([x, y, min_z], [x, y, max_z]))
        for z in np.arange(min_z, max_z + grid_spacing, grid_spacing):
            grid_lines.append(([min_x, y, z], [max_x, y, z]))
    if include_xz_max_y:
        y = max_y
        for x in np.arange(min_x, max_x + grid_spacing, grid_spacing):
            grid_lines.append(([x, y, min_z], [x, y, max_z]))
        for z in np.arange(min_z, max_z + grid_spacing, grid_spacing):
            grid_lines.append(([min_x, y, z], [max_x, y, z]))
    
    if include_yz_min_x:
        x = min_x
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            grid_lines.append(([x, y, min_z], [x, y, max_z]))
        for z in np.arange(min_z, max_z + grid_spacing, grid_spacing):
            grid_lines.append(([x, min_y, z], [x, max_y, z]))
    if include_yz_max_x:
        x = max_x
        for y in np.arange(min_y, max_y + grid_spacing, grid_spacing):
            grid_lines.append(([x, y, min_z], [x, y, max_z]))
        for z in np.arange(min_z, max_z + grid_spacing, grid_spacing):
            grid_lines.append(([x, min_y, z], [x, max_y, z]))
    
    return grid_lines

def render_point_cloud_view(points, colors, position, target, up, camera_params, bounds, point_size=1, max_points=50000, grid_spacing=10):
    # [Modified to use zorder for explicit layering]
    intrinsic = calculate_camera_intrinsics(camera_params)
    extrinsic = calculate_camera_extrinsics(position, target, up)
    
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_subset = points[indices]
        colors_subset = colors[indices]
    else:
        points_subset = points
        colors_subset = colors
    
    points_camera = transform_points(points_subset, extrinsic)
    mask = points_camera[:, 2] < 0
    points_camera = points_camera[mask]
    colors_subset = colors_subset[mask]
    pixels = project_points(points_camera, intrinsic)
    
    grid_lines = generate_grid_lines(bounds, position, grid_spacing)
    grid_pixels = []
    for start, end in grid_lines:
        start_camera = transform_points(np.array([start]), extrinsic)[0]
        end_camera = transform_points(np.array([end]), extrinsic)[0]
        if start_camera[2] < 0 or end_camera[2] < 0:
            start_2d = project_points(np.array([start_camera]), intrinsic)[0]
            end_2d = project_points(np.array([end_camera]), intrinsic)[0]
            grid_pixels.append((start_2d, end_2d))
    
    aspect_ratio = camera_params['width'] / camera_params['height']
    fig = plt.figure(figsize=(10, 10/aspect_ratio), dpi=100, facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    ax.set_xlim(0, camera_params['width'])
    ax.set_ylim(0, camera_params['height'])
    ax.invert_yaxis()
    ax.set_axis_off()
    
    # Draw grid lines first with low zorder (background)
    for start_2d, end_2d in grid_pixels:
        ax.plot([start_2d[0], end_2d[0]], [start_2d[1], end_2d[1]], 
                color='yellow', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Draw point cloud on top with higher zorder (foreground)
    ax.scatter(pixels[:, 0], pixels[:, 1], s=point_size, c=colors_subset, zorder=2)
    
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    img = Image.fromarray(img_array)
    img = img.resize((camera_params['width'], camera_params['height']), Image.LANCZOS)
    plt.close(fig)
    
    return img, extrinsic

def extract_omega_phi_kappa(rotation_matrix):
    # [Same as before, no changes]
    phi = np.arcsin(-rotation_matrix[2, 0])
    if np.abs(np.cos(phi)) > 1e-10:
        omega = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        kappa = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        omega = 0
        kappa = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
    return np.degrees(omega), np.degrees(phi), np.degrees(kappa)

def overlay_camera_info(img, position, extrinsic, camera_params, camera_index):
    # [Same as before, no changes]
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        except IOError:
            font = ImageFont.load_default()
    
    rotation_matrix = extrinsic[:3, :3]
    omega, phi, kappa = extract_omega_phi_kappa(rotation_matrix)
    X, Y, Z = position
    
    text = [
        f"Camera #{camera_index + 1}",
        f"Interior Orientation:",
        f"- Focal Length: {camera_params['focal_length']} mm",
        f"- Principal Point: ({camera_params['principal_point_x']:.2f}, {camera_params['principal_point_y']:.2f}) px",
        f"- Sensor Size: {camera_params['sensor_width']} x {camera_params['sensor_height']} mm",
        f"- Distortion: k1={camera_params['k1']:.6f}, k2={camera_params['k2']:.6f}, k3={camera_params['k3']:.6f}",
        f"- Distortion: p1={camera_params['p1']:.6f}, p2={camera_params['p2']:.6f}",
        f"Exterior Orientation:",
        f"- Position (X, Y, Z): ({X:.2f}, {Y:.2f}, {Z:.2f})",
        f"- Rotation Angles:",
        f"  Omega: {omega:.4f}°",
        f"  Phi: {phi:.4f}°",
        f"  Kappa: {kappa:.4f}°"
    ]
    
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(10, 10), (700, 440)], fill=(0, 0, 0, 128))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    
    draw = ImageDraw.Draw(img)
    y_position = 20
    for line in text:
        draw.text((20, y_position), line, fill=(255, 255, 255), font=font)
        y_position += 38
    
    return img

def main(input_file, output_dir="output", num_cameras=10, max_points=100000, grid_spacing=10):
    # [Same as before, no changes]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    points, colors, bounds = load_point_cloud(input_file)
    camera_params = dji_phantom_4_rtk_camera_params()
    camera_positions = generate_camera_positions(bounds, num_cameras=num_cameras)
    
    for i, (position, target, up) in enumerate(camera_positions):
        print(f"Rendering view from camera {i+1}...")
        img, extrinsic = render_point_cloud_view(points, colors, position, target, up, 
                                               camera_params, bounds, max_points=max_points, 
                                               grid_spacing=grid_spacing)
        img_with_info = overlay_camera_info(img, position, extrinsic, camera_params, i)
        output_path = os.path.join(output_dir, f"camera_{i+1}.jpg")
        img_with_info.save(output_path, "JPEG")
        print(f"Saved {output_path}")
    
    print(f"All {num_cameras} camera views have been rendered and saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate virtual camera views from a LiDAR point cloud with holodeck-style grid")
    parser.add_argument("input_file", help="Path to input LAS/LAZ file")
    parser.add_argument("--output_dir", default="output", help="Directory to save output images")
    parser.add_argument("--num_cameras", type=int, default=10, help="Number of camera positions to generate")
    parser.add_argument("--max_points", type=int, default=100000, help="Maximum number of points to render")
    parser.add_argument("--grid_spacing", type=float, default=10, help="Spacing between grid lines")
    
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.num_cameras, args.max_points, args.grid_spacing)