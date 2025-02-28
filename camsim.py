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
    """
    Load a LiDAR point cloud from a LAS/LAZ file.
    
    Args:
        file_path (str): Path to the LAS/LAZ file
        
    Returns:
        tuple: (points array, colors array, bounds of the point cloud (min_x, min_y, min_z, max_x, max_y, max_z))
    """
    print(f"Loading point cloud from {file_path}...")
    
    # Load the LAS/LAZ file using laspy
    las = laspy.read(file_path)
    
    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # If color information is available, extract it
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0  # Normalize to [0, 1]
    else:
        # If no color information, assign a default color (white)
        colors = np.ones((len(points), 3))  # White color
    
    # Get the bounds of the point cloud
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounds = (min_bound[0], min_bound[1], min_bound[2], max_bound[0], max_bound[1], max_bound[2])
    
    print(f"Point cloud loaded with {len(points)} points")
    print(f"Bounds: {bounds}")
    
    return points, colors, bounds

def generate_camera_positions(bounds, num_cameras=10, max_distance_factor=1.0):
    """
    Generate random camera positions around the point cloud.
    
    Args:
        bounds (tuple): (min_x, min_y, min_z, max_x, max_y, max_z)
        num_cameras (int): Number of camera positions to generate
        max_distance_factor (float): Maximum distance from center as a factor of the point cloud extent
        
    Returns:
        list: List of camera positions and orientations [(position, target, up), ...]
    """
    print(f"Generating {num_cameras} camera positions...")
    
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    
    # Calculate the center of the point cloud
    center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
    
    # Calculate the extents of the point cloud
    extents = [max_x - min_x, max_y - min_y, max_z - min_z]
    max_extent = max(extents)
    
    # Calculate the maximum distance
    max_distance = max_extent * max_distance_factor
    
    camera_positions = []
    
    for i in range(num_cameras):
        # Generate a random direction from the center
        theta = random.uniform(0, 2 * math.pi)  # Azimuth angle
        phi = random.uniform(0, math.pi)  # Polar angle
        
        # Calculate the distance from the center (random between 0.5 and max_distance_factor times the max extent)
        distance = random.uniform(0.5 * max_extent, max_distance)
        
        # Convert spherical coordinates to Cartesian coordinates
        x = center[0] + distance * math.sin(phi) * math.cos(theta)
        y = center[1] + distance * math.sin(phi) * math.sin(theta)
        z = center[2] + distance * math.cos(phi)
        
        # Camera position
        position = [x, y, z]
        
        # Camera target (looking at the center of the point cloud)
        target = center
        
        # Camera up vector (Z-axis is typically up in a geographic coordinate system)
        up = [0, 0, 1]
        
        camera_positions.append((position, target, up))
    
    print(f"Generated {len(camera_positions)} camera positions")
    return camera_positions

def dji_phantom_4_rtk_camera_params():
    """
    Return camera parameters for a DJI Phantom 4 RTK camera.
    
    Returns:
        dict: Camera parameters
    """
    # These are typical values for the DJI Phantom 4 RTK camera
    # Source: DJI specifications and common photogrammetry values
    
    params = {
        'width': 5472,  # Image width in pixels
        'height': 3648,  # Image height in pixels
        'sensor_width': 13.2,  # Sensor width in mm
        'sensor_height': 8.8,  # Sensor height in mm
        'focal_length': 8.8,  # Focal length in mm
        'pixel_size': 0.00241,  # Pixel size in mm
        'principal_point_x': 5472 / 2,  # Principal point x (typically center of image)
        'principal_point_y': 3648 / 2,  # Principal point y (typically center of image)
        'k1': -0.0012,  # Radial distortion coefficient k1
        'k2': 0.0006,  # Radial distortion coefficient k2
        'k3': -0.0001,  # Radial distortion coefficient k3
        'p1': 0.0001,  # Tangential distortion coefficient p1
        'p2': 0.0001,  # Tangential distortion coefficient p2
        'fov_x': None,  # Will be calculated
        'fov_y': None,  # Will be calculated
    }
    
    # Calculate field of view
    params['fov_x'] = 2 * math.atan(params['sensor_width'] / (2 * params['focal_length']))
    params['fov_y'] = 2 * math.atan(params['sensor_height'] / (2 * params['focal_length']))
    
    return params

def calculate_camera_intrinsics(camera_params):
    """
    Calculate camera intrinsic matrix from parameters.
    
    Args:
        camera_params (dict): Camera parameters
        
    Returns:
        numpy.ndarray: Camera intrinsic matrix
    """
    # Calculate focal length in pixels
    fx = camera_params['focal_length'] / camera_params['pixel_size']
    fy = camera_params['focal_length'] / camera_params['pixel_size']
    
    # Principal point
    cx = camera_params['principal_point_x']
    cy = camera_params['principal_point_y']
    
    # Create the intrinsic matrix
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return intrinsic

def calculate_camera_extrinsics(position, target, up):
    """
    Calculate the camera extrinsic matrix (view matrix) from position, target, and up vector.
    
    Args:
        position (list): Camera position [x, y, z]
        target (list): Camera target point [x, y, z]
        up (list): Camera up vector [x, y, z]
        
    Returns:
        numpy.ndarray: Camera extrinsic matrix (4x4)
    """
    position = np.array(position)
    target = np.array(target)
    up = np.array(up)
    
    # Calculate the camera's forward, right, and up vectors
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Create the rotation matrix
    rotation = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])
    
    # Create the translation matrix
    translation = np.array([
        [1, 0, 0, -position[0]],
        [0, 1, 0, -position[1]],
        [0, 0, 1, -position[2]],
        [0, 0, 0, 1]
    ])
    
    # Extrinsic matrix = rotation * translation
    extrinsic = np.matmul(rotation, translation)
    
    return extrinsic

def transform_points(points, extrinsic):
    """
    Transform points from world space to camera space using the extrinsic matrix.
    
    Args:
        points (numpy.ndarray): Points in world space (N x 3)
        extrinsic (numpy.ndarray): Extrinsic matrix (4x4)
        
    Returns:
        numpy.ndarray: Points in camera space (N x 3)
    """
    # Add homogeneous coordinate (w=1) to points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Transform points
    points_transformed = np.dot(points_homogeneous, extrinsic.T)
    
    # Convert back to 3D coordinates
    points_transformed = points_transformed[:, :3]
    
    return points_transformed

def project_points(points, intrinsic):
    """
    Project 3D points to 2D using the intrinsic matrix.
    
    Args:
        points (numpy.ndarray): Points in camera space (N x 3)
        intrinsic (numpy.ndarray): Intrinsic matrix (3x3)
        
    Returns:
        numpy.ndarray: Projected 2D points (N x 2)
    """
    # Normalize points by Z coordinate (perspective division)
    points_normalized = points.copy()
    mask = points[:, 2] != 0
    points_normalized[mask, 0] = points[mask, 0] / points[mask, 2]
    points_normalized[mask, 1] = points[mask, 1] / points[mask, 2]
    points_normalized[mask, 2] = 1.0
    
    # Project points using intrinsic matrix
    pixels = np.dot(points_normalized, intrinsic.T)
    
    # Return X, Y coordinates
    return pixels[:, :2]

def render_point_cloud_view(points, colors, position, target, up, camera_params, point_size=1, max_points=50000):
    """
    Render a view of the point cloud from a specific camera position using matplotlib.
    
    Args:
        points (numpy.ndarray): Points in the point cloud (N x 3)
        colors (numpy.ndarray): Colors of the points (N x 3)
        position (list): Camera position [x, y, z]
        target (list): Camera target point [x, y, z]
        up (list): Camera up vector [x, y, z]
        camera_params (dict): Camera parameters
        point_size (float): Size of points in the rendered image
        max_points (int): Maximum number of points to render (for performance)
        
    Returns:
        tuple: (PIL Image, extrinsic matrix)
    """
    # Calculate camera matrices
    intrinsic = calculate_camera_intrinsics(camera_params)
    extrinsic = calculate_camera_extrinsics(position, target, up)
    
    # If there are too many points, randomly sample some of them
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_subset = points[indices]
        colors_subset = colors[indices]
    else:
        points_subset = points
        colors_subset = colors
    
    # Transform points to camera space
    points_camera = transform_points(points_subset, extrinsic)
    
    # Keep only the points in front of the camera
    mask = points_camera[:, 2] < 0  # Z is negative in camera space (looking down -Z axis)
    points_camera = points_camera[mask]
    colors_subset = colors_subset[mask]
    
    # Project points to image space
    pixels = project_points(points_camera, intrinsic)
    
    # Create the figure with the specific aspect ratio
    aspect_ratio = camera_params['width'] / camera_params['height']
    fig = plt.figure(figsize=(10, 10/aspect_ratio), dpi=100, facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    # Set the limits to the image dimensions
    ax.set_xlim(0, camera_params['width'])
    ax.set_ylim(0, camera_params['height'])
    ax.invert_yaxis()  # Invert Y-axis to match image coordinates (0 at top)
    
    # Remove axis labels and ticks
    ax.set_axis_off()
    
    # Plot the projected points
    ax.scatter(pixels[:, 0], pixels[:, 1], s=point_size, c=colors_subset)
    
    # Adjust figure layout
    fig.tight_layout(pad=0)
    
    # Render the figure to a numpy array
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Resize to match the target resolution
    img = img.resize((camera_params['width'], camera_params['height']), Image.LANCZOS)
    
    # Close the figure to free memory
    plt.close(fig)
    
    return img, extrinsic

def extract_omega_phi_kappa(rotation_matrix):
    """
    Extract omega, phi, kappa angles from a rotation matrix.
    These are photogrammetric rotation angles.
    
    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix
        
    Returns:
        tuple: (omega, phi, kappa) in degrees
    """
    # Extract rotation angles (photogrammetric convention)
    # phi (y-rotation)
    phi = np.arcsin(-rotation_matrix[2, 0])
    
    # omega (x-rotation)
    if np.abs(np.cos(phi)) > 1e-10:
        omega = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        # Gimbal lock case
        omega = 0
    
    # kappa (z-rotation)
    if np.abs(np.cos(phi)) > 1e-10:
        kappa = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Gimbal lock case
        kappa = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
    
    # Convert to degrees
    omega_deg = np.degrees(omega)
    phi_deg = np.degrees(phi)
    kappa_deg = np.degrees(kappa)
    
    return omega_deg, phi_deg, kappa_deg

def overlay_camera_info(img, position, extrinsic, camera_params, camera_index):
    """
    Overlay camera information on the image.
    
    Args:
        img (PIL.Image): Image to overlay information on
        position (list): Camera position [x, y, z]
        extrinsic (numpy.ndarray): Camera extrinsic matrix
        camera_params (dict): Camera parameters
        camera_index (int): Index of the camera
        
    Returns:
        PIL.Image: Image with overlaid information
    """
    draw = ImageDraw.Draw(img)
    
    # Try to use a standard font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 36)  # 150% larger font (was 24)
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        except IOError:
            font = ImageFont.load_default()
    
    # Extract rotation angles (omega, phi, kappa) and position (X, Y, Z)
    rotation_matrix = extrinsic[:3, :3]
    omega, phi, kappa = extract_omega_phi_kappa(rotation_matrix)
    X, Y, Z = position
    
    # Prepare the text to overlay
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
    
    # Create a semi-transparent rectangle for better text visibility
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(10, 10), (700, 440)], fill=(0, 0, 0, 128))  # Made wider and taller for larger font
    
    # Paste the overlay onto the original image
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    
    # Draw the text
    draw = ImageDraw.Draw(img)
    y_position = 20
    for line in text:
        draw.text((20, y_position), line, fill=(255, 255, 255), font=font)
        y_position += 38  # Increased spacing for larger font (was 25)
    
    return img

def draw_ucs_icon(img, extrinsic, intrinsic, size=200):
    """
    Draw a 3D coordinate system (UCS) icon on the image that correctly represents
    the world coordinate axes from the camera's perspective.
    
    Args:
        img (PIL.Image): Image to draw on
        extrinsic (numpy.ndarray): Camera extrinsic matrix (4x4)
        intrinsic (numpy.ndarray): Camera intrinsic matrix (3x3)
        size (int): Size of the UCS icon background
        
    Returns:
        PIL.Image: Image with UCS icon
    """
    # Create a copy of the image to draw on
    img_with_ucs = img.copy()
    
    # Get image dimensions
    img_width, img_height = img.size
    
    # Position in the bottom-right corner of the image with margin
    origin_x = img_width - size - 40
    origin_y = img_height - size - 40
    
    # Define the world coordinate system axes in 3D space
    # We'll use a reasonable scale for the axes
    axis_length = 1.0  # 1 unit in world space
    
    # World origin and axis endpoints in world space
    world_origin = np.array([0, 0, 0, 1])  # Homogeneous coordinates
    world_x_end = np.array([axis_length, 0, 0, 1])
    world_y_end = np.array([0, axis_length, 0, 1])
    world_z_end = np.array([0, 0, axis_length, 1])
    
    # Transform world points to camera space
    camera_origin = extrinsic.dot(world_origin)
    camera_x_end = extrinsic.dot(world_x_end)
    camera_y_end = extrinsic.dot(world_y_end)
    camera_z_end = extrinsic.dot(world_z_end)
    
    # Convert to 3D coordinates (divide by w, which should be 1 in this case)
    camera_origin = camera_origin[:3]
    camera_x_end = camera_x_end[:3]
    camera_y_end = camera_y_end[:3]
    camera_z_end = camera_z_end[:3]
    
    # Project to 2D image space
    # We need to handle the perspective division manually
    def project_point(point):
        # Only project if the point is in front of the camera (z < 0 in camera space)
        if point[2] < 0:
            # Perspective division
            x = point[0] / -point[2]
            y = point[1] / -point[2]
            
            # Apply intrinsic matrix
            px = intrinsic[0, 0] * x + intrinsic[0, 2]
            py = intrinsic[1, 1] * y + intrinsic[1, 2]
            
            return (px, py)
        else:
            # Point is behind the camera, can't be seen
            return None
    
    # Project origin and endpoints
    origin_2d = project_point(camera_origin)
    x_end_2d = project_point(camera_x_end)
    y_end_2d = project_point(camera_y_end)
    z_end_2d = project_point(camera_z_end)
    
    # Check if any point is behind the camera
    if None in [origin_2d, x_end_2d, y_end_2d, z_end_2d]:
        # If world origin is behind camera, we need a different approach
        # Instead of trying to project the actual world axes,
        # we'll create a fixed-position UCS icon in the corner
        
        # Define the icon position and size
        icon_origin = (origin_x, origin_y)
        icon_scale = size * 0.5
        
        # Create 2D coordinates for axes that are visually meaningful
        origin_2d = icon_origin
        x_end_2d = (origin_x + icon_scale, origin_y)
        y_end_2d = (origin_x, origin_y - icon_scale)
        z_end_2d = (origin_x - icon_scale * 0.5, origin_y - icon_scale * 0.5)
    else:
        # If all points are visible, we'll preserve their orientation but
        # translate them to our desired corner position and scale them appropriately
        
        # First, center the axes on the origin
        x_vec = (x_end_2d[0] - origin_2d[0], x_end_2d[1] - origin_2d[1])
        y_vec = (y_end_2d[0] - origin_2d[0], y_end_2d[1] - origin_2d[1])
        z_vec = (z_end_2d[0] - origin_2d[0], z_end_2d[1] - origin_2d[1])
        
        # Calculate the maximum vector length for normalization
        max_length = max(
            math.sqrt(x_vec[0]**2 + x_vec[1]**2),
            math.sqrt(y_vec[0]**2 + y_vec[1]**2),
            math.sqrt(z_vec[0]**2 + z_vec[1]**2)
        )
        
        # Normalize and scale vectors - make them larger (200% of original size)
        icon_scale = size * 0.7  # Increased from 0.5 to 0.7 (140% larger)
        if max_length > 0:
            x_vec = (x_vec[0] * icon_scale / max_length, x_vec[1] * icon_scale / max_length)
            y_vec = (y_vec[0] * icon_scale / max_length, y_vec[1] * icon_scale / max_length)
            z_vec = (z_vec[0] * icon_scale / max_length, z_vec[1] * icon_scale / max_length)
        
        # Reposition to corner
        origin_2d = (origin_x, origin_y)
        x_end_2d = (origin_x + x_vec[0], origin_y + x_vec[1])
        y_end_2d = (origin_x + y_vec[0], origin_y + y_vec[1])
        z_end_2d = (origin_x + z_vec[0], origin_y + z_vec[1])
    
    # Create a semi-transparent black background for the UCS icon
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Determine the rectangle for the background based on the axes
    min_x = min(origin_2d[0], x_end_2d[0], y_end_2d[0], z_end_2d[0]) - 50  # More padding
    min_y = min(origin_2d[1], x_end_2d[1], y_end_2d[1], z_end_2d[1]) - 50  # More padding
    max_x = max(origin_2d[0], x_end_2d[0], y_end_2d[0], z_end_2d[0]) + 80  # More padding
    max_y = max(origin_2d[1], x_end_2d[1], y_end_2d[1], z_end_2d[1]) + 80  # More padding
    
    overlay_draw.rectangle(
        [(min_x, min_y), (max_x, max_y)], 
        fill=(0, 0, 0, 180)
    )
    
    # Apply the overlay
    img_with_ucs = img_with_ucs.convert("RGBA")
    img_with_ucs = Image.alpha_composite(img_with_ucs, overlay)
    img_with_ucs = img_with_ucs.convert("RGB")
    
    # Get a new drawing context
    draw = ImageDraw.Draw(img_with_ucs)
    
    # Define thicker line width for better visibility (200% larger)
    line_width = 16  # Increased from 8 to 16
    
    # Draw the axes with different colors and thicker lines
    draw.line([origin_2d, x_end_2d], fill=(255, 0, 0), width=line_width)  # X-axis in red
    draw.line([origin_2d, y_end_2d], fill=(0, 255, 0), width=line_width)  # Y-axis in green
    draw.line([origin_2d, z_end_2d], fill=(0, 0, 255), width=line_width)  # Z-axis in blue
    
    # Draw axis heads (arrowheads) - make them 200% larger
    arrow_size = 30  # Increased from 15 to 30
    
    # Helper function to draw arrowhead
    def draw_arrowhead(start, end, color):
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Calculate length of the vector
        length = math.sqrt(dx**2 + dy**2)
        
        if length < 1e-6:  # Avoid division by zero
            return
        
        # Normalize the vector
        dx /= length
        dy /= length
        
        # Calculate perpendicular vector
        px = -dy
        py = dx
        
        # Arrow points
        arrow_point1 = (
            end[0] - arrow_size * dx + arrow_size/2 * px,
            end[1] - arrow_size * dy + arrow_size/2 * py
        )
        arrow_point2 = (
            end[0] - arrow_size * dx - arrow_size/2 * px,
            end[1] - arrow_size * dy - arrow_size/2 * py
        )
        
        # Draw the arrowhead
        draw.polygon([end, arrow_point1, arrow_point2], fill=color)
    
    # Draw arrowheads
    draw_arrowhead(origin_2d, x_end_2d, (255, 0, 0))  # X-axis
    draw_arrowhead(origin_2d, y_end_2d, (0, 255, 0))  # Y-axis
    draw_arrowhead(origin_2d, z_end_2d, (0, 0, 255))  # Z-axis
    
    # Add labels with larger font size (200% larger)
    try:
        font = ImageFont.truetype("arial.ttf", 60)  # Increased from 40 to 60
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
        except IOError:
            font = ImageFont.load_default()
    
    # Position the labels at the end of each axis with additional offset
    label_offset = 15  # Increased offset slightly
    draw.text((x_end_2d[0] + label_offset, x_end_2d[1] - label_offset), "X", fill=(255, 0, 0), font=font)
    draw.text((y_end_2d[0] - label_offset, y_end_2d[1] - label_offset), "Y", fill=(0, 255, 0), font=font)
    draw.text((z_end_2d[0] - label_offset, z_end_2d[1] - label_offset), "Z", fill=(0, 0, 255), font=font)
    
    # Add a label indicating this is the UCS
    draw.text((origin_2d[0], origin_2d[1] + 40), "WORLD", fill=(255, 255, 255), font=font)
    
    return img_with_ucs

def main(input_file, output_dir="output", num_cameras=10, max_points=100000):
    """
    Main function to process a LiDAR point cloud and generate camera views.
    
    Args:
        input_file (str): Path to input LAS/LAZ file
        output_dir (str): Directory to save output images
        num_cameras (int): Number of camera positions to generate
        max_points (int): Maximum number of points to render (for performance)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load the point cloud
    points, colors, bounds = load_point_cloud(input_file)
    
    # Get camera parameters for a DJI Phantom 4 RTK
    camera_params = dji_phantom_4_rtk_camera_params()
    
    # Generate camera positions
    camera_positions = generate_camera_positions(bounds, num_cameras=num_cameras)
    
    # Render views from each camera position
    for i, (position, target, up) in enumerate(camera_positions):
        print(f"Rendering view from camera {i+1}...")
        
        # Render the view
        img, extrinsic = render_point_cloud_view(points, colors, position, target, up, camera_params, max_points=max_points)
        
        # Calculate intrinsic matrix
        intrinsic = calculate_camera_intrinsics(camera_params)
        
        # Overlay camera information
        img_with_info = overlay_camera_info(img, position, extrinsic, camera_params, i)
        
        # Add UCS icon with proper orientation based on camera position
        # Make the UCS icon 200% larger (400 instead of 200)
        img_with_ucs = draw_ucs_icon(img_with_info, extrinsic, intrinsic, size=400)
        
        # Save the image
        output_path = os.path.join(output_dir, f"camera_{i+1}.jpg")
        img_with_ucs.save(output_path, "JPEG")
        print(f"Saved {output_path}")
    
    print(f"All {num_cameras} camera views have been rendered and saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate virtual camera views from a LiDAR point cloud")
    parser.add_argument("input_file", help="Path to input LAS/LAZ file")
    parser.add_argument("--output_dir", default="output", help="Directory to save output images")
    parser.add_argument("--num_cameras", type=int, default=10, help="Number of camera positions to generate")
    parser.add_argument("--max_points", type=int, default=100000, help="Maximum number of points to render (for performance)")
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_dir, args.num_cameras, args.max_points)