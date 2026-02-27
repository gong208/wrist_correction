import numpy as np
import argparse

def axis_angle_to_readable(pose_vector):
    """
    Convert axis-angle representation to readable format.
    
    Args:
        pose_vector (np.array): 3D rotation vector [rx, ry, rz]
    
    Returns:
        tuple: (rotation_axis, rotation_angle_degrees)
    """
    # Calculate magnitude (rotation angle in radians)
    magnitude = np.linalg.norm(pose_vector)
    
    if magnitude < 1e-6:  # Very small rotation
        return np.array([0, 0, 0]), 0.0
    
    # Calculate unit vector (rotation axis)
    rotation_axis = pose_vector / magnitude
    
    # Convert angle to degrees
    rotation_angle_degrees = np.rad2deg(magnitude)
    
    return rotation_axis, rotation_angle_degrees

def angle_between_axes(axis1, axis2):
    """
    Calculate the angle between two rotation axes, handling axis-angle representation ambiguities.
    
    Args:
        axis1 (np.array): First rotation axis (unit vector)
        axis2 (np.array): Second rotation axis (unit vector)
    
    Returns:
        float: Angle between axes in degrees
    """
    # Calculate dot product
    dot_product = np.dot(axis1, axis2)
    
    # Clamp to [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    # Consider both positive and negative axes (equivalent representations)
    angle_rad = np.arccos(abs(dot_product))  # Use abs to get smallest angle
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg

def rotation_matrix_difference(pose1, pose2):
    """
    Calculate the difference between two rotations using rotation matrices.
    This is often more stable than axis-angle comparisons.
    
    Args:
        pose1 (np.array): First pose vector (3D rotation vector)
        pose2 (np.array): Second pose vector (3D rotation vector)
    
    Returns:
        float: Rotation difference in degrees
    """
    from scipy.spatial.transform import Rotation
    
    # Convert axis-angle to rotation objects
    rot1 = Rotation.from_rotvec(pose1)
    rot2 = Rotation.from_rotvec(pose2)
    
    # Calculate the relative rotation
    relative_rot = rot1.inv() * rot2
    
    # Get the rotation angle in degrees
    angle_diff = np.rad2deg(relative_rot.magnitude())
    
    return angle_diff


def print_joint_poses(file_path):
    """
    Load poses from .npy or .npz file and print angle differences between consecutive frames
    for shoulder, elbow, wrist, and joints 13 and 14.
    
    Args:
        file_path (str): Path to the .npy or .npz file containing poses
    """
    try:
        # Check file extension to determine loading method
        if file_path.endswith('.npz'):
            # Load from .npz file (human.npz format)
            with np.load(file_path, allow_pickle=True) as data:
                print(f"Loaded .npz file with keys: {list(data.keys())}")
                
                # Extract poses from the npz file
                if 'poses' in data:
                    poses = data['poses']
                    print(f"Found poses with shape: {poses.shape}")
                else:
                    print("Error: 'poses' key not found in .npz file")
                    print(f"Available keys: {list(data.keys())}")
                    return
                
                # Also print other available data for reference
                if 'betas' in data:
                    print(f"Betas shape: {data['betas'].shape}")
                if 'trans' in data:
                    print(f"Trans shape: {data['trans'].shape}")
                if 'gender' in data:
                    print(f"Gender: {data['gender']}")
        else:
            # Load from .npy file
            poses = np.load(file_path)
            print(f"Loaded poses from .npy file with shape: {poses.shape}")
        
        # SMPLX joint indices
        LEFT_SHOULDER = 16   # Left shoulder joint index in SMPLX
        RIGHT_SHOULDER = 17  # Right shoulder joint index in SMPLX
        LEFT_ELBOW = 18      # Left elbow joint index in SMPLX
        RIGHT_ELBOW = 19     # Right elbow joint index in SMPLX
        LEFT_WRIST = 20      # Left wrist joint index in SMPLX
        RIGHT_WRIST = 21     # Right wrist joint index in SMPLX
        JOINT_13 = 13        # Joint 13
        JOINT_14 = 14        # Joint 14
        
        # Check if we have the expected pose format
        if len(poses.shape) == 2:
            # This is likely the raw pose parameters (not joint positions)
            # We need to convert to joint positions using SMPLX
            print("Detected pose parameters format. Calculating axis angle differences between consecutive frames...")
            
            # For now, just print the pose parameters for the specified joints
            # In SMPLX, pose parameters are organized as:
            # Root orientation: 0:3
            # Body pose: 3:66 (21 joints × 3 parameters each)
            # Hand pose: 66:156 (30 hand joints × 3 parameters each)
            
            num_frames = poses.shape[0]
            print(f"Number of frames: {num_frames}")
            print("\nFrame | L.Shoulder | R.Shoulder | L.Elbow | R.Elbow | L.Wrist | R.Wrist | Joint13 | Joint14")
            print("-" * 120)
            
            # Store previous frame poses for comparison
            prev_poses = None
            
            for frame in range(num_frames):
                # Extract pose parameters for each joint
                # Joint to pose mapping: joint_index * 3 + offset
                # Root orientation: 0:3
                # Body pose: 3:66 (joints 1-21)
                # Hand pose: 66:156 (joints 22-51)
                
                # Left shoulder (joint 16): pose indices 48:51 in full pose
                left_shoulder_pose = poses[frame, 48:51]
                # Right shoulder (joint 17): pose indices 51:54 in full pose
                right_shoulder_pose = poses[frame, 51:54]
                # Left elbow (joint 18): pose indices 54:57 in full pose
                left_elbow_pose = poses[frame, 54:57]
                # Right elbow (joint 19): pose indices 57:60 in full pose
                right_elbow_pose = poses[frame, 57:60]
                # Left wrist (joint 20): pose indices 60:63 in full pose
                left_wrist_pose = poses[frame, 60:63]
                # Right wrist (joint 21): pose indices 63:66 in full pose
                right_wrist_pose = poses[frame, 63:66]
                # Joint 13: pose indices 39:42 in full pose
                joint13_pose = poses[frame, 39:42]
                # Joint 14: pose indices 42:45 in full pose
                joint14_pose = poses[frame, 42:45]
                
                # Store pose vectors for comparison
                current_poses = {
                    'l_shoulder': left_shoulder_pose,
                    'r_shoulder': right_shoulder_pose,
                    'l_elbow': left_elbow_pose,
                    'r_elbow': right_elbow_pose,
                    'l_wrist': left_wrist_pose,
                    'r_wrist': right_wrist_pose,
                    'joint13': joint13_pose,
                    'joint14': joint14_pose
                }
                
                if prev_poses is not None:
                    # Calculate angle differences between current and previous frame
                    l_shoulder_diff = rotation_matrix_difference(current_poses['l_shoulder'], prev_poses['l_shoulder'])
                    r_shoulder_diff = rotation_matrix_difference(current_poses['r_shoulder'], prev_poses['r_shoulder'])
                    l_elbow_diff = rotation_matrix_difference(current_poses['l_elbow'], prev_poses['l_elbow'])
                    r_elbow_diff = rotation_matrix_difference(current_poses['r_elbow'], prev_poses['r_elbow'])
                    l_wrist_diff = rotation_matrix_difference(current_poses['l_wrist'], prev_poses['l_wrist'])
                    r_wrist_diff = rotation_matrix_difference(current_poses['r_wrist'], prev_poses['r_wrist'])
                    joint13_diff = rotation_matrix_difference(current_poses['joint13'], prev_poses['joint13'])
                    joint14_diff = rotation_matrix_difference(current_poses['joint14'], prev_poses['joint14'])
                    
                    print(f"{frame:5d} | "
                          f"{l_shoulder_diff:8.2f}° | "
                          f"{r_shoulder_diff:8.2f}° | "
                          f"{l_elbow_diff:8.2f}° | "
                          f"{r_elbow_diff:8.2f}° | "
                          f"{l_wrist_diff:8.2f}° | "
                          f"{r_wrist_diff:8.2f}° | "
                          f"{joint13_diff:8.2f}° | "
                          f"{joint14_diff:8.2f}°")
                else:
                    # First frame - no previous frame to compare with
                    print(f"{frame:5d} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8}")
                
                # Store current poses for next iteration
                prev_poses = current_poses
                
        elif len(poses.shape) == 3:
            # This might be joint positions directly
            print("Detected joint positions format.")
            num_frames = poses.shape[0]
            print(f"Number of frames: {num_frames}")
            print("\nFrame | L.Shoulder | R.Shoulder | L.Elbow | R.Elbow | L.Wrist | R.Wrist | Joint13 | Joint14")
            print("-" * 120)
            
            # Store previous frame positions for comparison
            prev_positions = None
            
            for frame in range(num_frames):
                left_shoulder_pos = poses[frame, LEFT_SHOULDER]
                right_shoulder_pos = poses[frame, RIGHT_SHOULDER]
                left_elbow_pos = poses[frame, LEFT_ELBOW]
                right_elbow_pos = poses[frame, RIGHT_ELBOW]
                left_wrist_pos = poses[frame, LEFT_WRIST]
                right_wrist_pos = poses[frame, RIGHT_WRIST]
                joint13_pos = poses[frame, JOINT_13]
                joint14_pos = poses[frame, JOINT_14]
                
                current_positions = {
                    'l_shoulder': left_shoulder_pos,
                    'r_shoulder': right_shoulder_pos,
                    'l_elbow': left_elbow_pos,
                    'r_elbow': right_elbow_pos,
                    'l_wrist': left_wrist_pos,
                    'r_wrist': right_wrist_pos,
                    'joint13': joint13_pos,
                    'joint14': joint14_pos
                }
                
                if prev_positions is not None:
                    # Calculate position differences between current and previous frame
                    l_shoulder_diff = np.linalg.norm(current_positions['l_shoulder'] - prev_positions['l_shoulder'])
                    r_shoulder_diff = np.linalg.norm(current_positions['r_shoulder'] - prev_positions['r_shoulder'])
                    l_elbow_diff = np.linalg.norm(current_positions['l_elbow'] - prev_positions['l_elbow'])
                    r_elbow_diff = np.linalg.norm(current_positions['r_elbow'] - prev_positions['r_elbow'])
                    l_wrist_diff = np.linalg.norm(current_positions['l_wrist'] - prev_positions['l_wrist'])
                    r_wrist_diff = np.linalg.norm(current_positions['r_wrist'] - prev_positions['r_wrist'])
                    joint13_diff = np.linalg.norm(current_positions['joint13'] - prev_positions['joint13'])
                    joint14_diff = np.linalg.norm(current_positions['joint14'] - prev_positions['joint14'])
                    
                    print(f"{frame:5d} | "
                          f"{l_shoulder_diff:8.3f} | "
                          f"{r_shoulder_diff:8.3f} | "
                          f"{l_elbow_diff:8.3f} | "
                          f"{r_elbow_diff:8.3f} | "
                          f"{l_wrist_diff:8.3f} | "
                          f"{r_wrist_diff:8.3f} | "
                          f"{joint13_diff:8.3f} | "
                          f"{joint14_diff:8.3f}")
                else:
                    # First frame - no previous frame to compare with
                    print(f"{frame:5d} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8} | "
                          f"{'N/A':>8}")
                
                # Store current positions for next iteration
                prev_positions = current_positions
        else:
            print(f"Unexpected pose format with shape: {poses.shape}")
            return
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading poses: {e}")

def main():
    parser = argparse.ArgumentParser(description='Print angle differences between consecutive frames for shoulder, elbow, wrist, and joints 13 and 14 from .npy or .npz file')
    parser.add_argument('file', help='Path to the .npy or .npz file containing poses')
    args = parser.parse_args()
    
    print_joint_poses(args.file)

if __name__ == "__main__":
    main() 