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

def print_joint_axes(file_path):
    """
    Load poses from .npy or .npz file and print axis and rotation angle for each frame
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
            print("Detected pose parameters format. Printing axis and rotation angle for each frame...")
            
            # For SMPLX, pose parameters are organized as:
            # Root orientation: 0:3
            # Body pose: 3:66 (21 joints × 3 parameters each)
            # Hand pose: 66:156 (30 hand joints × 3 parameters each)
            
            num_frames = poses.shape[0]
            print(f"Number of frames: {num_frames}")
            
            # Print header for each joint
            joints_to_analyze = [
                ('L.Shoulder', 48, 51),   # pose indices 48:51
                ('R.Shoulder', 51, 54),   # pose indices 51:54
                ('L.Elbow', 54, 57),      # pose indices 54:57
                ('R.Elbow', 57, 60),      # pose indices 57:60
                ('L.Wrist', 60, 63),      # pose indices 60:63
                ('R.Wrist', 63, 66),      # pose indices 63:66
                ('Joint13', 39, 42),      # pose indices 39:42
                ('Joint14', 42, 45)       # pose indices 42:45
            ]
            
            # Print header
            header = "Frame"
            for joint_name, _, _ in joints_to_analyze:
                header += f" | {joint_name:>12} Axis | {joint_name:>12} Angle"
            print(header)
            print("-" * (len(header) + 20))
            
            for frame in range(num_frames):
                line = f"{frame:5d}"
                
                for joint_name, start_idx, end_idx in joints_to_analyze:
                    pose_vector = poses[frame, start_idx:end_idx]
                    rotation_axis, rotation_angle = axis_angle_to_readable(pose_vector)
                    
                    # Format axis as [x, y, z] with 3 decimal places
                    axis_str = f"[{rotation_axis[0]:.3f}, {rotation_axis[1]:.3f}, {rotation_axis[2]:.3f}]"
                    angle_str = f"{rotation_angle:8.2f}°"
                    
                    line += f" | {axis_str:>12} | {angle_str:>12}"
                
                print(line)
                
        elif len(poses.shape) == 3:
            # This might be joint positions directly
            print("Detected joint positions format.")
            print("Note: This format contains joint positions, not pose parameters.")
            print("Axis-angle information is not available for joint positions.")
            return
        else:
            print(f"Unexpected pose format with shape: {poses.shape}")
            return
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading poses: {e}")

def print_joint_axes_detailed(file_path, frame_range=None):
    """
    Print detailed axis-angle information for specific frames.
    
    Args:
        file_path (str): Path to the .npy or .npz file containing poses
        frame_range (tuple): Optional (start_frame, end_frame) to limit output
    """
    # Load poses
    if file_path.endswith('.npz'):
        with np.load(file_path, allow_pickle=True) as data:
            poses = data['poses']
    else:
        poses = np.load(file_path)
    
    if len(poses.shape) != 2:
        print("Error: Expected 2D pose array")
        return
    
    num_frames = poses.shape[0]
    
    # Define frames to analyze
    if frame_range:
        start_frame, end_frame = frame_range
        frames_to_analyze = range(max(0, start_frame), min(num_frames, end_frame))
    else:
        frames_to_analyze = range(num_frames)
    
    # Joint definitions
    joints_to_analyze = [
        ('L.Shoulder', 48, 51),
        ('R.Shoulder', 51, 54),
        ('L.Elbow', 54, 57),
        ('R.Elbow', 57, 60),
        ('L.Wrist', 60, 63),
        ('R.Wrist', 63, 66),
        ('Joint13', 39, 42),
        ('Joint14', 42, 45)
    ]
    
    print(f"Detailed axis-angle analysis for frames {frames_to_analyze[0]}-{frames_to_analyze[-1]}")
    print("=" * 80)
    
    for frame in frames_to_analyze:
        print(f"\nFrame {frame}:")
        print("-" * 40)
        
        for joint_name, start_idx, end_idx in joints_to_analyze:
            pose_vector = poses[frame, start_idx:end_idx]
            rotation_axis, rotation_angle = axis_angle_to_readable(pose_vector)
            
            print(f"  {joint_name}:")
            print(f"    Raw pose vector: [{pose_vector[0]:.6f}, {pose_vector[1]:.6f}, {pose_vector[2]:.6f}]")
            print(f"    Rotation axis: [{rotation_axis[0]:.6f}, {rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]")
            print(f"    Rotation angle: {rotation_angle:.2f}°")
            
            # Calculate magnitude for verification
            magnitude = np.linalg.norm(pose_vector)
            print(f"    Magnitude: {magnitude:.6f} radians ({np.rad2deg(magnitude):.2f}°)")
            print()

def print_problematic_frames(file_path, original_file_path=None):
    """
    Print detailed analysis of problematic frames (like frame 51) to understand the fixing issues.
    
    Args:
        file_path (str): Path to the fixed poses file
        original_file_path (str): Path to the original poses file for comparison
    """
    # Load fixed poses
    if file_path.endswith('.npz'):
        with np.load(file_path, allow_pickle=True) as data:
            fixed_poses = data['poses']
    else:
        fixed_poses = np.load(file_path)
    
    # Load original poses if provided
    original_poses = None
    if original_file_path:
        if original_file_path.endswith('.npz'):
            with np.load(original_file_path, allow_pickle=True) as data:
                original_poses = data['poses']
        else:
            original_poses = np.load(original_file_path)
    
    # Define problematic frames to analyze
    problematic_frames = [50, 51, 52]  # Around the major flip
    
    # Joint definitions
    joints_to_analyze = [
        ('L.Shoulder', 48, 51),
        ('R.Shoulder', 51, 54),
        ('L.Elbow', 54, 57),
        ('R.Elbow', 57, 60),
        ('L.Wrist', 60, 63),
        ('R.Wrist', 63, 66),
        ('Joint13', 39, 42),
        ('Joint14', 42, 45)
    ]
    
    print("Analysis of Problematic Frames")
    print("=" * 80)
    
    for frame in problematic_frames:
        print(f"\nFrame {frame}:")
        print("-" * 40)
        
        for joint_name, start_idx, end_idx in joints_to_analyze:
            print(f"  {joint_name}:")
            
            # Fixed poses
            fixed_pose_vector = fixed_poses[frame, start_idx:end_idx]
            fixed_axis, fixed_angle = axis_angle_to_readable(fixed_pose_vector)
            
            print(f"    Fixed pose: [{fixed_pose_vector[0]:.6f}, {fixed_pose_vector[1]:.6f}, {fixed_pose_vector[2]:.6f}]")
            print(f"    Fixed axis: [{fixed_axis[0]:.6f}, {fixed_axis[1]:.6f}, {fixed_axis[2]:.6f}]")
            print(f"    Fixed angle: {fixed_angle:.2f}°")
            
            # Original poses (if available)
            if original_poses is not None:
                original_pose_vector = original_poses[frame, start_idx:end_idx]
                original_axis, original_angle = axis_angle_to_readable(original_pose_vector)
                
                print(f"    Original pose: [{original_pose_vector[0]:.6f}, {original_pose_vector[1]:.6f}, {original_pose_vector[2]:.6f}]")
                print(f"    Original axis: [{original_axis[0]:.6f}, {original_axis[1]:.6f}, {original_axis[2]:.6f}]")
                print(f"    Original angle: {original_angle:.2f}°")
                
                # Calculate difference
                pose_diff = fixed_pose_vector - original_pose_vector
                angle_diff = fixed_angle - original_angle
                print(f"    Pose difference: [{pose_diff[0]:.6f}, {pose_diff[1]:.6f}, {pose_diff[2]:.6f}]")
                print(f"    Angle difference: {angle_diff:.2f}°")
            
            print()

def main():
    parser = argparse.ArgumentParser(description='Print axis and rotation angle for each pose at each frame from .npy or .npz file')
    parser.add_argument('file', help='Path to the .npy or .npz file containing poses')
    parser.add_argument('--detailed', action='store_true', help='Print detailed information for each joint')
    parser.add_argument('--start-frame', type=int, help='Start frame for detailed analysis')
    parser.add_argument('--end-frame', type=int, help='End frame for detailed analysis')
    parser.add_argument('--problematic', action='store_true', help='Analyze problematic frames (50-52)')
    parser.add_argument('--original', type=str, help='Path to original poses file for comparison')
    
    args = parser.parse_args()
    
    if args.problematic:
        print_problematic_frames(args.file, args.original)
    elif args.detailed:
        frame_range = None
        if args.start_frame is not None and args.end_frame is not None:
            frame_range = (args.start_frame, args.end_frame)
        print_joint_axes_detailed(args.file, frame_range)
    else:
        print_joint_axes(args.file)

if __name__ == "__main__":
    main() 