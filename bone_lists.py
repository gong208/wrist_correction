bone_list_behave = [
    (0, 1),   # pelvis → left_hip
    (0, 2),   # pelvis → right_hip
    (0, 3),   # pelvis → spine1
    (1, 4),   # left_hip → left_knee
    (2, 5),   # right_hip → right_knee
    (3, 6),   # spine1 → spine2
    (4, 7),   # left_knee → left_ankle
    (5, 8),   # right_knee → right_ankle
    (6, 9),   # spine2 → spine3
    (7, 10),  # left_ankle → left_foot
    (8, 11),  # right_ankle → right_foot
    (9, 12),  # spine3 → neck
    (12, 13), # neck → left_collar
    (12, 14), # neck → right_collar
    (12, 15), # neck → head
    (13, 16), # left_collar → left_shoulder
    (14, 17), # right_collar → right_shoulder
    (16, 18), # left_shoulder → left_elbow
    (17, 19), # right_shoulder → right_elbow
    (18, 20), # left_elbow → left_wrist
    (19, 21), # right_elbow → right_wrist

    # Left fingers
    (20, 22), (22, 23), (23, 24),       # left_index
    (20, 25), (25, 26), (26, 27),       # left_middle
    (20, 28), (28, 29), (29, 30),       # left_pinky
    (20, 31), (31, 32), (32, 33),       # left_ring
    (20, 34), (34, 35), (35, 36),       # left_thumb

    # Right fingers
    (21, 37), (37, 38), (38, 39),       # right_index
    (21, 40), (40, 41), (41, 42),       # right_middle
    (21, 43), (43, 44), (44, 45),       # right_pinky
    (21, 46), (46, 47), (47, 48),       # right_ring
    (21, 49), (49, 50), (50, 51),       # right_thumb

    # Facial landmarks (optional)
    (15, 52),  # head → nose
    (52, 53),  # nose → right_eye
    (52, 54),  # nose → left_eye
    (53, 55),  # right_eye → right_ear
    (54, 56),  # left_eye → left_ear

    # Feet toes
    (10, 57), (10, 58), (10, 59),       # left_foot → big toe, small toe, heel
    (11, 60), (11, 61), (11, 62),       # right_foot → big toe, small toe, heel

    # Mid-hand landmarks (optional, may overlap with fingers above)
    # (20, 63), (20, 64), (20, 65), (20, 66), (20, 67),  # left hand landmarks
    # (21, 68), (21, 69), (21, 70), (21, 71), (21, 72),  # right hand landmarks
]

bone_list_omomo = [
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Spine and neck
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left arm
    (3, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (3, 14), (14, 17), (17, 19), (19, 21),

    # Left hand (starts from left_wrist = 20)
    (20, 25), (25, 26), (26, 27),       # left_index1-3
    (20, 28), (28, 29), (29, 30),       # left_middle1-3
    (20, 31), (31, 32), (32, 33),       # left_pinky1-3
    (20, 34), (34, 35), (35, 36),       # left_ring1-3
    (20, 37), (37, 38), (38, 39),       # left_thumb1-3

    # Right hand (starts from right_wrist = 21)
    (21, 40), (40, 41), (41, 42),       # right_index1-3
    (21, 43), (43, 44), (44, 45),       # right_middle1-3
    (21, 46), (46, 47), (47, 48),       # right_pinky1-3
    (21, 49), (49, 50), (50, 51),       # right_ring1-3
    (21, 52), (52, 53), (53, 54)        # right_thumb1-3
]