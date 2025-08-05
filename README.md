1. To visualize ground thruth sequence, run 
```
python visualize_single_sequence.py --dataset_path data/omomo --sequence_name sub9_tripod_019
```
You can modify the transparency of human/object mesh in `render/mesh_viz.py` at line 72 and 79 respectively by editing the alpha channel.
To move the camera position you can modify the array at line 83 in `render/mesh_utils.py`.

2. First you should fix the poses of joints left/right collar, shoulder, elbow, and wrist by running
```
python joint_pose_fix.py --dataset_path data/omomo --sequence_name sub9_tripod_019
```

3. Then you can fix the palm orientation by runnning
```
python palm_fix.py --dataset_path data/omomo --sequence_name sub9_tripod_019
```