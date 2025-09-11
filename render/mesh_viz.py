import numpy as np
import trimesh
import math
import torch
from render.mesh_utils import MeshViewer
from render.utils import colors
import imageio
import pyrender
from PIL import Image
from PIL import ImageDraw 

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c

def visualize_body_obj(body_verts, body_face, obj_verts, obj_face, save_path,
                       multi_angle=False, h=768, w=768, bg_color='white', show_frame=True,
                       highlight_frame=None, highlight_vertex=None):
    """Visualize body and object with optional highlight for a specific frame and vertex."""
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    # Convert tensors to numpy if needed
    if torch.is_tensor(body_verts):
        mesh_rec = body_verts.cpu().numpy()
    else:
        mesh_rec = body_verts
        
    if torch.is_tensor(obj_verts):
        obj_mesh_rec = obj_verts.cpu().numpy()
    else:
        obj_mesh_rec = obj_verts

    # Compute bounding box for scaling the marker size
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    bbox_size = max(maxx - minx, maxy - miny)
    marker_radius = bbox_size * 0.01  # Scale marker size based on mesh size

    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()

    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):
        # Set object mesh color (pink)
        rgba_color = np.concatenate([c2rgba(colors['pink'])[:3], [0.5]])  # RGB + Alpha
        obj_mesh_color = np.tile(rgba_color, (obj_mesh_rec.shape[1], 1))
        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        # Set body mesh color (yellow pale)
        rgba_color_2 = np.concatenate([c2rgba(colors['yellow_pale'])[:3], [1]])  # RGB + Alpha
        mesh_color = np.tile(rgba_color_2, (mesh_rec.shape[1], 1))
        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)

        # Initialize mesh list
        all_meshes = [obj_m_rec, m_rec]

        # Add the highlight sphere if conditions are met
        if highlight_frame is not None and highlight_vertex is not None:
            if i >= highlight_frame - 15 and i <= highlight_frame+15:
                print(f"Highlighting frame {highlight_frame}, vertex {highlight_vertex}")
                # Create a small red sphere at the vertex position
                vertex_pos = mesh_rec[i, highlight_vertex]
                marker_sphere = trimesh.creation.icosphere(subdivisions=3, radius=marker_radius)
                marker_sphere.apply_translation(vertex_pos)
                marker_sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]  # Red sphere

                # Append the marker sphere to the mesh list
                all_meshes.append(marker_sphere)

        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                Ry = trimesh.transformations.rotation_matrix(np.radians(270), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = [obj_m_rec, m_rec]
                if highlight_frame is not None and highlight_vertex is not None:
                    if i >= highlight_frame - 15 and i <= highlight_frame+15:
                        vertex_pos = mesh_rec[i, highlight_vertex]
                        marker_sphere = trimesh.creation.icosphere(subdivisions=3, radius=marker_radius)
                        marker_sphere.apply_translation(vertex_pos)
                        marker_sphere.apply_transform(Ry)
                        marker_sphere.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
                        all_meshes.append(marker_sphere)
                mv.set_meshes(all_meshes, group_name='static')
                video_views.append(mv.render())
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text, fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv


def points_to_spheres(points, radius=0.01):
    spheres = []
    for p in points:
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(p)
        spheres.append(sphere)
    return trimesh.util.concatenate(spheres)

def visualize_points_obj(m_pcd, obj_verts, obj_face, save_path,
                       multi_angle=False, h=256, w=256, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(m_pcd)

    # Convert tensors to numpy if needed
    if torch.is_tensor(m_pcd):
        mesh_rec = m_pcd.cpu().numpy()
    else:
        mesh_rec = m_pcd
        
    if torch.is_tensor(obj_verts):
        obj_mesh_rec = obj_verts.cpu().numpy()
    else:
        obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        # m_rec = trimesh.points.PointCloud(mesh_rec[i])
        m_rec = points_to_spheres(mesh_rec[i])

        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv
