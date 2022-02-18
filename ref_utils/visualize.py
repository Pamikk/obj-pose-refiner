from open3d import *
import numpy as np
import os
"""
write or visualize point clouds using open3d
"""
def write_3d_open3d(path,name,pcd):
    io.write_point_cloud(os.path.join(path,name), pcd, write_ascii=False, compressed=False, print_progress=False)
def gen_pt_cloud(pt_clouds,cols=None):
    if cols!=None:
        assert len(pt_clouds)==len(cols)
    points = utility.Vector3dVector()
    colors = utility.Vector3dVector()
    for pts in pt_clouds:
        pts = pts.squeeze()
        pt_num = len(pts)
        if cols == None:
            cols = np.random.rand(1,3)
            cols = np.repeat(cols,pt_num,axis=0)
        points.extend(pts)
        colors.extend(cols)
    pcd = geometry.PointCloud()
    pcd.points = points
    pcd.colors = colors
    return pcd