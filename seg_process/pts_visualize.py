import pandas as pd
import numpy as np 
import math
from pyntcloud import PyntCloud
import cv2
from utils import paint_points, cal_corner_after_rotation, eight_points, create_lines

COLOR_LIST = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255],'light-blue':[155,187,247],
         'black':[0, 0, 0], 'white': [255, 255, 255], 'aque':[0, 255, 255],
         'yellow': [255, 255, 0], 'orange': [255, 125, 0], 'grey':[125, 125, 125],
         'gray': [125, 125, 125]
}

class PointCloudPlot:
    def __init__(self):
        self.points = None
        self.dim = None
        self.scene = None
        self.boxes = []
    
    def load_points(self, points):
        self.points = points
    
    def load_points_from_file(self, filename, dim):
        self.dim = dim
        # file_client = mmcv.FileClient(backend='disk')
        # pts_bytes = file_client.get(filename)
        points = np.fromfile(filename, dtype=np.float32)
        points = points.reshape(-1, dim)
        self.points = points
    
    def plot(self, initial_point_size=0.02):
        assert self.scene is not None
        self.scene.plot(initial_point_size=initial_point_size, backend="pythreejs", polylines=self.boxes)
    
    def init_scene(self, point_color='light-blue', with_rgb=False):
        assert self.points is not None
        color = COLOR_LIST[point_color]
        if with_rgb:
            pc_plot = self.points[:, :3]
            color = self.points[:, -3:]
        else:
            pc_plot = self.points[:, :3]
        pc_plot = pd.DataFrame(paint_points(pc_plot, color=color), columns=['x', 'y', 'z', 'red', 'green', 'blue'])
        self.scene = PyntCloud(pc_plot)
    

    def add_bboxes(self, boxes, color='blue'):
        """
        boxes: List[List] [cx, cy, cz, depth, width, height, rotation]
        """
        pc_cornors = []
        for bbox in boxes:
            center = bbox[:3]
            size = bbox[3:6]
            try:
                rotation = bbox[6]
            except:
                rotation = 0
            eight_corners = eight_points(center, size, rotation)
            pc_cornors.append(eight_corners)
        for corners in pc_cornors:
            self.boxes.extend(create_lines(cornors=corners.tolist(), color=color))

    def add_one_box(self, box, color='yellow'):
        self.add_bboxes([box], color)
    
    
        