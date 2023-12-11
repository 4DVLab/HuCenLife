import numpy as np
import math
import cv2

def find_min_gt_index(gt_bbox, pred_bbox):
    dist_list = [1000] * len(pred_bbox)
    min_dist, min_idx = 1000, -1
    for i, pbox in enumerate(pred_bbox):
        dist_list[i] = cal_distance(gt_bbox, pbox)
        if dist_list[i] < min_dist:
            min_dist = dist_list[i]
            min_idx = i
    return min_dist, min_idx

def find_max_gt_index(gt_bbox, pred_bbox):
    iou_list = [0] * len(pred_bbox)
    max_iou, max_idx = 0, -1
    for i, pbox in enumerate(pred_bbox):
        iou_list[i] = cal_iou3d(gt_bbox, pbox)
        if iou_list[i] > max_iou:
            max_iou = iou_list[i]
            max_idx = i
    return max_iou, max_idx

def paint_points(points, color=[192,0,0]):
    color = np.array([color])
    new_pts = np.zeros([points.shape[0],6])
    new_pts[:,:3] = points
    new_pts[:, 3:] = new_pts[:, 3:] + color
    return new_pts.astype(np.float32)

def cal_corner_after_rotation(corner, center, r):
    x1, y1 = corner
    x0, y0 = center
    x2 = math.cos(r) * (x1 - x0) - math.sin(r) * (y1 - y0) + x0
    y2 = math.sin(r) * (x1 - x0) + math.cos(r) * (y1 - y0) + y0
    return x2, y2

def eight_points(center, size, rotation=0):
    x, y, z = center
    w, l, h = size
    w = w/2
    l = l/2
    h = h/2

    x1, y1, z1 = x-w, y-l, z+h
    x2, y2, z2 = x+w, y-l, z+h
    x3, y3, z3 = x+w, y-l, z-h
    x4, y4, z4 = x-w, y-l, z-h
    x5, y5, z5 = x-w, y+l, z+h
    x6, y6, z6 = x+w, y+l, z+h
    x7, y7, z7 = x+w, y+l, z-h
    x8, y8, z8 = x-w, y+l, z-h

    if rotation != 0:
        x1, y1 = cal_corner_after_rotation(corner=(x1, y1), center=(x, y), r=rotation)
        x2, y2 = cal_corner_after_rotation(corner=(x2, y2), center=(x, y), r=rotation)
        x3, y3 = cal_corner_after_rotation(corner=(x3, y3), center=(x, y), r=rotation)
        x4, y4 = cal_corner_after_rotation(corner=(x4, y4), center=(x, y), r=rotation)
        x5, y5 = cal_corner_after_rotation(corner=(x5, y5), center=(x, y), r=rotation)
        x6, y6 = cal_corner_after_rotation(corner=(x6, y6), center=(x, y), r=rotation)
        x7, y7 = cal_corner_after_rotation(corner=(x7, y7), center=(x, y), r=rotation)
        x8, y8 = cal_corner_after_rotation(corner=(x8, y8), center=(x, y), r=rotation)

    conern1 = np.array([x1, y1, z1])
    conern2 = np.array([x2, y2, z2])
    conern3 = np.array([x3, y3, z3])
    conern4 = np.array([x4, y4, z4])
    conern5 = np.array([x5, y5, z5])
    conern6 = np.array([x6, y6, z6])
    conern7 = np.array([x7, y7, z7])
    conern8 = np.array([x8, y8, z8])
    
    eight_corners = np.stack([conern1, conern2, conern3, conern4, conern5, conern6, conern7, conern8], axis=0)
    return eight_corners

def create_lines(cornors, color="yellow"):
    ind = [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4]]
    vertices_list = []
    for i in range(4):
        vertices_list.append([cornors[ind[0][i]], cornors[ind[0][i + 1]]])
        vertices_list.append([cornors[ind[1][i]], cornors[ind[1][i + 1]]])
        vertices_list.append([cornors[ind[0][i]], cornors[ind[1][i]]])
    return [{"color": color, "vertices": vertices_list[i]} for i in range(len(vertices_list))] 

from shapely.geometry import Polygon

def cal_corner_after_rotation(corner, center, r):
    x1, y1 = corner
    x0, y0 = center
    x2 = math.cos(r) * (x1 - x0) - math.sin(r) * (y1 - y0) + x0
    y2 = math.sin(r) * (x1 - x0) + math.cos(r) * (y1 - y0) + y0
    return x2, y2

def eight_points(center, size, rotation=0):
    x, y, z = center
    w, l, h = size
    w = w/2
    l = l/2
    h = h/2

    x1, y1, z1 = x-w, y-l, z+h
    x2, y2, z2 = x+w, y-l, z+h
    x3, y3, z3 = x+w, y-l, z-h
    x4, y4, z4 = x-w, y-l, z-h
    x5, y5, z5 = x-w, y+l, z+h
    x6, y6, z6 = x+w, y+l, z+h
    x7, y7, z7 = x+w, y+l, z-h
    x8, y8, z8 = x-w, y+l, z-h

    if rotation != 0:
        x1, y1 = cal_corner_after_rotation(corner=(x1, y1), center=(x, y), r=rotation)
        x2, y2 = cal_corner_after_rotation(corner=(x2, y2), center=(x, y), r=rotation)
        x3, y3 = cal_corner_after_rotation(corner=(x3, y3), center=(x, y), r=rotation)
        x4, y4 = cal_corner_after_rotation(corner=(x4, y4), center=(x, y), r=rotation)
        x5, y5 = cal_corner_after_rotation(corner=(x5, y5), center=(x, y), r=rotation)
        x6, y6 = cal_corner_after_rotation(corner=(x6, y6), center=(x, y), r=rotation)
        x7, y7 = cal_corner_after_rotation(corner=(x7, y7), center=(x, y), r=rotation)
        x8, y8 = cal_corner_after_rotation(corner=(x8, y8), center=(x, y), r=rotation)

    conern1 = np.array([x1, y1, z1])
    conern2 = np.array([x2, y2, z2])
    conern3 = np.array([x3, y3, z3])
    conern4 = np.array([x4, y4, z4])
    conern5 = np.array([x5, y5, z5])
    conern6 = np.array([x6, y6, z6])
    conern7 = np.array([x7, y7, z7])
    conern8 = np.array([x8, y8, z8])
    
    eight_corners = np.stack([conern1, conern2, conern6, conern5, conern4, conern3, conern7, conern8], axis=0)
    return eight_corners

def cal_inter_area(box1, box2):
    """
    box: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    a=np.array(box1).reshape(4, 2)   #四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 
    
    b=np.array(box2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    
    union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
    if not poly1.intersects(poly2): #如果两四边形不相交
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return poly1.area, poly2.area, inter_area

def cal_iou3d(box1, box2):
    """
    box: [x, y, z, w, h, l, r] center(x, y, z)
    """
    center1 = box1[:3]
    size1 = box1[3:6]
    rotation1 = box1[6]
    eight_corners1 = eight_points(center1, size1, rotation1)
    
    center2 = box2[:3]
    size2 = box2[3:6]
    rotation2 = box2[6]
    eight_corners2 = eight_points(center2, size2, rotation2)
    
    area1, area2, inter_area = cal_inter_area(eight_corners1[:4, :2].reshape(-1), eight_corners2[:4, :2].reshape(-1))
    
    h1 = box1[5]
    z1 = box1[2]
    h2 = box2[5]
    z2 = box2[2]
    volume1 = h1 * area1
    volume2 = h2 * area2
    
    bottom1, top1 = z1 - h1/2, z1 + h1/2
    bottom2, top2 = z2 - h2/2, z2 + h2/2
    
    inter_bottom = max(bottom1, bottom2)
    inter_top = min(top1, top2)
    inter_h = inter_top - inter_bottom if inter_top > inter_bottom else 0
    
    inter_volume = inter_area * inter_h
    union_volume = volume1 + volume2 - inter_volume
    
    iou = inter_volume / union_volume
    
    return iou

def cal_distance(box1, box2):
    """
    box: [x, y, z, w, h, l, r] center(x, y, z)
    """
    x1, y1, z1, w1, l1, h1, r1 = box1
    x2, y2, z2, w2, l2, h2, r2 = box2
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return dist

def loc_pc2img(points):
    x, y, z, w, l, h, r = points
    center = [x, y, z]
    size = [w, l, h]
    points = eight_points(center, size, r)
    points = np.insert(points, 3, values=1, axis=1)
    points_T = np.transpose(points)
    points_T[3, :] = 1.0

    ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
    in_matrix = np.array([[683.8, 0.0, 673.5907],
                         [0.0, 684.147, 372.8048],
                         [0.0, 0.0, 1.0]])
    # lidar2camera
    points_T_camera = np.dot(ex_matrix, points_T)
    # camera2pixel
    pixel = np.dot(in_matrix, points_T_camera).T
    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
    pixel_xy = np.around(pixel_xy).astype(int)
    
    return pixel_xy.T

def point_cloud_position2image_position(points):
    corner_2d = []
    for p in points:
        p = p[:7]
        new_p = loc_pc2img(p)
        corner_2d.append(new_p)
    corner_2d = np.stack(corner_2d, axis=0)
    return corner_2d

def draw_3dbox_in_image_per_corner(image, corner_2d, color='b', thickness=2):
    """
    image: np.Array
    corner_2d: [2, 8]
    """
    COLOR = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255)}
    color = COLOR[color]
    for corner_i in range(0, 4):
        i, j = corner_i, (corner_i + 1) % 4
        cv2.line(image, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
        i, j = corner_i + 4, (corner_i + 1) % 4 + 4
        cv2.line(image, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
        i, j = corner_i, corner_i + 4
        cv2.line(image, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color, thickness)
    return image

def draw_3dbox_in_image(image, corner_2d, color='b', thickness=2):
    for corner in corner_2d:
        image = draw_3dbox_in_image_per_corner(image, corner, color, thickness)
    return image

def crop_pc_utils(pointcloud,data,i):
    crop_range = [
        data[i][0]-data[i][3]/2,
        data[i][0]+data[i][3]/2,
        data[i][1]-data[i][4]/2,
        data[i][1]+data[i][4]/2,
        data[i][2]-data[i][5]/2,
        data[i][2]+data[i][5]/2
        ]
    crop_ind = np.where(
        (pointcloud[:,0] > crop_range[0]) & (pointcloud[:,0] < crop_range[1]) &
        ( pointcloud[:,1]>crop_range[2]) & ( pointcloud[:,1]<crop_range[3]) &
        (pointcloud[:,2] > crop_range[4]) & (pointcloud[:,2] < crop_range[5]) 
    )
    return crop_ind