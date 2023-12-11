import numpy as np
remap = {
    23:8,
    8:0,
    19:18,
    26:19,
    25:17,
    22:17,
    17:17,
    24:17,
    27:20,
    7:4,
    21:7,
}
dir_replace = {
    "09-23-1":"09-23-13-44-53-1",
    "09-23-2":"09-23-13-44-53-2",
    "09-23-3":"09-23-13-44-53-3",
    "09-23-4":"09-23-13-44-53-4"
}
ACTION_MAP = {
    "WALKING_STANDING": 0,
    "CROUCHING_SITTING": 1,
    "PULL_PUSH": 2,
    "LIFT": 3,
    "ESCALATOR": 4,
    "CARRY": 5,
    "BASKETBALL": 6,
    "HUMAN_INTERACTION": 7,
    "BADMINTON": 8,
    "ENTERTAINMENT": 9,
    "RIDE_SIT": 10,
    "RIDE_STAND": 11,
    "FITNESS": 12,
    "MOVE": 13,
    "BENDING_OVER": 14,
    "RUNNING": 15,
    "OTHER": 16,
    
}
class_map = {
    "eraser": "small thing",
    "phone": "small thing",
    "cup":"small thing",
    "food":"small thing",
    "cellphone":"small thing",
    "red flag":"small thing",
    "cap": "small thing",
    "camera": "small thing",
    "sponge": "small thing",
    "projector": "small thing", # 投影仪
    
    "plush toy": "small thing", # 毛绒玩具
    "toy wings": "small thing", # 玩具

    
    "clothes": "small thing",
    "flower":"small thing",

    "cart":"cart",
    "flat car":"cart",
    "stroller":"cart",
    "perambulator": "cart",# 摇篮车

    "two-wheeled balancing car":"small vehicle",
    "balance car": "small vehicle",
    "scooter": "small vehicle",
    "two-wheels self-balancing scooter": "small vehicle",

    "bicycle": "electrocar",

    "toy car": "other", # together 286 
    "merry go round" :"other",

    "rockery": "children's slide",

    "car": "other",
    "tricycle": "other",
    "umbrella": "other",
    "banner": "other",
    "plank": "other",
    # "ground": "other",
    "paper": "other",

    "door":"other",

    "dog": "other",
    "megaphone": "other", # 扩音器？传声筒
    
    "printer": "cabinet",
    "podium": "cabinet", # 讲台

    # "badminton rocket": "badminton Rocket",


    "badminton rocket": "small thing",

    "guitar": "other",
    "handbag": "small thing",
    "suitcase": "box",
    "plastic bag": "small thing",
    "stool": "chair",
    "badminton Rocket": "small thing",
    "balloon": "small thing"
}


CLASS_COLOR = {
    "electrocar": [160,240,200], # 粉绿
    "toy car": [195,19,246],
    "cabinet": [84,155,65],
    "small thing": [249,68,182], # light green
    "box": [200,200,100],  # grass green
    "baby": [160,30,240], # dark purple
    "plastic bag": [255,255,30], # 黄
    "stool": [60,100,120], #  灰蓝
    "other": [255,255,255],
    "basketball": [255,0,0], # red
    "handbag": [255,165,0], # orange
    "spring car": [ 255,140,105], # rou
    "blackboard": [139,134,130], #深灰
    "balloon": [73,143,231], # 靛蓝
    "person": [0,0,255],
    "guitar": [125,125,30], # 黄
    "badminton Rocket": [185,77,36],
    "fitness equipment": [60,15,18],
    "children's slide": [95,49,39], 
    "seesaw": [135,206,255], # skyblue
    "table": [255,62,150], # pink
    "suitcase": [30,75,20], # 墨绿
    "computer": [190,190,10], # 土黄
    "cart": [90,20,125], # 深紫色
    "chair": [150,75,60],  # 砖红色
    "small vehicle": [255,160,200], # 肉粉
    "backpack": [255,100,100], # IndianRed1
    "ground": [125,125,125], # IndianRed1
    "staircase": [195,19,246]
}
SEMANTIC_NAMES = np.array(['other', 'person', 'electrocar', 'table', 'box', 'cart', 'seesaw', 'basketball', 'fitness equipment', 'cabinet', 'baby', 'blackboard', 'staircase', "children's slide", 'small vehicle', 'computer', 'backpack', 'small thing', 'chair', 'spring car', 'ground'
    # 'other','person', 'electrocar', 'table', 'box', \
    # 'cart', 'seesaw', 'suitcase', 'guitar', 'cabinet', 'baby', 'blackboard', 'toy car',\
    #     "children's slide", 'small vehicle', 'computer', 'backpack', 'handbag', 'chair', \
    #         'stool', 'balloon', 'basketball', 'small thing', 'fitness equipment', 'plastic bag', \
    #             'badminton Rocket', 'spring car','ground'
                ])

SEMANTIC_IDX2NAME = {
    0: "other",
    1: "person",
    2: "electrocar",
    3: "table",
    4: "box",
    5: "cart",
    6: "seesaw",
    7: "suitcase",
    8: "guitar",
    9: "cabinet",
    10: "baby",
    11: "blackboard",
    12: "toy car",
    13: "children's slide",
    14: "small vehicle",
    15: "computer",
    16: "backpack",
    17: "handbag",
    18: "chair",
    19: "stool",
    20: "balloon",
    21: "basketball",
    22: "small thing",
    23: "fitness equipment",
    24: "plastic bag",
    25: "badminton Rocket",
    26: "spring car",
    27: "ground"
}

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255