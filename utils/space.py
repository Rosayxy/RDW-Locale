import numpy as np
import math
from tqdm import tqdm, trange
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# class RDWFramework:
#     def __init__(self, physical_width, physical_height, user_x, user_y, user_direction):
#         self.physical_space = PhysicalSpace(physical_width, physical_height, user_x, user_y, user_direction)
#         self.virtual_space = VirtualSpace()
#         self.reset_num = 0

class UserInfo:
    """
    The representation of the user (position, direction, velocity and angular velocity).
    """
    def __init__(self, x, y, direction, v, w):
        self.x = x
        self.y = y
        self.direction = direction
        self.v = v
        self.w = w

class Space:
    """
    The representation of the physical space.
    We consider it as a rectangle from (0,0) to (width, height) with polygonal obstacles obstacle_list.
    Additionally, we record the position of the user (user_x, user_y), its direction (user_direction) and velocity (user_v) and angular velocity (user_w).
    """
    def __init__(self, width, height, raw_obstacle_list):
        self.width = width
        self.height = height
        self.obstacle_list = []
        for raw_obstacle in raw_obstacle_list:
            obstacle = [(t['x'],t['y']) for t in raw_obstacle]
            self.add_obstacle(obstacle)
        
    def add_obstacle(self, obstacle):
        self.obstacle_list.append(obstacle)

    def in_obstacle(self, x, y):
        if x <= 0 or x >= self.width or y <= 0 or y >= self.height:
            return True
        for obstacle in self.obstacle_list:
            polygon = Polygon(obstacle)
            if polygon.contains(Point(x,y)):
                return True
        return False