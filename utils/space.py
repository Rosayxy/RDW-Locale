import numpy as np
import math
from tqdm import tqdm, trange
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
HUGE_VALUE = 2000
class UserInfo:
    """
    The representation of the user (position, angle, velocity and angular velocity).
    """
    def __init__(self, x, y, angle, v, w):
        self.x = x
        self.y = y
        self.angle = angle
        self.v = v
        self.w = w

class Space:
    """
    The representation of the physical space.
    We consider it as a polygon space with polygonal obstacles obstacle_list.
    Additionally, we record the position of the user (user_x, user_y), its angle (user_angle) and velocity (user_v) and angular velocity (user_w).
    """
    def __init__(self, border, raw_obstacle_list):
        self.border= [(t['x'],t['y']) for t in border]
        self.obstacle_list = []
        for raw_obstacle in raw_obstacle_list:
            obstacle = [(t['x'],t['y']) for t in raw_obstacle]
            self.add_obstacle(obstacle)
        
    def add_obstacle(self, obstacle):
        self.obstacle_list.append(obstacle)

    def in_obstacle(self, x, y):
        if not Polygon(self.border).contains(Point(x,y)):
            return True
        for obstacle in self.obstacle_list:
            polygon = Polygon(obstacle)
            if polygon.contains(Point(x,y)):
                return True
        return False
    
    def get_center(self):
        # 对 boarder 中顶点求平均值
        x = 0
        y = 0
        for point in self.border:
            x += point[0]
            y += point[1]
        x /= 4
        y /= 4
        return x,y

    def get_dist(self,point, rect_vertices):
        px, py = point
        rect = np.array(rect_vertices)
        
        # Separate rectangle coordinates
        min_x = np.min(rect[:, 0])  # Smallest x value
        max_x = np.max(rect[:, 0])  # Largest x value
        min_y = np.min(rect[:, 1])  # Smallest y value
        max_y = np.max(rect[:, 1])  # Largest y value

        # Clamp the point's coordinates to the rectangle's bounds
        closest_x = np.clip(px, min_x, max_x)
        closest_y = np.clip(py, min_y, max_y)

        # Calculate distance
        closest_point = (closest_x, closest_y)

        return (closest_point[0] - px, closest_point[1] - py)
    
    def get_boarder_dist(self,point,border_end1,border_end2):
        x1, y1 = border_end1
        x2, y2 = border_end2
        x0, y0 = point
        if x1 == x2:
            return abs(x1-x0)
        if y1 == y2:
            return abs(y1-y0)
        k = (y2-y1)/(x2-x1)
        b = y1 - k*x1
        return abs(k*x0-y0+b)/math.sqrt(k**2+1)
    
    def get_apf_val(self,point):
        if self.in_obstacle(point[0], point[1]): # 处理 centered finite difference 中的边界情况
            return HUGE_VALUE
        val = 0
        for i in range(len(self.obstacle_list)):
            x, y = self.get_dist(point, self.obstacle_list[i])
            length = math.sqrt(x**2 + y**2)+0.05 # 防止除0
            val += 1/length
        # todo add border
        for i in range(4):
            point1 = self.border[i]
            point2 = self.border[(i+1)%4]
            val += 1/(self.get_boarder_dist(point,point1,point2)+0.05)
            
        val += 0.5*math.sqrt((point[0]-self.get_center()[0])**2 + (point[1]-self.get_center()[1])**2)
        return val
            