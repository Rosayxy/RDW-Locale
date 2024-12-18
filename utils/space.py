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
    def __init__(self, x, y, angle, v, w,vir_x,vir_y,vir_angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.v = v
        self.w = w
        self.vir_x = vir_x
        self.vir_y = vir_y
        self.vir_angle = vir_angle

class Space:
    """
    The representation of the physical space.
    We consider it as a polygon space with polygonal obstacles obstacle_list.
    Additionally, we record the position of the user (user_x, user_y), its angle (user_angle) and velocity (user_v) and angular velocity (user_w).
    """
    def __init__(self, border, raw_obstacle_list,virtual_border,virtual_obstacle_list):
        self.border= [(t['x'],t['y']) for t in border]
        self.obstacle_list = []
        self.virtual_border = [(t['x'],t['y']) for t in virtual_border]
        self.virtual_obstacle_list = []
        for raw_obstacle in raw_obstacle_list:
            obstacle = [(t['x'],t['y']) for t in raw_obstacle]
            self.add_obstacle(obstacle)
        for raw_obstacle in virtual_obstacle_list:
            obstacle = [(t['x'],t['y']) for t in raw_obstacle]
            self.add_virtual_obstacle(obstacle)
        
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
    
    def in_virtual_obstacle(self, x, y):
        if not Polygon(self.virtual_border).contains(Point(x,y)):
            return True
        for obstacle in self.virtual_obstacle_list:
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
    
    def get_dist_arc(self, point, angle, is_virtual=False):
        # Choose the appropriate border and obstacle list based on is_virtual
        if is_virtual:
            border = self.virtual_border
            obstacle_list = self.virtual_obstacle_list
        else:
            border = self.border
            obstacle_list = self.obstacle_list

        # Direction vector
        direction = np.array([math.cos(angle), math.sin(angle)])
        point = np.array(point)

        # Initialize minimum distance
        min_dist = HUGE_VALUE

        # Check distance to borders
        for i in range(len(border)):
            p1 = np.array(border[i])
            p2 = np.array(border[(i + 1) % len(border)])
            intersection = self._ray_segment_intersection(point, direction, p1, p2)
            if intersection is not None:
                dist = np.linalg.norm(intersection - point)
                min_dist = min(min_dist, dist)

        # Check distance to obstacles
        for obstacle in obstacle_list:
            for i in range(len(obstacle)):
                p1 = np.array(obstacle[i])
                p2 = np.array(obstacle[(i + 1) % len(obstacle)])
                intersection = self._ray_segment_intersection(point, direction, p1, p2)
                if intersection is not None:
                    dist = np.linalg.norm(intersection - point)
                    min_dist = min(min_dist, dist)
        return min_dist

    def _ray_segment_intersection(self, ray_origin, ray_direction, segment_start, segment_end):
        """
        Calculates the intersection point between a ray and a line segment.

        :param ray_origin: The starting point of the ray (numpy array).
        :param ray_direction: The direction vector of the ray (numpy array).
        :param segment_start: The starting point of the segment (numpy array).
        :param segment_end: The ending point of the segment (numpy array).
        :return: The intersection point as a numpy array, or None if there's no intersection.
        """
        # Segment vector
        segment_vector = segment_end - segment_start

        # Ray and segment cross products
        r_cross_s = np.cross(ray_direction, segment_vector)
        
        # If r_cross_s is zero, the ray and the segment are parallel or collinear
        if np.isclose(r_cross_s, 0):
            return None

        # Find the t and u parameters
        diff = segment_start - ray_origin
        t = np.cross(diff, segment_vector) / r_cross_s
        u = np.cross(diff, ray_direction) / r_cross_s
        # Check if t and u are valid
        if t < 0 or u < 0 or u > 1:
            return None

        # Calculate the intersection point
        intersection_point = ray_origin + t * ray_direction
        return intersection_point


    def get_a_qt(self,physical_point,physical_angle,virtual_point,virtual_angle):
        phy_dist_1 = self.get_dist_arc(physical_point, physical_angle)
        phy_dist_2 = self.get_dist_arc(physical_point, physical_angle + math.pi/2)
        phy_dist_3 = self.get_dist_arc(physical_point, physical_angle - math.pi/2)
        
        virt_dist_1 = self.get_dist_arc(virtual_point, virtual_angle,True)
        virt_dist_2 = self.get_dist_arc(virtual_point, virtual_angle + math.pi/2,True)
        virt_dist_3 = self.get_dist_arc(virtual_point, virtual_angle - math.pi/2,True)
        return abs(phy_dist_1-virt_dist_1)+abs(phy_dist_2-virt_dist_2)+abs(phy_dist_3-virt_dist_3)
    
    def get_misalign(self,physical_point,physical_angle,virtual_point,virtual_angle):
        # left misalign
        left_phy_dist = self.get_dist_arc(physical_point, physical_angle - math.pi/2)
        left_virt_dist = self.get_dist_arc(virtual_point, virtual_angle - math.pi/2,True)
        left_misalign = (left_phy_dist - left_virt_dist)/left_virt_dist
        # right misalign
        right_phy_dist = self.get_dist_arc(physical_point, physical_angle + math.pi/2)
        right_virt_dist = self.get_dist_arc(virtual_point, virtual_angle + math.pi/2,True)
        right_misalign = (right_phy_dist - right_virt_dist)/right_virt_dist
        return left_misalign, right_misalign
    

    def get_normal_at_rst(self, physical_point):
        """
        Find the obstacle or border that the user is colliding with at reset,
        and calculate the normal vector pointing away from the obstacle or border.
        """
        point = np.array(physical_point)

        # Check for collision with borders
        for i in range(len(self.border)):
            p1 = np.array(self.border[i])
            p2 = np.array(self.border[(i + 1) % len(self.border)])
            if self._is_point_near_segment(point, p1, p2):
                # Calculate the normal vector of the segment
                segment_dir = p2 - p1
                normal = np.array([-segment_dir[1], segment_dir[0]])  # Perpendicular vector
                normal = normal / np.linalg.norm(normal)  # Normalize the vector
                
                # Ensure the normal points outward (away from the polygon)
                to_point = point - p1
                if np.dot(normal, to_point) < 0:  # If the normal points inward
                    normal = -normal
                return normal

        # Check for collision with obstacles
        for obstacle in self.obstacle_list:
            for i in range(len(obstacle)):
                p1 = np.array(obstacle[i])
                p2 = np.array(obstacle[(i + 1) % len(obstacle)])
                if self._is_point_near_segment(point, p1, p2):
                    # Calculate the normal vector of the segment
                    segment_dir = p2 - p1
                    normal = np.array([-segment_dir[1], segment_dir[0]])  # Perpendicular vector
                    normal = normal / np.linalg.norm(normal)  # Normalize the vector
                    
                    # Ensure the normal points outward (away from the obstacle)
                    to_point = point - p1
                    if np.dot(normal, to_point) < 0:  # If the normal points inward
                        normal = -normal
                    return normal

        # If no collision is detected, return None
        return None

    def _is_point_near_segment(self, point, segment_start, segment_end, threshold=2):
        """
        Check if a point is near a line segment within a specified threshold distance.
        """
        # Calculate the distance from the point to the line segment
        distance = self.get_boarder_dist(point, segment_start, segment_end)
        return distance <= threshold