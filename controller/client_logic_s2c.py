from utils.constants import *
from utils.space import *
import math
import numpy as np
from shapely.geometry import Polygon, Point

def calc_gain(user: UserInfo, physical_space: Space, delta: float, improved_s2c=False):
    """
    Calculates gain values based on the Steer-to-Center (S2C) algorithm.
    Uses the formula C = k * sin(theta) for curvature gain.
    Includes an improved version of S2C when improved_s2c is True.
    
    Returns:
        trans_gain (float): Translation gain.  (Not used in S2C)
        rot_gain (float): Rotation gain. (Not used in S2C)
        curvature_radius (float): Curvature gain radius.
        direction (int): Direction for curvature gain (+1 or -1).
    """
    # Get the center point of the physical space
    border_polygon = Polygon(physical_space.border)
    center_point = border_polygon.centroid

    # User's current position and heading
    user_pos = np.array([user.x, user.y])
    user_heading = np.array([math.cos(user.angle), math.sin(user.angle)])

    # Default steering point is the center of the physical space
    steering_point = np.array([center_point.x, center_point.y])

    # Vector from user position to steering point
    to_steering_vec = steering_point - user_pos
    distance_to_steering = np.linalg.norm(to_steering_vec)

    # Calculate angle between user's heading and vector to steering point
    if distance_to_steering == 0:
        angle_to_steering = 0.0
        return 1.0, 1.0, INF_CUR_GAIN_R, 1
    else:
        to_steering_dir = to_steering_vec / distance_to_steering
        dot_product = np.dot(user_heading, to_steering_dir)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_to_steering = math.acos(dot_product)  # Angle in radians

    # Improved S2C algorithm: create a temporary steering point when angle exceeds 160 degrees
    if improved_s2c and angle_to_steering > math.radians(160):
        perp_direction_left = np.array([-user_heading[1], user_heading[0]]) 
        perp_direction_right = np.array([user_heading[1], -user_heading[0]]) 

        
        temp_steering_distance = 10

        temp_point_left = user_pos + perp_direction_left * temp_steering_distance
        temp_point_right = user_pos + perp_direction_right * temp_steering_distance


        point_left_inside = border_polygon.contains(Point(temp_point_left))
        point_right_inside = border_polygon.contains(Point(temp_point_right))

        if point_left_inside and not point_right_inside:
            steering_point = temp_point_left
        elif point_right_inside and not point_left_inside:
            steering_point = temp_point_right
        elif point_left_inside and point_right_inside:
            # Both points are inside; choose the one closer to the center
            dist_left_to_center = np.linalg.norm(temp_point_left - np.array([center_point.x, center_point.y]))
            dist_right_to_center = np.linalg.norm(temp_point_right - np.array([center_point.x, center_point.y]))
            steering_point = temp_point_left if dist_left_to_center < dist_right_to_center else temp_point_right
        else:
            # Neither point is inside; keep the original steering point (center)
            steering_point = np.array([center_point.x, center_point.y])

        # Update vector and angle calculations with the new steering point
        to_steering_vec = steering_point - user_pos
        distance_to_steering = np.linalg.norm(to_steering_vec)
        if distance_to_steering == 0:
            angle_to_steering = 0.0
            return 1.0, 1.0, INF_CUR_GAIN_R, 1
        else:
            to_steering_dir = to_steering_vec / distance_to_steering
            dot_product = np.dot(user_heading, to_steering_dir)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_to_steering = math.acos(dot_product)

    # Calculate curvature radius using the formula C = 900 - k * sin(theta)
    k = 750 - MIN_CUR_GAIN_R
    sin_theta = math.sin(angle_to_steering)
    curvature_radius = 750 - k * sin_theta
    curvature_radius = np.clip(curvature_radius, MIN_CUR_GAIN_R, 750)

    # Determine direction (+1 for clockwise, -1 for counter-clockwise)
    cross_product = np.cross(user_heading, to_steering_dir)
    direction = 1 if cross_product > 0 else -1

    # Set translation and rotation gains
    trans_gain = 1.0  # No adjustment to translation speed
    rot_gain = 1.0    # No adjustment to rotation speed

    return trans_gain, rot_gain, curvature_radius, direction

def update_reset(user: UserInfo, physical_space: Space, delta: float):
    """
    Reset strategy when necessary.
    Rotates the user's orientation by 180 degrees.
    
    Returns:
        user (UserInfo): Updated user information after reset.
    """
    user.angle = (user.angle + math.pi) % (2 * math.pi)
    return user
