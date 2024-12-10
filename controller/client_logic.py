from utils.constants import *
from utils.space import *
import math
# todo 思考怎么对待那个 temporary target
# define target globally
target_x = None
target_y = None
method = "arc"
STEP_SIZE = 1
STEP_NUM = 1
prev_frame_UserInfo = None
# todo refactor calc_gain to switch case
def calc_gain(user : UserInfo, physical_space : Space, delta : float):
    if method == "s2c":
        return calc_gain_s2c(user, physical_space, delta)
    elif method == "s2mt":
        return calc_gain_s2mt(user, physical_space, delta)
    elif method == "apf":
        return calc_gain_apf(user, physical_space, delta)
    elif method == "arc":
        return calc_gain_arc(user, physical_space, delta)

def update_reset(user : UserInfo, physical_space : Space, delta : float):
    if method == "s2c":
        return update_reset_base(user, physical_space, delta)
    elif method == "s2mt":
        return update_reset_base(user, physical_space, delta)
    elif method == "apf":
        return update_reset_SFR2G(user, physical_space, delta)
    elif method == "arc":
        return update_reset_arc(user, physical_space, delta)
    
def calc_gain_s2c(user : UserInfo, physical_space : Space, delta : float):
    """
    Return three gains (平移增益、旋转增益、曲率增益半径) and the direction (+1 (clockwise) or -1 (counter clockwise)) when cur_gain used. Implement your own logic here.
    所有的增益都是 虚拟/物理空间
    """
    # what is a rotation gain seems that I still don't understand....
    center_x, center_y = physical_space.get_center()
    angle = math.atan2(center_y - user.y, center_x - user.x)
    angle_diff = angle - user.angle
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff)) # normalization
    # judge whether change target or not
    if abs(angle_diff)>math.pi*8/9:
        # generate new target 90 degrees from the current user angle and 4 meters away from user, with the direction close to center
        target_x = user.x + 400 * math.cos(user.angle + math.pi/2)
        target_y = user.y + 400 * math.sin(user.angle + math.pi/2)
        # judge whether the target makes the user closer to the center  
        if (target_x - user.x) * (-user.x + center_x) + (target_y - user.y) * (-user.y + center_y) < 0:
            # if not, change the direction
            target_x = user.x + 400 * math.cos(user.angle - math.pi/2)
            target_y = user.y + 400 * math.sin(user.angle - math.pi/2)
    else:
        target_x = center_x
        target_y = center_y
        
    dist = math.sqrt((target_x - user.x) ** 2 + (target_y - user.y) ** 2)
    direction = 1 if angle_diff > 0 else -1
    curvature_gain_radius = MIN_CUR_GAIN_R
    translation_gain = MAX_TRANS_GAIN
    # if rotating closer to the target, decrease the rotation gain else increase it
    rotation_gain = MAX_ROT_GAIN
    if user.w*direction < 0:
        rotation_gain = MIN_ROT_GAIN
    # calculate the rotation caused by curvature

    if dist < 125:
        curvature_gain_radius = MIN_CUR_GAIN_R*(2.5/(dist+1.25))
    return translation_gain, rotation_gain, curvature_gain_radius, direction       

def update_reset_base(user : UserInfo, physical_space : Space, delta : float):
    """
    Return new UserInfo when RESET. Implement your RESET logic here.
    """
    center_x, center_y = physical_space.get_center()
    # 转 180 度
    new_orientation = ( user.angle + math.pi ) % (2 * math.pi)
    user.angle = new_orientation
    return user

def calc_gain_s2mt(user : UserInfo, physical_space : Space, delta : float):
    """
    Return three gains (平移增益、旋转增益、曲率增益半径) and the direction (+1 (clockwise) or -1 (counter clockwise)) when cur_gain used. Implement your own logic here.
    """
    center_x, center_y = physical_space.get_center()
    radius = 500
    targets = [
        (center_x + radius, center_y),  # Target 1
        (center_x - radius / 2, center_y + math.sqrt(3) * radius / 2),  # Target 2
        (center_x - radius / 2, center_y - math.sqrt(3) * radius / 2),  # Target 3
    ]
    # find the target closest to being in front of the user
    closest_target = None
    smallest_bearing_difference = math.pi
    bearing_diff = None
    for target_x, target_y in targets:
        # Compute the angle to the target
        to_target_x = target_x - user.x
        to_target_y = target_y - user.y
        angle_to_target = math.atan2(to_target_y, to_target_x)

        # Calculate bearing difference to the user's current heading
        bearing_diff = angle_to_target - user.angle
        bearing_diff = math.atan2(math.sin(bearing_diff), math.cos(bearing_diff))  # Normalize to [-π, π]

        if abs(bearing_diff) < smallest_bearing_difference:
            smallest_bearing_difference = abs(bearing_diff)
            closest_target = (target_x, target_y)
    direction = 1 if bearing_diff > 0 else -1
    curvature_gain_radius = MIN_CUR_GAIN_R
    translation_gain = MAX_TRANS_GAIN
    rotation_gain = MAX_ROT_GAIN
    if user.w*direction < 0:
        rotation_gain = MIN_ROT_GAIN
    dist = math.sqrt((closest_target[0] - user.x) ** 2 + (closest_target[1] - user.y) ** 2)
    if dist < 125:
        curvature_gain_radius = MIN_CUR_GAIN_R*(2.5/(dist+1.25))
        
    return translation_gain, rotation_gain, curvature_gain_radius, direction 
    
def update_reset_s2mt(user : UserInfo, physical_space : Space, delta : float):
    """
    Return new UserInfo when RESET. Implement your RESET logic here.
    Turns towards the nearest target. This implementation has the drawback that the new direction might still be blocked by an obstacle.
    """
    center_x, center_y = physical_space.get_center()
    radius = 500
    targets = [
        (center_x + radius, center_y),  # Target 1
        (center_x - radius / 2, center_y + math.sqrt(3) * radius / 2),  # Target 2
        (center_x - radius / 2, center_y - math.sqrt(3) * radius / 2),  # Target 3
    ]
    # find the target
    closest_target = None
    smallest_bearing_difference = math.pi
    bearing_diff = None
    for target_x, target_y in targets:
        # Compute the angle to the target
        to_target_x = target_x - user.x
        to_target_y = target_y - user.y
        angle_to_target = math.atan2(to_target_y, to_target_x)

        # Calculate bearing difference to the user's current heading
        bearing_diff = angle_to_target - user.angle
        bearing_diff = math.atan2(math.sin(bearing_diff), math.cos(bearing_diff))  # Normalize to [-π, π]

        if abs(bearing_diff) < smallest_bearing_difference:
            smallest_bearing_difference = abs(bearing_diff)
            closest_target = (target_x, target_y)
            
    new_orientation = math.atan2(closest_target[1] - user.y, closest_target[0] - user.x)
    user.angle = new_orientation
    return user

def calc_gain_apf(user: UserInfo, physical_space: Space, delta: float):
    """
    Return three gains (平移增益、旋转增益、曲率增益半径) and the direction (+1 (clockwise) or -1 (counter clockwise)) when cur_gain used. Implement your own logic here.
    """
    val = physical_space.get_apf_val((user.x, user.y))
    # calculate the x and y component of the force using centered finite difference method 
    x_force = (-physical_space.get_apf_val((user.x + 0.1, user.y)) + physical_space.get_apf_val((user.x - 0.1, user.y))) / 0.2
    y_force = (-physical_space.get_apf_val((user.x, user.y + 0.1)) + physical_space.get_apf_val((user.x, user.y - 0.1))) / 0.2
    # calculate the angle of the force
    angle = math.atan2(y_force, x_force)
    curvature_gain_radius = MIN_CUR_GAIN_R
    # direction 看速度方向和受力方向的叉积
    user_x_v=user.v*math.cos(user.angle)
    user_y_v=user.v*math.sin(user.angle)
    if (user_x_v*y_force-user_y_v*x_force)<=0:
        direction = -1
    else:
        direction = 1
    # judge dot product to calculate the translation gain
    translation_gain = 1
    if user_x_v*x_force+user_y_v*y_force<0:
        translation_gain = MAX_TRANS_GAIN
    # judge rotation
    rotation_gain = MAX_ROT_GAIN
    if user.w*direction < 0:
        rotation_gain = MIN_ROT_GAIN
    return translation_gain, rotation_gain, curvature_gain_radius, direction

def update_reset_SFR2G(user : UserInfo, physical_space : Space, delta : float):
    """
    Return new UserInfo when RESET. Implement your RESET logic here.
    """
    for i in range(STEP_NUM):
        val = physical_space.get_apf_val((user.x, user.y))
        # calculate the x and y component of the force using centered finite difference method
        x_force = (- physical_space.get_apf_val((user.x + 2, user.y)) + physical_space.get_apf_val((user.x - 2, user.y))) / 4
        y_force = (-physical_space.get_apf_val((user.x, user.y + 2)) + physical_space.get_apf_val((user.x, user.y - 2))) / 4
        print("val",val,"x_force_1",physical_space.get_apf_val((user.x + 2, user.y)),"x_force_2",physical_space.get_apf_val((user.x-2,user.y)),"yforce 1",physical_space.get_apf_val((user.x,user.y+2)),"yforce 2",physical_space.get_apf_val((user.x,user.y - 2)),"y force",y_force)
        # calculate the angle of the force
        angle = math.atan2(y_force, x_force)
        # calculate the new position
        user.x += STEP_SIZE * math.cos(angle)
        user.y += STEP_SIZE * math.sin(angle)
        user.angle = angle
        print("in updating reset, user x is ", user.x, "user y is ", user.y,"user angle is",user.angle,"x force is ", x_force, "y force is ", y_force)
    return user
    
def calc_gain_arc(user: UserInfo,physical_space: Space, delta: float):
    """
    Return three gains (平移增益、旋转增益、曲率增益半径) and the direction (+1 (clockwise) or -1 (counter clockwise)) when cur_gain used. Implement your own logic here.
    """
    global prev_frame_UserInfo
    print("calc gain arc")
    print(physical_space.get_dist_arc((user.x,user.y),user.angle),physical_space.get_dist_arc((user.virtual_x,user.virtual_y),user.virtual_angle,True),physical_space.get_dist_arc((user.x,user.y),user.angle)/physical_space.get_dist_arc((user.virtual_x,user.virtual_y),user.virtual_angle,True))
    translation_gain = physical_space.get_dist_arc((user.virtual_x,user.virtual_y),user.virtual_angle,True)/physical_space.get_dist_arc((user.x,user.y),user.angle)
    if translation_gain>MAX_TRANS_GAIN:
        translation_gain = MAX_TRANS_GAIN
    if translation_gain < MIN_TRANS_GAIN:
        translation_gain = MIN_TRANS_GAIN
        
    a_qt = physical_space.get_a_qt((user.x,user.y),user.angle,(user.virtual_x,user.virtual_y),user.virtual_angle)
    
    if a_qt == 0:
        prev_frame_UserInfo = user
        return 1,1,INF_CUR_GAIN_R,1
    
    # radius gain
    misalign_left, misalign_right = physical_space.get_misalign((user.x,user.y),user.angle,(user.virtual_x,user.virtual_y),user.virtual_angle)
    direction = 1
    radius = INF_CUR_GAIN_R
    if misalign_left < misalign_right: # steer to the right, so it is clockwise
        direction = 1
        if misalign_right < 1:
            radius = min(MIN_CUR_GAIN_R/misalign_right+0.001,radius)
        else:
            radius = MIN_CUR_GAIN_R
    elif misalign_left > misalign_right: # steer to the left, so it is counter clockwise
        direction = -1
        if misalign_left < 1:
            radius = min(MIN_CUR_GAIN_R/misalign_left,radius)
        else:
            radius = MIN_CUR_GAIN_R
    
    # rotation gain
    rotation_gain = 1
    if prev_frame_UserInfo is None:
        prev_frame_UserInfo = user
    else:
        prev_a_qt = physical_space.get_a_qt((prev_frame_UserInfo.x,prev_frame_UserInfo.y),prev_frame_UserInfo.angle,(prev_frame_UserInfo.virtual_x,prev_frame_UserInfo.virtual_y),prev_frame_UserInfo.virtual_angle)
        if prev_a_qt > a_qt:
            rotation_gain = MIN_ROT_GAIN
        elif prev_a_qt < a_qt:
            rotation_gain = MAX_ROT_GAIN
        else:
            rotation_gain = 1
        prev_frame_UserInfo = user
    
    return translation_gain,rotation_gain,radius,direction
    
def update_reset_arc(user : UserInfo, physical_space : Space, delta : float):
    """
    Return new UserInfo when RESET. Implement your RESET logic here.
    """
    normal = physical_space.get_normal_at_rst((user.x,user.y)) # a 2D np array
    dist_virt = physical_space.get_dist_arc((user.virtual_x,user.virtual_y),user.virtual_angle,True)
    dist_phys_list = []
    print("dist virt at reset",dist_virt)
    for i in range(20):
        dist_phys = physical_space.get_dist_arc((user.x,user.y),user.angle+math.pi*i/10+math.pi,False)
        if normal[0]*math.cos(user.angle+math.pi*i/10+math.pi)+normal[1]*math.sin(user.angle+math.pi*i/10+math.pi) > 0:
            dist_phys_list.append((user.angle + math.pi + math.pi*i/10,dist_phys))
            if dist_phys > dist_virt:
                print("dist_phys",dist_phys,"dist_virt",dist_virt)
                user.angle = user.angle + math.pi + math.pi*i/10
                return user
    print("dist_phys_list",dist_phys_list)
    for i in dist_phys_list:
        # get the angle of the element which the corresponding dist_phys is the largest
        if i[1] == max(dist_phys_list,key=lambda x:x[1])[1]:
            user.angle = i[0]
            return user
    return None