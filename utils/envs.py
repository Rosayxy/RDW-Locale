import numpy as np
from utils.space import *
from controller.client_logic import *

class PassiveHapticsEnv(object):
    def __init__(self, config, func_name = "s2c"):
       self.func_name = func_name
       self.config = config
       self.physical_space = Space(config["border_phys"], config["obstacles_phys"],config["border_virt"],config["obstacles_virt"])
       self.v_x, self.v_y, self.v_dir, self.v, self.w = 0.0, 0.0, 0.0, 0.0, 0.0
       self.p_x, self.p_y, self.p_dir = 0.0, 0.0, 0.0


    def reset(self):
        self.v_x, self.v_y, self.v_dir, self.v, self.w = 0.0, 0.0, 0.0, 0.0, 0.0
        self.p_x, self.p_y, self.p_dir = 0.0, 0.0, 0.0
        
        


    def vPathUpdate(self):
        self.v_x, self.v_y, self.v_dir, self.v, self.w = self.v_path[self.v_step_pointer]
        self.v_step_pointer += 1

    def physical_step(self, gt, gr, gc):
        x = self.p_x
        y = self.p_y
        angle = self.p_dir
        v = self.v
        w = self.w
        
        trans = v / gt
        rot = w / gr
        
        angle += rot + trans/gc
        angle = angle % (2 * np.pi)
        x += trans * np.cos(angle)
        y += trans * np.sin(angle)
        if self.physical_space.in_obstacle(x, y):
            return True # need reset
        self.p_x = x
        self.p_y = y
        self.p_dir = angle
        return False



    def step_specific_path(self, path):
        x_l = []
        y_l = []
        collide_num = 0
        self.v_path = path
        self.v_step_pointer = 0
        self.v_x, self.v_y, self.v_dir, self.v, self.w = self.v_path[self.v_step_pointer]
        self.p_x, self.p_y = self.physical_space.get_center()
        self.p_dir = 0.0
        self.v_step_pointer += 1
        i = self.v_step_pointer
        need_reset = False
        while i < len(self.v_path):
            user = UserInfo(self.p_x,self.p_y,self.p_dir,self.v,self.w,self.v_x,self.v_y,self.v_dir)
            delta_t = 0.02
            if need_reset:
                user = update_reset(user, self.physical_space, delta_t,self.func_name)
                self.p_dir = user.angle
                need_reset = False
                collide_num += 1
            else:
                gt, gr, gc, cur_direction = calc_gain(user, self.physical_space, delta_t,self.func_name) # type: ignore
                gc = gc*(cur_direction)/abs(cur_direction)
                need_reset = self.physical_step(gt, gr, gc)
                if not need_reset: # only no collide that we can update the user's state
                    x_l.append(self.p_x)
                    y_l.append(self.p_y)
                    self.vPathUpdate()
                    i += 1

        return  x_l, y_l, collide_num

   
    def print(self):
        print("v_x: ", self.v_x, "v_y: ", self.v_y, "v_dir: ", self.v_dir, "v: ", self.v, "w: ", self.w)
        print("p_x: ", self.p_x, "p_y: ", self.p_y, "p_dir: ", self.p_dir)



