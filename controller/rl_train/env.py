import numpy as np
import gym
from gym import spaces
import json
import random
from shapely.geometry import Polygon, Point, LineString
import math

class RedirectWalkingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        super(RedirectWalkingEnv, self).__init__()

        self.env_config = env_config
        # Define state dimensions, defaulting to 6 (e.g., [px, py, p_theta, vx, vy, v_theta])
        self.state_dim = env_config.get("state_dim", 6)
        self.stack_num = env_config.get("stack_num", 10)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim * self.stack_num,), dtype=np.float32
        )

        # Load path data from file
        with open(env_config["paths_file"], 'r', encoding='utf-8') as f:
            self.paths_data = json.load(f)

        # Define physical and virtual boundaries
        self.boundary_virt = Polygon([(p["x"], p["y"]) for p in env_config["border_virt"]])
        print("boundary_virt valid:", self.boundary_virt.is_valid)
        self.boundary_phys = Polygon([(p["x"], p["y"]) for p in env_config["border_phys"]])
        print("boundary_phys valid:", self.boundary_phys.is_valid)

        # Load virtual obstacles
        self.obstacles_virt = []
        for obs in env_config["obstacles_phys"]:
            self.obstacles_virt.append(Polygon([(p["x"], p["y"]) for p in obs]))
            print("obstacle_virt valid:", self.obstacles_virt[-1].is_valid)

        # Initialize state buffer
        self.state_buffer = np.zeros((self.stack_num, self.state_dim), dtype=np.float32)

        # Randomly select a path and initialize variables
        self.current_path = random.choice(self.paths_data)
        self.physical_position = np.array([0.0, 0.0], dtype=np.float32)
        self.virtual_position = np.array([0.0, 0.0], dtype=np.float32)
        self.physical_orientation = 0.0
        self.virtual_orientation = 0.0
        self.target_position = np.array([0.0, 0.0], dtype=np.float32)

        # Step counter
        self.step_count = 0

    def reset(self):
        # Randomly select a path
        self.current_path = random.choice(self.paths_data)
        # Set target points
        self.target_position_vir = np.array([
            self.current_path["target"]["x"], self.current_path["target"]["y"]
        ], dtype=np.float32)
        self.target_position_phys = np.array([100.0, 100.0], dtype=np.float32)  # Example center point

        # Initialize user position to the first point of the path
        start_x = self.current_path["path"][0]["x"]
        start_y = self.current_path["path"][0]["y"]
        start_dir = self.current_path["path"][0]["dir"]

        self.physical_position = np.array([100.0, 100.0], dtype=np.float32)  # Physical space origin
        self.virtual_position = np.array([start_x, start_y], dtype=np.float32)
        self.step_count = 0

        # Set initial orientations
        self.virtual_orientation = start_dir
        self.physical_orientation = self.virtual_orientation

        # Clear state buffer
        self.state_buffer = np.zeros((self.stack_num, self.state_dim), dtype=np.float32)
        initial_state = self._get_current_state()
        for i in range(self.stack_num):
            self.state_buffer[i] = initial_state
            
        print(len(self.current_path["path"]))

        return self.state_buffer.flatten()

    def step(self, action):
        if self.step_count + 1 >= len(self.current_path["path"]):
            return self.state_buffer.flatten(), 0.0, True, {}
        # Update positions and orientations based on action
        if_turn = self._apply_action(action)
        reward = self._compute_reward(if_turn)

        new_state = self._get_current_state()
        self.state_buffer = np.roll(self.state_buffer, shift=-1, axis=0)
        self.state_buffer[-1] = new_state

        done = self._check_done()
        return self.state_buffer.flatten(), reward, done, {}

    def _apply_action(self, action):
        if_turn = False
        self.step_count += 1

        # Sample gains from action
        translation_gain = action[0]
        rotation_gain = action[1]
        curvature_gain = action[2]

        # Update virtual position
        new_x = self.current_path["path"][self.step_count]["x"]
        new_y = self.current_path["path"][self.step_count]["y"]
        new_dir = self.current_path["path"][self.step_count]["dir"]

        # Compute v and w
        dx = new_x - self.virtual_position[0]
        dy = new_y - self.virtual_position[1]
        v = np.linalg.norm([dx, dy])  # 相当于calc_move_with_gain中的v
        w = new_dir - self.virtual_orientation  # 相当于calc_move_with_gain中的w

        # Compute trans and rot
        eps = 1e-6
        trans_gain = max(eps, translation_gain)
        rot_gain = max(eps, rotation_gain)
        cur_gain = max(eps, curvature_gain)

        trans = v / trans_gain
        rot = w / rot_gain

        # Update physical orientation and position
        self.physical_orientation += rot + trans / cur_gain
        self.physical_orientation %= (2 * math.pi)

        new_phys_x = self.physical_position[0] + trans * math.cos(self.physical_orientation)
        new_phys_y = self.physical_position[1] + trans * math.sin(self.physical_orientation)

        # Check boundary
        if not self._in_boundary([new_phys_x, new_phys_y]):
            if_turn = True
            self.physical_orientation = (self.physical_orientation + math.pi) % (2*math.pi)
            new_phys_x = self.physical_position[0] + trans * math.cos(self.physical_orientation)
            new_phys_y = self.physical_position[1] + trans * math.sin(self.physical_orientation)

        self.physical_position = np.array([new_phys_x, new_phys_y])
        self.virtual_position = np.array([new_x, new_y])
        self.virtual_orientation = new_dir

        return if_turn


    def _in_boundary(self, pos):
        """
        Check if the position is within boundaries and not colliding with obstacles.
        """
        if not self.boundary_phys.contains(Point(pos)):
            return False
        for obs in self.obstacles_virt:
            if obs.contains(Point(pos)):
                return False
        return True

    def _get_current_state(self):
        """
        Return the current state, e.g., [px, py, p_theta, vx, vy, v_theta].
        """
        state = np.concatenate([
            self.physical_position,
            [self.physical_orientation],
            self.virtual_position,
            [self.virtual_orientation]
        ])
        return state[:self.state_dim]

    def _ray_dist_to_boundary(self, pos, phi, max_dist=1000.0):
        start_x, start_y = pos
        end_x = start_x + max_dist * math.cos(phi)
        end_y = start_y + max_dist * math.sin(phi)
        ray = LineString([(start_x, start_y), (end_x, end_y)])
        intersection = self.boundary_phys.intersection(ray)

        if intersection.is_empty:
            return max_dist

        # 根据geometry类型分别处理
        if intersection.geom_type == 'Point':
            # 单一点交点
            return intersection.distance(Point(start_x, start_y))
        elif intersection.geom_type == 'MultiPoint':
            # 多点集合，取最近距离
            return min(pt.distance(Point(start_x, start_y)) for pt in intersection.geoms)
        elif intersection.geom_type == 'LineString':
            # 若为LineString，与起点距离即为ray从start到该线段的最短距离
            return intersection.distance(Point(start_x, start_y))
        else:
            # 如果有其他类型，例如Polygon或MultiLineString，根据实际情况处理
            # 尝试提取边界点或转换为点集合来求最近距离
            # 一个简化的处理是获取intersection的边界（boundary）或转换为点
            boundary = intersection.boundary
            if boundary.geom_type == 'MultiPoint':
                return min(pt.distance(Point(start_x, start_y)) for pt in boundary.geoms)
            elif boundary.geom_type == 'Point':
                return boundary.distance(Point(start_x, start_y))
            else:
                # fallback方案：直接返回max_dist或对boundary再次处理
                return max_dist


    def _compute_reward(self, if_turn):
        """
        Compute reward based on the current state and actions.
        """
        if if_turn:
            return -1.0

        dist_to_target = np.linalg.norm(self.virtual_position - self.target_position_vir)
        dist_to_target_phys = np.linalg.norm(self.physical_position - self.target_position_phys)

        if dist_to_target < 0.5:
            return 1.0 - 0.1 * dist_to_target_phys

        d_v = dist_to_target
        d_p = dist_to_target_phys

        vec_v = self.target_position_vir - self.virtual_position
        phi_target_v = math.atan2(vec_v[1], vec_v[0])
        theta_v = (phi_target_v - self.virtual_orientation + math.pi) % (2 * math.pi) - math.pi

        vec_p = self.target_position_phys - self.physical_position
        phi_target_p = math.atan2(vec_p[1], vec_p[0])
        theta_p = (phi_target_p - self.physical_orientation + math.pi) % (2 * math.pi) - math.pi

        R_d = -abs(d_v - d_p)
        R_o = -abs(theta_v - theta_p)

        phi_p = self.physical_orientation
        R_hat_forward = self._ray_dist_to_boundary(self.physical_position, phi_p)
        R_hat_left = self._ray_dist_to_boundary(self.physical_position, phi_p + math.pi / 2)
        R_hat_right = self._ray_dist_to_boundary(self.physical_position, phi_p - math.pi / 2)
        R_a = R_hat_forward + min(R_hat_left, R_hat_right)

        R_c = 0.5

        d_b = min(R_hat_forward, R_hat_left, R_hat_right)
        alpha = 5.0
        beta = 3.0
        lambda_a = math.exp(-alpha * d_b)
        lambda_d = math.exp(-beta * d_v)
        lambda_o = 1.0

        R = lambda_a * R_a + lambda_d * R_d + lambda_o * R_o + R_c

        return R

    def _check_done(self):
        """
        Check if the episode is done based on distance to the target or other criteria.
        """
        dist_to_target = np.linalg.norm(self.virtual_position - self.target_position_vir)
        if dist_to_target < 0.5:
            return True
        return False

    def render(self, mode='human'):
        pass

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        env_config = json.load(f)
    env = RedirectWalkingEnv(env_config)
    obs = env.reset()
