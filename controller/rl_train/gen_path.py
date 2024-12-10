import json
import random
import math
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import tqdm
import os

def visualize_paths(boundary_virt_polygon, paths, show_target=True):
    """
    可视化路径及虚拟边界。
    
    参数:
    - boundary_virt_polygon: shapely Polygon对象，表示虚拟边界。
    - paths: 路径列表，每个路径为dict:
        {
            "target": {"x":..., "y":...},
            "path": [{"x":..., "y":..., "dir": ...}, ...]
        }
      若只传入一条路径，也应为包含该字典的列表。
    - show_target: 是否在图中标记目标点，默认True。
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制虚拟边界
    x,y = boundary_virt_polygon.exterior.xy
    ax.plot(x, y, color='black', linewidth=2, label="Virtual Boundary")
    
    for i, data in enumerate(paths):
        path = data["path"]
        target = data["target"]
        
        # 提取轨迹坐标
        px = [p["x"] for p in path]
        py = [p["y"] for p in path]
        
        # 绘制路径
        ax.plot(px, py, linewidth=1, label=f"Path {i+1}" if len(paths) > 1 else "Path")
        
        # 绘制起点(反转后路径的第一个点是用户起点)
        start_x, start_y = px[0], py[0]
        ax.scatter(start_x, start_y, c='green', s=30, marker='o', label="Start" if i == 0 else None)
        
        # 绘制终点（实际上是反转前的目标点所在处）
        end_x, end_y = px[-1], py[-1]
        ax.scatter(end_x, end_y, c='red', s=30, marker='x', label="End" if i == 0 else None)
        
        # 绘制目标点
        if show_target:
            ax.scatter(target["x"], target["y"], c='blue', s=30, marker='^', label="Target" if i == 0 else None)
    
    # 设置图形属性
    ax.set_aspect('equal', 'box')
    ax.set_title("Visualized Paths")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    
    # 去重图例条目
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.grid(True)
    plt.show()

def point_in_polygon(polygon, x_min, x_max, y_min, y_max, max_tries=10000):
    """在给定多边形内随机选点"""
    for _ in range(max_tries):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if polygon.contains(Point(x, y)):
            return x, y
    raise ValueError("Cannot find a random point inside the polygon within max_tries.")

def create_polygon_from_points(points):
    """从边界点构建多边形"""
    coords = [(p["x"], p["y"]) for p in points]
    return Polygon(coords)

def is_out_of_boundary(pos, boundary_polygon):
    """判断当前位置是否超出虚拟空间边界"""
    point = Point(pos[0], pos[1])
    return not boundary_polygon.contains(point)

def generate_path(boundary_virt_polygon,
                  num_paths=10,
                  speed=1.4,       # 行走速度: 1.4m/s
                  freq=60,         # 频率: 60Hz
                  min_turn_dist=0.5,
                  max_turn_dist=3.5,
                  turn_angle_std=math.pi/4,
                  min_radius=2.0,
                  max_radius=4.0,
                  output_file="paths.json"):
    """
    根据描述生成模拟路径数据并保存为JSON文件。
    每条数据格式：
    {
        "target": {"x":..., "y":...},
        "path": [
            {"x":..., "y":..., "dir": ...},
            ...
        ]
    }
    """
    # 首先获取边界的包围盒，方便随机取点
    minx, miny, maxx, maxy = boundary_virt_polygon.bounds

    dt = 1.0/freq
    results = []

    for i in tqdm.tqdm(range(num_paths)):
        # 1. 随机选取目标点(虚拟空间中)
        target_x, target_y = point_in_polygon(boundary_virt_polygon, minx, maxx, miny, maxy)

        # 2. 从目标点出发，随机初始方向(0~2π)
        direction = random.uniform(0, 2*math.pi)
        current_pos = (target_x, target_y)

        # 3. 模拟向外行走，直到出界
        path_with_dir = []
        path_with_dir.append({"x": current_pos[0], "y": current_pos[1], "dir": direction})
        
        distance_to_next_turn = random.uniform(min_turn_dist, max_turn_dist)
        dist_since_last_turn = 0.0

        # 行走直到出界
        turning = False
        turn_steps = 0
        turn_step_count = 0
        angle_increment = 0.0

        while True:
            # 如果出界则停止
            if is_out_of_boundary(current_pos, boundary_virt_polygon):
                break

            # 行进一步
            step_dist = speed * dt
            # 如果正在转向，则调整方向
            if turning:
                direction += angle_increment
                turn_step_count += 1
                if turn_step_count >= turn_steps:
                    turning = False

            # 前进
            new_x = current_pos[0] + step_dist * math.cos(direction)
            new_y = current_pos[1] + step_dist * math.sin(direction)
            current_pos = (new_x, new_y)
            path_with_dir.append({"x": current_pos[0], "y": current_pos[1], "dir": direction})

            dist_since_last_turn += step_dist

            # 检查是否需要转向
            if not turning and dist_since_last_turn > distance_to_next_turn:
                # 产生新的转向
                angle_change = random.gauss(0, turn_angle_std)
                if angle_change != 0:
                    radius = random.uniform(min_radius, max_radius)
                    total_time = abs(angle_change) * radius / speed
                    turn_steps = max(1, int(total_time / dt))
                    angle_increment = (angle_change / turn_steps)
                    turning = True
                    turn_step_count = 0

                # 重置距离计数和下一次转向距离
                dist_since_last_turn = 0.0
                distance_to_next_turn = random.uniform(min_turn_dist, max_turn_dist)

        # 反转路径
        reversed_path = list(reversed(path_with_dir))

        # 保存数据
        result = {
            "target": {"x": target_x, "y": target_y},
            "path": reversed_path
        }
        results.append(result)

        if (i+1) % 1000 == 0:
            print(f"Generated {i+1} paths...")

    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    # 示例环境数据
    env = {
        "border_phys": [
            {"x":0,"y":0},
            {"x":200,"y":0},
            {"x":200,"y":200},
            {"x":0,"y":200}
        ],
        "border_virt": [
            {"x":0,"y":0},
            {"x":400,"y":0},
            {"x":400,"y":400},
            {"x":0,"y":400}
        ],
        "obstacles_phys": [],
        "obstacles_virt": []
    }
    
    virt_polygon = create_polygon_from_points(env["border_virt"])
    if not os.path.exists("example_data"):
        os.makedirs("example_data")
    
    # # 生成路径数据
    # generate_path(virt_polygon, num_paths=128, output_file="example_data/paths_example_0.json")
    
    # 可视化生成的路径
    with open("example_data/paths_example_0.json", 'r', encoding='utf-8') as f:
        paths = json.load(f)
    visualize_paths(virt_polygon, paths, show_target=True)
