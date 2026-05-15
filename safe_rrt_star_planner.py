# safe_rrt_star_planner.py
"""
Safe RRT* (安全裕度优先) - 高质量稳定变体
在路径代价中加入安全裕度惩罚，使路径倾向于远离障碍物
适合毕业论文对比分析
"""
import numpy as np
import time
from shapely.geometry import Point
from rrt_star_planner import RRTStarPlanner
from config import MAX_ITER, STEP_SIZE, GOAL_RADIUS


class SafeRRTStarPlanner(RRTStarPlanner):
    def __init__(self, env, buffer_radius=0.3, safety_weight=8.0):
        super().__init__(env, buffer_radius)
        self.safety_weight = safety_weight  # 安全权重，可调（越大越安全）

    def calculate_clearance(self, point):
        """计算单个点的安全裕度"""
        pt = Point(point)
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = obs.distance(pt) - self.buffer_radius
            if dist < min_dist:
                min_dist = dist
        return max(0.0, min_dist)

    def run_single(self):
        start_time = time.time()
        self.nodes = [self.env.start_point.copy()]
        self.parents = {0: -1}
        self.costs = {0: 0.0}
        self.edges = []

        for i in range(MAX_ITER):
            rand_point = self.sample()
            nearest_idx = self.nearest_neighbor(rand_point)
            nearest_node = self.nodes[nearest_idx]
            new_node = self.steer(nearest_node, rand_point)

            if (self.is_collision_free(new_node) and
                    self.is_path_collision_free(nearest_node, new_node)):

                new_idx = len(self.nodes)
                self.nodes.append(new_node)
                self.parents[new_idx] = nearest_idx

                # 基础路径长度
                base_cost = self.costs[nearest_idx] + np.linalg.norm(new_node - nearest_node)

                # 安全裕度惩罚（clearance越小惩罚越大）
                clearance = self.calculate_clearance(new_node)
                safety_penalty = self.safety_weight * max(0.0, self.buffer_radius - clearance * 0.8)

                self.costs[new_idx] = base_cost + safety_penalty
                self.edges.append((nearest_idx, new_idx))

                # 重连优化（保留RRT*优势）
                self.rewire(new_idx, radius=min(30.0, 100 * np.sqrt(np.log(len(self.nodes)) / len(self.nodes))))

                if np.linalg.norm(new_node - self.env.finish_point) < GOAL_RADIUS:
                    break

        planning_time = time.time() - start_time
        path = self.extract_path()
        path_length = self.calculate_path_length(path) if path is not None else 1e6
        min_clearance = self.calculate_min_clearance(path) if path is not None else 0.0
        num_nodes = len(self.nodes)

        return path, path_length, planning_time, num_nodes, min_clearance