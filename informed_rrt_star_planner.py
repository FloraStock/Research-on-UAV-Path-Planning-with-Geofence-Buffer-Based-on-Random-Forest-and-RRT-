# informed_rrt_star_planner.py
"""
Informed RRT* 路径规划器（支持地理围栏缓冲约束 + 安全裕度计算）
"""
import numpy as np
import time
from shapely.geometry import Point, LineString, Polygon
from config import MAX_ITER, STEP_SIZE, GOAL_RADIUS
from rrt_star_planner import RRTStarPlanner

class InformedRRTStarPlanner(RRTStarPlanner):
    def __init__(self, env, buffer_radius=0.3):
        super().__init__(env, buffer_radius)
        self.best_cost = float('inf')
        self.start = env.start_point.copy()
        self.goal = env.finish_point.copy()
        self.c_min = np.linalg.norm(self.goal - self.start)
        self.first_solution_found = False

    def sample(self, goal_bias=0.2):
        if self.best_cost < float('inf') and self.first_solution_found:
            return self._sample_informed_ellipse()
        else:
            if np.random.rand() < goal_bias:
                return self.goal.copy()
            x_min, y_min = -5, -5
            x_max, y_max = 50, 50
            return np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])

    def _sample_informed_ellipse(self):
        c_best = self.best_cost
        c_min = self.c_min
        if c_best <= c_min + 1e-6:
            return self.goal.copy()
        center = (self.start + self.goal) / 2.0
        dx = self.goal[0] - self.start[0]
        dy = self.goal[1] - self.start[1]
        theta = np.arctan2(dy, dx)
        a = c_best / 2.0
        b = np.sqrt(max(0.0, c_best**2 - c_min**2)) / 2.0
        r = np.sqrt(np.random.uniform(0, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        x = r * a * np.cos(phi)
        y = r * b * np.sin(phi)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        return center + np.array([x_rot, y_rot])

    def run_single(self):
        start_time = time.time()
        self.best_cost = float('inf')
        self.first_solution_found = False
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
                self.costs[new_idx] = self.costs[nearest_idx] + np.linalg.norm(new_node - nearest_node)
                self.edges.append((nearest_idx, new_idx))
                self.rewire(new_idx, radius=min(30.0, 100 * np.sqrt(np.log(len(self.nodes)) / len(self.nodes))))
                if np.linalg.norm(new_node - self.goal) < GOAL_RADIUS:
                    tentative_cost = self.costs[new_idx] + np.linalg.norm(new_node - self.goal)
                    if tentative_cost < self.best_cost:
                        self.best_cost = tentative_cost
                        self.first_solution_found = True
        planning_time = time.time() - start_time
        path = self.extract_path()
        path_length = self.calculate_path_length(path) if path is not None else 1e6
        min_clearance = self.calculate_min_clearance(path) if path is not None else 0.0
        num_nodes = len(self.nodes)
        return path, path_length, planning_time, num_nodes, min_clearance