# rrt_star_planner.py
"""
RRT* 路径规划器（支持地理围栏缓冲约束 + 安全裕度计算）
完全兼容 MATLAB map_circle_12.m 和 Plot_3D.m 的路径数据结构
"""

import numpy as np
import time
from shapely.geometry import Point, LineString, Polygon
from config import MAX_ITER, STEP_SIZE, GOAL_RADIUS

class RRTStarPlanner:
    def __init__(self, env: 'UAVEnvironment', buffer_radius=0.3):
        self.env = env
        self.buffer_radius = buffer_radius
        self.obstacles = self._create_shapely_obstacles()

        self.nodes = [env.start_point.copy()]
        self.edges = []
        self.parents = {0: -1}
        self.costs = {0: 0.0}

    def _create_shapely_obstacles(self):
        """将环境中的多边形和椭圆转换为 Shapely Polygon（用于精确碰撞检测）"""
        obstacles = []
        for poly in self.env.transformed_polygons:
            obstacles.append(Polygon(poly))
        for ellipse_poly in self.env.poly_circle:
            obstacles.append(Polygon(ellipse_poly))
        return obstacles

    def is_collision_free(self, point):
        """判断单个点是否在缓冲区外"""
        pt = Point(point)
        for obs in self.obstacles:
            if obs.buffer(self.buffer_radius).intersects(pt):
                return False
        return True

    def is_path_collision_free(self, p1, p2):
        """判断路径段是否与缓冲区相交"""
        line = LineString([p1, p2])
        for obs in self.obstacles:
            if obs.buffer(self.buffer_radius).intersects(line):
                return False
        return True

    def sample(self, goal_bias=0.2):
        """采样：带目标偏置"""
        if np.random.rand() < goal_bias:
            return self.env.finish_point.copy()
        x_min, y_min = -5, -5
        x_max, y_max = 50, 50
        return np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])

    def nearest_neighbor(self, point):
        """找到最近节点"""
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return np.argmin(distances)

    def steer(self, from_node, to_point, step_size=STEP_SIZE):
        """从 from_node 向 to_point 扩展 step_size 距离"""
        direction = to_point - from_node
        dist = np.linalg.norm(direction)
        if dist < step_size:
            return to_point.copy()
        return from_node + (direction / dist) * step_size

    def rewire(self, new_idx, radius=20.0):
        """RRT* 重规划（优化父节点）"""
        new_node = self.nodes[new_idx]
        new_cost = self.costs[new_idx]
        for i, node in enumerate(self.nodes):
            if i == new_idx or np.linalg.norm(node - new_node) > radius:
                continue
            if not self.is_path_collision_free(new_node, node):
                continue
            potential_cost = new_cost + np.linalg.norm(new_node - node)
            if potential_cost < self.costs.get(i, float('inf')):
                self.parents[i] = new_idx
                self.costs[i] = potential_cost

    def run_single(self):
        """单次 RRT* 规划主函数"""
        start_time = time.time()

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

                if np.linalg.norm(new_node - self.env.finish_point) < GOAL_RADIUS:
                    break

        planning_time = time.time() - start_time
        path = self.extract_path()
        path_length = self.calculate_path_length(path) if path is not None else 1e6
        min_clearance = self.calculate_min_clearance(path) if path is not None else 0.0
        num_nodes = len(self.nodes)

        return path, path_length, planning_time, num_nodes, min_clearance

    def calculate_path_length(self, path):
        """计算路径总长度"""
        if not path or len(path) < 2:
            return 1e6
        return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))

    def calculate_min_clearance(self, path):
        """计算路径最小安全裕度（到障碍物的最小距离 - buffer）"""
        if not path or len(path) < 2:
            return 0.0
        min_dist = float('inf')
        for p in path:
            pt = Point(p)
            for obs in self.obstacles:
                dist = obs.distance(pt) - self.buffer_radius
                if dist < min_dist:
                    min_dist = dist
        return max(0.0, min_dist)

    def extract_path(self):
        """从目标回溯提取最终路径"""
        if not self.nodes:
            return None
        goal_dists = [np.linalg.norm(node - self.env.finish_point) for node in self.nodes]
        nearest_to_goal = np.argmin(goal_dists)
        path = []
        current = nearest_to_goal
        while current != -1:
            path.append(self.nodes[current])
            current = self.parents.get(current, -1)
        return list(reversed(path))

    def get_buffered_obstacles(self):
        """返回缓冲后的障碍物（供 GUI 可视化）"""
        buffered = []
        for obs in self.obstacles:
            buffered.append(obs.buffer(self.buffer_radius))
        return buffered

    def plot_environment_with_buffer(self, ax, show_buffer=True, title=None):
        """2D环境 + 缓冲区可视化（供 GUI 调用）"""
        if title is None:
            title = f"Scenario {self.env.scenario_id} - Buffer {self.buffer_radius}m"

        orange = [0.9290, 0.6940, 0.1250]
        for poly in self.env.poly_circle:
            ax.fill(poly[:, 0], poly[:, 1], 'white', edgecolor='none')
        for j, (cx, cy, a, b, angle_deg) in enumerate(self.env.circle_paras):
            theta = np.linspace(0, 2 * np.pi, 360)
            angle_rad = np.radians(angle_deg)
            x = a * np.cos(theta) * np.cos(angle_rad) - b * np.sin(theta) * np.sin(angle_rad) + cx
            y = a * np.cos(theta) * np.sin(angle_rad) + b * np.sin(theta) * np.cos(angle_rad) + cy
            ax.fill(x, y, color=orange, edgecolor='black', linewidth=1)
        for poly in self.env.transformed_polygons:
            ax.fill(poly[:, 0], poly[:, 1], color=orange, edgecolor='black', linewidth=1)

        if show_buffer:
            for buf in self.get_buffered_obstacles():
                if buf is not None:
                    x, y = buf.exterior.xy
                    ax.fill(x, y, color='red', alpha=0.15, linewidth=0)

        ax.plot(self.env.start_point[0], self.env.start_point[1], 'ro',
                markersize=12, markerfacecolor='r', markeredgewidth=2, label='Start')
        ax.plot(self.env.finish_point[0], self.env.finish_point[1], 'g*',
                markersize=15, markerfacecolor='g', markeredgewidth=2, label='Goal')

        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()