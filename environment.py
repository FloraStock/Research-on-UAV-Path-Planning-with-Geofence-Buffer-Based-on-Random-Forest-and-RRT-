# environment.py
"""
UAV仿真环境建模模块（支持多种场景）
完全复现MATLAB环境，同时支持3种不同复杂度地图
用于构建UAV路径规划仿真数据库
"""

import numpy as np
import matplotlib.pyplot as plt
from config import K_SCALE, PT_BASE, N_CIRCLE_EDGE

class UAVEnvironment:
    """UAV路径规划仿真环境（支持多场景）"""

    def __init__(self):
        self.scenario_id = 1
        self.start_point = None
        self.finish_point = None
        self.transformed_polygons = []
        self.circle_paras = None
        self.poly_circle = []
        self.node = None
        self.N_node = 0

    def build_environment(self, scenario_id=1):
        """构建指定场景的环境"""
        self.scenario_id = scenario_id
        print(f"正在构建 Scenario {scenario_id} 环境...")

        if scenario_id == 1:
            # === Scenario 1: 标准地图（你原来的复杂场景）===
            self.start_point = np.array([70, 210])
            self.finish_point = np.array([105, 350])
            polygons_data = [
                np.array([[40, 220], [80, 220], [80, 210]]),
                np.array([[125, 220], [85, 220], [85, 210]]),
                np.array([[5, 260], [15, 260], [15, 285], [45, 285],
                          [45, 295], [15, 295], [15, 340], [5, 340]]),
                np.array([]),  # 对称生成
                np.array([[80, 310], [55, 310], [35, 320], [35, 330], [80, 330]]),
                np.array([]),  # 对称生成
                np.array([[70, 340], [35, 340], [35, 345], [50, 345],
                          [50, 360], [55, 360], [55, 345], [70, 345]]),
                np.array([]),  # 对称生成
                np.array([[75, 340], [90, 340], [90, 345], [75, 345]])
            ]
            polygons_data[3] = np.column_stack([165 - polygons_data[2][:, 0], polygons_data[2][:, 1]])
            polygons_data[5] = np.column_stack([165 - polygons_data[4][:, 0], polygons_data[4][:, 1]])
            polygons_data[7] = np.column_stack([165 - polygons_data[6][:, 0], polygons_data[6][:, 1]])

            self.circle_paras = np.array([
                [55, 250, 30, 10, -45],
                [110, 250, 30, 10, 45],
                [82.5, 280, 25, 25, 0]
            ])

        elif scenario_id == 2:
            # === Scenario 2: 简单地图（障碍物少，空间较大）===
            self.start_point = np.array([30, 30])
            self.finish_point = np.array([120, 180])
            polygons_data = [
                np.array([[60, 80], [90, 80], [90, 100], [60, 100]]),   # 一个矩形
                np.array([[100, 120], [130, 120], [130, 140], [100, 140]])
            ]
            self.circle_paras = np.array([
                [75, 130, 18, 8, 30]   # 一个倾斜椭圆
            ])

        elif scenario_id == 3:
            # === Scenario 3: 复杂地图（障碍物密集，挑战性更高）===
            self.start_point = np.array([20, 40])
            self.finish_point = np.array([140, 190])
            polygons_data = [
                np.array([[40, 70], [70, 70], [70, 85], [40, 85]]),
                np.array([[90, 60], [110, 60], [110, 90], [90, 90]]),
                np.array([[30, 110], [50, 110], [50, 130], [30, 130]]),
                np.array([[120, 100], [145, 100], [145, 125], [120, 125]]),
                np.array([[55, 150], [80, 150], [80, 170], [55, 170]])
            ]
            # 增加更多椭圆
            self.circle_paras = np.array([
                [65, 100, 22, 12, -40],
                [105, 140, 25, 10, 35],
                [80, 50, 15, 15, 0]
            ])

        # 统一进行缩放和平移 + 生成椭圆多边形 + 节点矩阵
        self._apply_transformation(polygons_data)
        self._create_ellipse_polygons_matlab()
        self._build_node_matrix()

        print(f"Scenario {scenario_id} 环境构建完成！")
        print(f"起点: {self.start_point.round(2)}, 终点: {self.finish_point.round(2)}")
        print(f"多边形障碍物: {len(self.transformed_polygons)} 个 | 椭圆障碍物: {len(self.poly_circle)} 个\n")

    def _apply_transformation(self, polygons_data):
        """应用缩放 (k=1/5) 和平移变换"""
        self.circle_paras = self.circle_paras.astype(float)
        self.circle_paras[:, :4] *= K_SCALE
        self.circle_paras[:, :2] -= PT_BASE

        self.start_point = self.start_point * K_SCALE - PT_BASE
        self.finish_point = self.finish_point * K_SCALE - PT_BASE

        self.transformed_polygons = []
        for poly in polygons_data:
            if len(poly) > 0:
                transformed = poly * K_SCALE - PT_BASE
                self.transformed_polygons.append(transformed)

    def _create_ellipse_polygons_matlab(self):
        """完全按照MATLAB逻辑生成椭圆多边形"""
        self.poly_circle = []
        for j in range(len(self.circle_paras)):
            cx, cy, a, b, angle_deg = self.circle_paras[j]
            list_indices = np.arange(1, N_CIRCLE_EDGE + 1).reshape(-1, 1)
            cos_vals = np.cos((2 * list_indices - 1) * np.pi / N_CIRCLE_EDGE)
            sin_vals = np.sin((2 * list_indices - 1) * np.pi / N_CIRCLE_EDGE)
            trig_matrix = np.column_stack([cos_vals, sin_vals])

            diag_ab = np.diag([a, b])
            angle_rad = np.radians(angle_deg)
            rotation_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                        [-np.sin(angle_rad), np.cos(angle_rad)]])

            denominator = np.cos(np.pi / N_CIRCLE_EDGE)
            offset = trig_matrix @ diag_ab @ rotation_matrix / denominator
            center_points = np.tile([cx, cy], (N_CIRCLE_EDGE, 1))
            ellipse_poly = center_points + offset
            self.poly_circle.append(ellipse_poly)

    def _build_node_matrix(self):
        """构建节点矩阵"""
        all_nodes = [self.start_point]
        for poly in self.transformed_polygons:
            all_nodes.extend(poly)
        for ellipse_poly in self.poly_circle:
            all_nodes.extend(ellipse_poly)
        all_nodes.append(self.finish_point)

        self.node = np.array(all_nodes)
        self.N_node = len(self.node)

    def visualize(self, title=None, save=True):
        """可视化当前场景环境"""
        if title is None:
            title = f"UAV Environment - Scenario {self.scenario_id} (Geofence Buffer Constraints)"

        plt.figure(figsize=(10, 8))

        # 1. 白色填充椭圆多边形
        for poly in self.poly_circle:
            plt.fill(poly[:, 0], poly[:, 1], 'white', edgecolor='none')

        # 2. 橙色填充椭圆
        orange = [0.9290, 0.6940, 0.1250]
        for j, (cx, cy, a, b, angle_deg) in enumerate(self.circle_paras):
            theta = np.linspace(0, 2 * np.pi, 360)
            angle_rad = np.radians(angle_deg)
            x = a * np.cos(theta) * np.cos(angle_rad) - b * np.sin(theta) * np.sin(angle_rad) + cx
            y = a * np.cos(theta) * np.sin(angle_rad) + b * np.sin(theta) * np.cos(angle_rad) + cy
            plt.fill(x, y, color=orange, edgecolor='black', linewidth=1)

        # 3. 橙色填充多边形障碍物
        for poly in self.transformed_polygons:
            plt.fill(poly[:, 0], poly[:, 1], color=orange, edgecolor='black', linewidth=1)

        # 4. 起点和终点
        plt.plot(self.start_point[0], self.start_point[1], 'ro',
                 markersize=12, markerfacecolor='r', markeredgewidth=2, label='Start')
        plt.plot(self.finish_point[0], self.finish_point[1], 'g*',
                 markersize=15, markerfacecolor='g', markeredgewidth=2, label='Goal')

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()

        if save:
            import os
            os.makedirs("results/figures", exist_ok=True)
            filename = f"results/figures/environment_scenario_{self.scenario_id}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"环境图已保存：{filename}")
        plt.show()