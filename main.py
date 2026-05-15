# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from config import BUFFER_RADII
from environment import UAVEnvironment
from rrt_star_planner import RRTStarPlanner
from utils import run_statistical_experiment, save_results


def visualize_single_path(env, planner, buffer_radius, scenario_id):
    """生成单次完整路径可视化图（带RRT树和缓冲区）"""
    print(f"正在生成 Scenario {scenario_id} (buffer={buffer_radius}) 的路径可视化图...")
    path, path_length, _, _, min_clearance = planner.run_single()

    plt.figure(figsize=(12, 10))

    # 绘制环境（与environment.py一致）
    orange = [0.9290, 0.6940, 0.1250]
    for poly in env.poly_circle:
        plt.fill(poly[:, 0], poly[:, 1], 'white', edgecolor='none')
    for j, (cx, cy, a, b, angle_deg) in enumerate(env.circle_paras):
        theta = np.linspace(0, 2 * np.pi, 360)
        angle_rad = np.radians(angle_deg)
        x = a * np.cos(theta) * np.cos(angle_rad) - b * np.sin(theta) * np.sin(angle_rad) + cx
        y = a * np.cos(theta) * np.sin(angle_rad) + b * np.sin(theta) * np.cos(angle_rad) + cy
        plt.fill(x, y, color=orange, edgecolor='black', linewidth=1)
    for poly in env.transformed_polygons:
        plt.fill(poly[:, 0], poly[:, 1], color=orange, edgecolor='black', linewidth=1)

    # 绘制RRT搜索树
    for edge in planner.edges:
        n1, n2 = planner.nodes[edge[0]], planner.nodes[edge[1]]
        plt.plot([n1[0], n2[0]], [n1[1], n2[1]], 'b-', alpha=0.25, linewidth=0.6)

    # 绘制最终路径
    if path and len(path) > 1:
        path_arr = np.array(path)
        plt.plot(path_arr[:, 0], path_arr[:, 1], 'r-', linewidth=3, label=f'Path (Length: {path_length:.2f}m)')

    # 起点终点
    plt.plot(env.start_point[0], env.start_point[1], 'ro', markersize=12, label='Start')
    plt.plot(env.finish_point[0], env.finish_point[1], 'g*', markersize=15, label='Goal')

    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Scenario {scenario_id} - Buffer {buffer_radius}m\n'
              f'Min Clearance: {min_clearance:.2f}m | Success: {len(path) > 1}')
    plt.legend()

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/path_scenario{scenario_id}_buffer{buffer_radius}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"路径可视化图已保存：results/figures/path_scenario{scenario_id}_buffer{buffer_radius}.png\n")


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/data", exist_ok=True)

    print("=" * 70)
    print("UAV 地理围栏缓冲约束下 RRT* 路径规划实验（多场景 + 安全裕度 + 可视化）")
    print("=" * 70)

    for sid in [1, 2, 3]:
        print(f"\n{'=' * 20} Scenario {sid} {'=' * 20}")
        env = UAVEnvironment()
        env.build_environment(scenario_id=sid)
        env.visualize()

        num_runs = 30 if sid == 1 else 10

        for br in BUFFER_RADII:
            print(f"\n  → Buffer Radius = {br} (运行 {num_runs} 次)")
            df = run_statistical_experiment(RRTStarPlanner, env, br, num_runs)
            save_results(df, f"stats_scenario{sid}_buffer_{br}.csv")

            # 只对主场景 + buffer=0.2 生成可视化路径图（节省时间）
            if sid == 1 and br == 0.2:
                # 使用最后一次planner生成可视化
                planner = RRTStarPlanner(env, buffer_radius=br)
                visualize_single_path(env, planner, br, sid)

    print("\n" + "=" * 70)
    print("实验全部完成！已包含安全裕度指标和路径可视化图")
    print("可直接用于毕业论文第5章")
    print("=" * 70)