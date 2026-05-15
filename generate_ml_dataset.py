# generate_ml_dataset.py
"""
生成 Random Forest 缓冲区动态决策数据集
使用现有3个场景，批量运行 RRT*，提取特征 + 自动生成标签
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from environment import UAVEnvironment
from rrt_star_planner import RRTStarPlanner
from config import BUFFER_RADII


def calculate_local_density(env, point, radius=8.0):
    """计算局部障碍物密度"""
    count = 0
    pt = np.array(point)
    for poly in env.transformed_polygons:
        if np.any(np.linalg.norm(poly - pt, axis=1) < radius):
            count += 1
    for ellipse in env.poly_circle:
        if np.any(np.linalg.norm(ellipse - pt, axis=1) < radius):
            count += 1
    return count


def extract_features(env, planner, path, buffer_used):
    """提取单次规划的所有特征"""
    if not path or len(path) < 3:
        return None

    path_arr = np.array(path)

    # 1. 最小距离 & 安全裕度
    min_clearance = planner.calculate_min_clearance(path)

    # 2. 局部障碍物密度（取路径中点）
    mid_idx = len(path) // 2
    mid_point = path[mid_idx]
    local_density = calculate_local_density(env, mid_point)

    # 3. 路径长度相关
    path_length = planner.calculate_path_length(path)
    straight_dist = np.linalg.norm(env.finish_point - env.start_point)
    length_increase_ratio = (path_length - straight_dist) / straight_dist if straight_dist > 0 else 0

    # 4. 规划复杂度
    num_nodes = len(planner.nodes)
    planning_time = 0.0  # 后续可记录实际时间

    # 5. 偏航程度（路径与直线偏离）
    deviation = 0.0
    for p in path_arr:
        proj = np.dot(p - env.start_point, env.finish_point - env.start_point) / straight_dist
        closest = env.start_point + (proj / straight_dist) * (env.finish_point - env.start_point)
        deviation += np.linalg.norm(p - closest)
    avg_deviation = deviation / len(path)

    # 6. 模拟电量消耗（路径越长越耗电）
    sim_battery = max(0.0, 100 - path_length * 0.8)  # 简单线性模拟

    features = {
        'scenario_id': env.scenario_id,
        'buffer_used': buffer_used,
        'min_clearance': min_clearance,
        'local_density': local_density,
        'length_increase_ratio': length_increase_ratio,
        'num_nodes': num_nodes,
        'avg_deviation': avg_deviation,
        'sim_battery': sim_battery,
        'path_length': path_length,
        'straight_dist': straight_dist,
    }

    # 生成标签：是否值得开启缓冲区
    # 规则：安全提升明显 且 路径代价增加可接受
    clearance_gain = min_clearance if buffer_used else 0.0
    features['need_buffer'] = 1 if (clearance_gain > 0.08 and length_increase_ratio < 0.18) else 0

    return features


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs("results/ml_data", exist_ok=True)

    all_data = []
    num_runs_per_config = 150  # 每个场景-缓冲组合跑150次，可根据时间调整

    print("🚀 开始生成缓冲区决策数据集...")

    for sid in [1, 2, 3]:
        print(f"\n=== Scenario {sid} ===")
        env = UAVEnvironment()
        env.build_environment(sid)

        for br in [0.0, 0.2]:
            print(f"  Buffer = {br}m | 运行 {num_runs_per_config} 次...")
            for i in tqdm(range(num_runs_per_config)):
                planner = RRTStarPlanner(env, buffer_radius=br)
                path, path_length, planning_time, num_nodes, min_clearance = planner.run_single()

                if len(path) > 5:  # 成功路径
                    feat = extract_features(env, planner, path, br > 0)
                    if feat:
                        all_data.append(feat)

    # 保存数据集
    df = pd.DataFrame(all_data)
    df.to_csv("results/ml_data/buffer_decision_dataset.csv", index=False)

    print("\n✅ 数据集生成完成！")
    print(f"总样本数: {len(df)}")
    print(f"正样本 (need_buffer=1): {df['need_buffer'].sum()} ({df['need_buffer'].mean():.1%})")
    print(f"文件保存至: results/ml_data/buffer_decision_dataset.csv")

    # 简单统计
    print("\n特征统计:")
    print(df.describe())

    # 保存标签分布
    df.groupby(['scenario_id', 'buffer_used'])['need_buffer'].value_counts().unstack().to_csv(
        "results/ml_data/label_distribution.csv")
    print("标签分布已保存！")