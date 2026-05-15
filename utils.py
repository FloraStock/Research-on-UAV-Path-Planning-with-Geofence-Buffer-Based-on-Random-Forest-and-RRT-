# utils.py
import numpy as np
import pandas as pd
import os
from config import NUM_RUNS

def run_statistical_experiment(planner_class, env, buffer_radius, num_runs=NUM_RUNS):
    results = []
    success_count = 0

    print(f"正在运行 {num_runs} 次独立实验 (buffer_radius = {buffer_radius}) ...")

    for i in range(num_runs):
        np.random.seed(42 + i)

        planner = planner_class(env, buffer_radius=buffer_radius)
        path, path_length, planning_time, num_nodes, min_clearance = planner.run_single()

        success = path_length < 1e6 and len(path) > 1 if path is not None else False
        if success:
            success_count += 1

        results.append({
            'run_id': i + 1,
            'success': success,
            'path_length': path_length,
            'planning_time': planning_time,
            'num_nodes': num_nodes,
            'min_clearance': min_clearance,   # 新增安全裕度
            'buffer_radius': buffer_radius
        })

    df = pd.DataFrame(results)
    success_rate = (success_count / num_runs) * 100

    print(f"Buffer {buffer_radius}: 成功率 {success_rate:.1f}% | "
          f"平均路径长度 {df['path_length'].mean():.2f} | "
          f"平均安全裕度 {df['min_clearance'].mean():.2f} | "
          f"平均时间 {df['planning_time'].mean():.2f}s")

    return df


def save_results(df, filename):
    os.makedirs("results/data", exist_ok=True)
    filepath = f"results/data/{filename}"
    df.to_csv(filepath, index=False)
    print(f"统计结果已保存：{filepath}")