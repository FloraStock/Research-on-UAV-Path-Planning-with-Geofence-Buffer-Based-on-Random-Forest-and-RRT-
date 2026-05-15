# plot_3d.py  【最终修复版 - 解决文件名非法字符 + 保持MATLAB风格】
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from environment import UAVEnvironment
import os
import re   # 新增：用于清理文件名

def _safe_filename(name):
    name = re.sub(r'[\\/:*?"<>|]', '_', name)   # Windows非法字符全部替换为下划线
    return name.strip()

def plot_3d_path(env: UAVEnvironment, path, planner_name="RRT* (标准)", title=None):
    if title is None:
        title = f"3D - {planner_name} (Geofence Buffer)"

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    orange = [0.9290, 0.6940, 0.1250]
    height = 5.0

    # ================== 障碍物（完整3D柱体）==================
    for poly in env.poly_circle + env.transformed_polygons:
        n = len(poly)
        bottom_verts = [list(zip(poly[:,0], poly[:,1], np.zeros(n)))]
        top_verts = [list(zip(poly[:,0], poly[:,1], np.full(n, height)))]
        side_verts = []
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            side_verts.append([(p1[0], p1[1], 0), (p2[0], p2[1], 0),
                               (p2[0], p2[1], height), (p1[0], p1[1], height)])
        ax.add_collection3d(Poly3DCollection(bottom_verts, facecolors=orange, alpha=0.85, edgecolor='black', linewidth=0.8))
        ax.add_collection3d(Poly3DCollection(top_verts,   facecolors=orange, alpha=0.85, edgecolor='black', linewidth=0.8))
        ax.add_collection3d(Poly3DCollection(side_verts,   facecolors=orange, alpha=0.75, edgecolor='black', linewidth=0.6))

    # ================== 路径 ===================
    if path and len(path) > 1:
        path_arr = np.array(path)
        ax.plot(path_arr[:,0], path_arr[:,1], 2.0 * np.ones(len(path_arr)),
                'r-', linewidth=6, alpha=0.95, label='规划路径')

    # ================== 起点 & 终点 ==================
    ax.scatter([env.start_point[0]], [env.start_point[1]], [2], c='red', s=200, marker='o',
               edgecolors='darkred', linewidth=2.5, label='起点')
    ax.scatter([env.finish_point[0]], [env.finish_point[1]], [2], c='lime', s=260, marker='*',
               edgecolors='darkgreen', linewidth=2.5, label='终点')

    # ================== UAV模型 ==================
    if path and len(path) > 1:
        uav_pos = path[1]
        ax.plot([uav_pos[0]-1.2, uav_pos[0]+1.2], [uav_pos[1], uav_pos[1]], [2, 2], 'k-', linewidth=4.5)
        ax.plot([uav_pos[0], uav_pos[0]], [uav_pos[1]-2, uav_pos[1]+2], [2, 2], 'k-', linewidth=3)

    # ================== 视角与样式 ==================
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=15, pad=20)
    ax.set_xlim([-8, 38])
    ax.set_ylim([-8, 38])
    ax.set_zlim([-1, 6])
    ax.view_init(elev=35, azim=-60)     # 俯视角度清晰

    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.show()

    # ================== 保存图片（关键修复：清理文件名）==================
    safe_planner = _safe_filename(planner_name)
    save_path = f"results/figures/3d_path_scenario{env.scenario_id}_{safe_planner}.png"
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"✅ 3D图已成功保存：{save_path}")