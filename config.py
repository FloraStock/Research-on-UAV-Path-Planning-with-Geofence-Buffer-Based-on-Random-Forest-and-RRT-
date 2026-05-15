# config.py
import numpy as np

# 实验参数（可统一修改）
SEED = 42
NUM_RUNS = 50                      # 统计实验次数，建议50~100
MAX_ITER = 3000
STEP_SIZE = 2.0
GOAL_RADIUS = 2.0

# 缓冲区参数（对应geofence buffer约束）
BUFFER_RADII = [0.0, 0.2, 0.35]

# 环境建模参数（与MATLAB完全一致）
K_SCALE = 1.0 / 5
PT_BASE = np.array([1.0, 42.0])
N_CIRCLE_EDGE = 12