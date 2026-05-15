# buffer_utils.py
from shapely.geometry import Polygon

def apply_geofence_buffer(polygons, buffer_r1=0.3, buffer_r2=0.1):
    """两级缓冲区：硬约束 + 柔性约束（对应研究内容1）"""
    buffered = []
    for poly in polygons:
        hard = poly.buffer(buffer_r1)
        soft = poly.buffer(buffer_r1 + buffer_r2).difference(hard)
        buffered.extend([hard, soft])
    return buffered