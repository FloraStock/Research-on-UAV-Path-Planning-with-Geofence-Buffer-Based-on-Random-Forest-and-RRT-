# gui_app.py  【完整最终版 - 第一阶段整合完成】
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import pandas as pd

from environment import UAVEnvironment
from rrt_star_planner import RRTStarPlanner
from informed_rrt_star_planner import InformedRRTStarPlanner
from safe_rrt_star_planner import SafeRRTStarPlanner
from utils import run_statistical_experiment
from config import BUFFER_RADII
from plot_3d import plot_3d_path   # 3D模块

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class UAVPathPlanningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UAV RRT* 路径规划 - 地理围栏缓冲约束演示系统")
        self.root.geometry("1400x900")

        self.env = None
        self.current_scenario = 1
        self.current_buffer = 0.2

        self.planner_classes = {
            "RRT* (标准)": RRTStarPlanner,
            "Informed RRT* (智能采样)": InformedRRTStarPlanner,
            "Safe RRT* (安全裕度优先)": SafeRRTStarPlanner
        }

        # 左侧控制面板
        self.control_frame = tk.Frame(root, width=340, padx=12, pady=12)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 右侧可视化区域
        self.vis_frame = tk.Frame(root)
        self.vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_controls()
        self.setup_visualization()
        self.load_environment()   # 自动加载默认场景

    def setup_controls(self):
        """完整控件初始化（已修复所有变量）"""
        ttk.Label(self.control_frame, text="UAV RRT* 路径规划系统", font=("微软雅黑", 16, "bold")).pack(pady=8)

        # 场景选择
        ttk.Label(self.control_frame, text="选择地图场景:", font=("微软雅黑", 11)).pack(anchor=tk.W, pady=(10,2))
        self.scenario_var = tk.IntVar(value=1)
        for sid, name in [(1,"Scenario 1 - 标准复杂地图"), (2,"Scenario 2 - 简单地图"), (3,"Scenario 3 - 高密度地图")]:
            ttk.Radiobutton(self.control_frame, text=name, variable=self.scenario_var, value=sid,
                           command=self.load_environment).pack(anchor=tk.W)

        # 缓冲区滑块
        ttk.Label(self.control_frame, text="缓冲区半径 (m):", font=("微软雅黑", 11)).pack(anchor=tk.W, pady=(15,2))
        self.buffer_slider = tk.Scale(self.control_frame, from_=0.0, to=0.5, resolution=0.05,
                                     orient=tk.HORIZONTAL, command=self.update_buffer, length=280)
        self.buffer_slider.set(0.2)
        self.buffer_slider.pack(fill=tk.X)

        # 规划算法选择
        ttk.Label(self.control_frame, text="选择规划算法:", font=("微软雅黑", 11)).pack(anchor=tk.W, pady=(15,2))
        self.planner_var = tk.StringVar(value="RRT* (标准)")
        planner_combo = ttk.Combobox(self.control_frame, textvariable=self.planner_var,
                                    values=list(self.planner_classes.keys()), state="readonly", width=30)
        planner_combo.pack(anchor=tk.W)

        # 统计实验次数
        ttk.Label(self.control_frame, text="统计实验次数:", font=("微软雅黑", 11)).pack(anchor=tk.W, pady=(10,2))
        self.runs_var = tk.IntVar(value=15)
        tk.Entry(self.control_frame, textvariable=self.runs_var, width=10).pack(anchor=tk.W)

        # 按钮区
        btn_pady = 8
        tk.Button(self.control_frame, text="📊 显示初始环境", width=30, bg="#4CAF50", fg="white",
                  command=self.show_environment).pack(pady=btn_pady, fill=tk.X)
        tk.Button(self.control_frame, text="🚀 单次规划", width=30, bg="#2196F3", fg="white",
                  command=self.run_single_planning).pack(pady=btn_pady, fill=tk.X)
        tk.Button(self.control_frame, text="📈 运行统计实验", width=30, bg="#FF9800", fg="white",
                  command=self.run_statistical).pack(pady=btn_pady, fill=tk.X)
        tk.Button(self.control_frame, text="📊 三种方法对比", width=30, bg="#E91E63", fg="white",
                  command=self.run_comparison).pack(pady=btn_pady, fill=tk.X)
        tk.Button(self.control_frame, text="💾 保存当前结果", width=30, bg="#9C27B0", fg="white",
                  command=self.save_current).pack(pady=btn_pady, fill=tk.X)

        # 结果显示区
        ttk.Label(self.control_frame, text="运行结果:", font=("微软雅黑", 11)).pack(anchor=tk.W, pady=(15,2))
        self.result_text = tk.Text(self.control_frame, height=18, width=38, font=("Consolas", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def setup_visualization(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_environment(self):
        self.current_scenario = self.scenario_var.get()
        self.env = UAVEnvironment()
        self.env.build_environment(self.current_scenario)
        self.update_buffer(None)
        self.show_environment()

    def update_buffer(self, val=None):
        self.current_buffer = self.buffer_slider.get()

    def show_environment(self):
        self.ax.clear()
        if self.env:
            temp_planner = RRTStarPlanner(self.env, buffer_radius=self.current_buffer)
            temp_planner.plot_environment_with_buffer(self.ax, show_buffer=True,
                title=f"Scenario {self.current_scenario} - Buffer = {self.current_buffer:.2f}m")
            self.canvas.draw()

    def get_current_planner_class(self):
        return self.planner_classes[self.planner_var.get()]

    def run_single_planning(self):
        try:
            if not self.env:
                messagebox.showwarning("警告", "请先加载环境！")
                return
            planner_class = self.get_current_planner_class()
            self.planner = planner_class(self.env, buffer_radius=self.current_buffer)
            path, path_length, planning_time, num_nodes, min_clearance = self.planner.run_single()

            # 更新2D图
            self.ax.clear()
            title = f"Scenario {self.current_scenario} - {self.planner_var.get()}\nBuffer={self.current_buffer:.2f}m | 最小安全裕度={min_clearance:.3f}m"
            self.planner.plot_environment_with_buffer(self.ax, show_buffer=True, title=title)
            self.canvas.draw()

            # 显示3D图（参考Plot_3D.m）
            if messagebox.askyesno("3D结果", "是否立即显示3D路径规划结果图？"):
                plot_3d_path(self.env, path, title=f"3D - {self.planner_var.get()}")

            # 显示结果
            success = len(path) > 1 and path_length < 1e6
            result_str = f"✅ 单次规划完成！\n\n成功：{'是' if success else '否'}\n路径长度：{path_length:.2f} m\n规划时间：{planning_time:.2f} s\n最小安全裕度：{min_clearance:.3f} m"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_str)
        except Exception as e:
            messagebox.showerror("运行错误", f"发生错误：{e}\n\n请把终端完整报错信息发给我，我会立刻修复！")

    # 其余方法（run_statistical、run_comparison、save_current）保持您原文件逻辑
    # 为简洁此处省略（与您原来完全一致），实际保存时请保留

    def run_statistical(self):
        if not self.env:
            messagebox.showwarning("警告", "请先加载环境！");
            return
        num_runs = self.runs_var.get()
        planner_class = self.get_current_planner_class()
        df = run_statistical_experiment(planner_class, self.env, self.current_buffer, num_runs)
        avg_success = df['success'].mean() * 100
        avg_length = df['path_length'].mean()
        avg_time = df['planning_time'].mean()
        avg_clearance = df['min_clearance'].mean()
        result_str = f"📈 统计实验完成 ({self.planner_var.get()})\n运行 {num_runs} 次\n\n成功率: {avg_success:.1f}%\n平均路径长度: {avg_length:.2f} m\n平均规划时间: {avg_time:.2f} s\n平均安全裕度: {avg_clearance:.3f} m\n\n数据已保存至 results/data/"
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_str)

    def run_comparison(self):
        if not self.env:
            messagebox.showwarning("警告", "请先加载环境！");
            return
        num_runs = self.runs_var.get()
        results = []
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在运行三种方法对比，请稍候...\n")
        for name, planner_class in self.planner_classes.items():
            self.result_text.insert(tk.END, f"  → 正在运行 {name} ...\n")
            self.root.update()
            df = run_statistical_experiment(planner_class, self.env, self.current_buffer, num_runs)
            results.append({
                '方法': name,
                '成功率(%)': round(df['success'].mean() * 100, 1),
                '平均路径长度(m)': round(df['path_length'].mean(), 2),
                '平均规划时间(s)': round(df['planning_time'].mean(), 2),
                '平均安全裕度(m)': round(df['min_clearance'].mean(), 3),
            })
        comp_df = pd.DataFrame(results)
        text = f"\n📊 三种方法对比结果\n场景: Scenario {self.current_scenario} | Buffer = {self.current_buffer}m | 次数: {num_runs}\n\n" + comp_df.to_string(
            index=False)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        os.makedirs("results/data", exist_ok=True)
        filename = f"results/data/comparison_scenario{self.current_scenario}_buffer{self.current_buffer}.csv"
        comp_df.to_csv(filename, index=False)
        messagebox.showinfo("对比完成", f"对比完成！\n已保存至：{filename}")

    def save_current(self):
        os.makedirs("results/figures", exist_ok=True)
        self.fig.savefig(f"results/figures/gui_scenario{self.current_scenario}_buffer{self.current_buffer:.2f}.png",
                         dpi=300)
        messagebox.showinfo("保存成功", "当前可视化结果已保存！")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = UAVPathPlanningGUI(root)
        root.mainloop()
    except Exception as e:
        print("启动GUI失败！错误信息：", e)
        import traceback
        traceback.print_exc()