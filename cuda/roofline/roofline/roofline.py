#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[INPUT]: 依赖 matplotlib.pyplot, numpy
[OUTPUT]: 对外提供 RooflinePlotter 类、plot_roofline 函数
[POS]: roofline 模块的核心绘图组件，被 __init__.py 导出
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class RooflinePlotter:
    """Roofline 模型可视化工具"""

    def __init__(
        self,
        peak_flops: float,
        memory_bandwidth: float,
        title: str = "Roofline Model"
    ):
        self.peak_flops = peak_flops
        self.memory_bandwidth = memory_bandwidth
        self.title = title
        self.kernels: List[Tuple[str, float, float]] = []

    def add_kernel(self, name: str, flops: float, bytes_per_byte: float):
        self.kernels.append((name, flops, bytes_per_byte))

    def plot(self, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 7))

        oi = np.logspace(-2, 2, 500)
        compute_roof = np.full_like(oi, self.peak_flops)
        memory_roof = oi * self.memory_bandwidth
        roofline = np.minimum(compute_roof, memory_roof)

        ax.loglog(oi, compute_roof, 'k--', linewidth=1.5, label='Compute Peak')
        ax.loglog(oi, memory_roof, 'k:', linewidth=1.5, label='Memory Peak')
        ax.loglog(oi, roofline, 'k-', linewidth=2.5, label='Roofline')

        for name, flops, oi_val in self.kernels:
            ax.scatter(oi_val, flops, s=150, zorder=5)
            ax.annotate(
                name, (oi_val, flops),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=12)
        ax.set_ylabel('Performance (FLOPS)', fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.set_xlim(0.01, 100)
        ax.set_ylim(self.peak_flops * 0.01, self.peak_flops * 1.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def plot_roofline(
    peak_flops: float,
    memory_bandwidth: float,
    kernels: List[Tuple[str, float, float]],
    title: str = "Roofline Model",
    save_path: Optional[str] = None
):
    plotter = RooflinePlotter(peak_flops, memory_bandwidth, title)
    for name, flops, oi in kernels:
        plotter.add_kernel(name, flops, oi)
    plotter.plot(save_path)


if __name__ == '__main__':
    peak_flops = 7e12
    memory_bandwidth = 900e9

    kernels = [
        ("GEMM", 6e12, 50),
        ("Stencil", 2e12, 0.5),
        ("Reduce", 0.8e12, 0.1)
    ]

    plot_roofline(peak_flops, memory_bandwidth, kernels, "V100 Roofline")