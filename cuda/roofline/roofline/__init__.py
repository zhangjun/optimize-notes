"""
[INPUT]: 依赖 roofline.roofline 模块
[OUTPUT]: 对外导出 RooflinePlotter, plot_roofline
[POS]: roofline 包入口，模块级别的 API 暴露
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .roofline import RooflinePlotter, plot_roofline

__all__ = ['RooflinePlotter', 'plot_roofline']