#------------------------------------------------------------------------------
# 窗口控制模块
# 负责将Field Map转换为势场并控制窗口运动
#------------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

@dataclass
class SpringConfig:
    """弹簧模型配置"""
    k: float = 1.0      # 弹性系数
    b: float = 2.0      # 阻尼系数
    dt: float = 0.1     # 时间步长

class WindowController:
    def __init__(self, 
                 resolution: Tuple[int, int],
                 spring_config: SpringConfig = SpringConfig()):
        self.resolution = resolution
        self.spring_config = spring_config
        
        # 窗口状态
        self.position = np.zeros(2)  # 当前位置
        self.velocity = np.zeros(2)  # 当前速度
        
        # 势场
        self.potential_field = np.zeros(resolution)
        self.gradient_field = np.zeros((*resolution, 2))
    
    def update_potential_field(self, field_map: np.ndarray) -> None:
        """
        更新势场
        
        Args:
            field_map: 输入的Field Map
        """
        # 使用高斯滤波平滑势场
        self.potential_field = gaussian_filter(field_map, sigma=2.0)
        
        # 计算梯度场
        self._compute_gradient_field()
    
    def _compute_gradient_field(self) -> None:
        """计算梯度场"""
        # 使用中心差分计算梯度
        grad_y, grad_x = np.gradient(self.potential_field)
        self.gradient_field[..., 0] = grad_x
        self.gradient_field[..., 1] = grad_y
    
    def get_force(self, position: np.ndarray) -> np.ndarray:
        """
        获取指定位置的力
        
        Args:
            position: 位置坐标
            
        Returns:
            力向量
        """
        # 将世界坐标转换为地图坐标
        map_pos = self._world_to_map(position)
        
        # 获取该位置的梯度（力）
        force = self.gradient_field[int(map_pos[1]), int(map_pos[0])]
        
        return force
    
    def update_window_position(self) -> Tuple[float, float]:
        """
        更新窗口位置
        
        Returns:
            新的窗口位置
        """
        # 获取当前位置的力
        force = self.get_force(self.position)
        
        # 过阻尼弹簧模型
        # F = -kx - bv
        # 其中k是弹性系数，b是阻尼系数
        acceleration = force - self.spring_config.k * self.position - self.spring_config.b * self.velocity
        
        # 更新速度和位置
        self.velocity += acceleration * self.spring_config.dt
        self.position += self.velocity * self.spring_config.dt
        
        # 确保窗口在可视范围内
        self._clamp_position()
        
        return tuple(self.position)
    
    def _clamp_position(self) -> None:
        """限制窗口位置在可视范围内"""
        self.position = np.clip(self.position, 0, 1)
    
    def _world_to_map(self, world_coords: np.ndarray) -> np.ndarray:
        """世界坐标转换为地图坐标"""
        return world_coords * np.array(self.resolution)
    
    def _map_to_world(self, map_coords: np.ndarray) -> np.ndarray:
        """地图坐标转换为世界坐标"""
        return map_coords / np.array(self.resolution) 