#------------------------------------------------------------------------------
# Field Map构建模块
# 负责将特征转换为多层次的网格代价图
#------------------------------------------------------------------------------

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class WindowConfig:
    """窗口配置"""
    position: Tuple[float, float]  # 初始位置
    size: Tuple[float, float]      # 窗口大小
    margin: float                  # 与障碍物的安全距离
    anchor_position: Tuple[float, float]  # 目标位置（用于吸引力计算）

class FieldMapLayer:
    """Field Map层基类"""
    def __init__(self, resolution: Tuple[int, int]):
        self.resolution = resolution
        self.map = np.zeros(resolution)
        
    def update(self, features: Dict[str, Any]) -> None:
        """更新层数据"""
        raise NotImplementedError
        
    def get_map(self) -> np.ndarray:
        """获取层数据"""
        return self.map

class ObstacleLayer(FieldMapLayer):
    """障碍物层基类"""
    def __init__(self, resolution: Tuple[int, int]):
        super().__init__(resolution)
        self.normalization_range = (0, 1)  # 归一化范围
        
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据到指定范围"""
        min_val, max_val = self.normalization_range
        return (data - data.min()) / (data.max() - data.min() + 1e-6) * (max_val - min_val) + min_val

class StaticLayer(ObstacleLayer):
    """静态障碍物层"""
    def update(self, features: Dict[str, Any]) -> None:
        # 使用障碍物掩膜
        obstacle_mask = features.get('obstacle_mask', None)
        if obstacle_mask is not None:
            self.map = self.normalize(obstacle_mask)

class DynamicLayer(ObstacleLayer):
    """动态障碍物层"""
    def update(self, features: Dict[str, Any]) -> None:
        # 使用光流构建动态障碍物层
        flow = features.get('optical_flow', None)
        if flow is not None:
            # 计算运动强度
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            self.map = self.normalize(magnitude)

class SaliencyLayer(ObstacleLayer):
    """显著性层"""
    def update(self, features: Dict[str, Any]) -> None:
        saliency = features.get('saliency', None)
        if saliency is not None:
            # 显著性图直接作为阻力场
            self.map = self.normalize(saliency)

class WindowLayer(FieldMapLayer):
    """窗口位置层"""
    def __init__(self, resolution: Tuple[int, int], window_config: WindowConfig):
        super().__init__(resolution)
        self.window_config = window_config
        self.attraction_strength = -2.0  # 吸引力强度（负值表示吸引力）
        
    def update(self, features: Dict[str, Any]) -> None:
        # 创建吸引力场
        self.map = np.zeros(self.resolution)
        
        # 计算目标位置的地图坐标
        target_pos = self._world_to_map(self.window_config.anchor_position)
        
        # 创建距离场
        y, x = np.ogrid[:self.resolution[0], :self.resolution[1]]
        dist = np.sqrt((x - target_pos[0])**2 + (y - target_pos[1])**2)
        
        # 使用高斯函数创建吸引力场
        sigma = 50  # 控制吸引力范围
        self.map = self.attraction_strength * np.exp(-dist**2 / (2 * sigma**2))

class FieldMap:
    """Field Map管理器"""
    def __init__(self, resolution: Tuple[int, int], window_config: WindowConfig):
        self.resolution = resolution
        self.window_config = window_config
        
        # 初始化各层
        self.obstacle_layers = {
            'static': StaticLayer(resolution),
            'dynamic': DynamicLayer(resolution),
            'saliency': SaliencyLayer(resolution)
        }
        
        # 窗口位置层（独立于障碍物层）
        self.window_layer = WindowLayer(resolution, window_config)
        
        # 障碍物层权重
        self.weights = {
            'static': 0.5,
            'dynamic': 0.3,
            'saliency': 0.2
        }
        
    def update(self, features: Dict[str, Any]) -> None:
        """更新所有层"""
        # 更新障碍物层
        for layer in self.obstacle_layers.values():
            layer.update(features)
        
        # 更新窗口位置层
        self.window_layer.update(features)
    
    def get_monolithic_map(self) -> np.ndarray:
        """获取合成的单层地图"""
        # 首先合成障碍物层
        obstacle_map = np.zeros(self.resolution)
        for name, layer in self.obstacle_layers.items():
            obstacle_map += self.weights[name] * layer.get_map()
        
        # 添加窗口位置层（直接叠加，不参与加权）
        monolithic_map = obstacle_map + self.window_layer.get_map()
        
        return monolithic_map
    
    def _world_to_map(self, world_coords: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转换为地图坐标"""
        x = int(world_coords[0] * self.resolution[0])
        y = int(world_coords[1] * self.resolution[1])
        return (x, y) 