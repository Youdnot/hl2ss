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

class StaticLayer(FieldMapLayer):
    """静态障碍物层"""
    def update(self, features: Dict[str, Any]) -> None:
        # 使用深度图构建静态障碍物层
        depth_map = features.get('depth', None)
        if depth_map is not None:
            # 将深度图转换为障碍物代价
            self.map = self._depth_to_cost(depth_map)
    
    def _depth_to_cost(self, depth_map: np.ndarray) -> np.ndarray:
        # 深度图转换为代价的简单实现
        return 1.0 / (depth_map + 1e-6)

class DynamicLayer(FieldMapLayer):
    """动态障碍物层"""
    def update(self, features: Dict[str, Any]) -> None:
        # 使用光流构建动态障碍物层
        flow = features.get('optical_flow', None)
        if flow is not None:
            # 计算运动强度
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            self.map = self._flow_to_cost(magnitude)
    
    def _flow_to_cost(self, magnitude: np.ndarray) -> np.ndarray:
        # 运动强度转换为代价
        return np.clip(magnitude / 10.0, 0, 1)

class SaliencyLayer(FieldMapLayer):
    """显著性层"""
    def update(self, features: Dict[str, Any]) -> None:
        # 使用显著性图构建层
        saliency = features.get('saliency', None)
        if saliency is not None:
            self.map = saliency

class FieldMap:
    """Field Map管理器"""
    def __init__(self, resolution: Tuple[int, int], window_config: WindowConfig):
        self.resolution = resolution
        self.window_config = window_config
        
        # 初始化各层
        self.layers = {
            'static': StaticLayer(resolution),
            'dynamic': DynamicLayer(resolution),
            'saliency': SaliencyLayer(resolution)
        }
        
        # 层权重
        self.weights = {
            'static': 0.5,
            'dynamic': 0.3,
            'saliency': 0.2
        }
        
    def update(self, features: Dict[str, Any]) -> None:
        """更新所有层"""
        for layer in self.layers.values():
            layer.update(features)
    
    def get_monolithic_map(self) -> np.ndarray:
        """获取合成的单层地图"""
        monolithic_map = np.zeros(self.resolution)
        
        # 加权合成
        for name, layer in self.layers.items():
            monolithic_map += self.weights[name] * layer.get_map()
        
        # 添加窗口位置信息
        self._add_window_info(monolithic_map)
        
        return monolithic_map
    
    def _add_window_info(self, map: np.ndarray) -> None:
        """添加窗口位置信息到地图"""
        # 将窗口位置转换为地图坐标
        window_pos = self._world_to_map(self.window_config.position)
        window_size = self._world_to_map(self.window_config.size)
        
        # 在窗口位置添加标记
        x, y = window_pos
        w, h = window_size
        map[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = -1
    
    def _world_to_map(self, world_coords: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转换为地图坐标"""
        # 简单的线性映射
        x = int(world_coords[0] * self.resolution[0])
        y = int(world_coords[1] * self.resolution[1])
        return (x, y) 