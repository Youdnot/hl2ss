#------------------------------------------------------------------------------
# 特征模型模块
# 提供各种特征计算的具体实现
#------------------------------------------------------------------------------

import cv2
import numpy as np
from typing import Optional, Tuple

class BaseFeatureModel:
    """特征模型基类"""
    def compute(self, *args, **kwargs) -> np.ndarray:
        """计算特征"""
        raise NotImplementedError

class SaliencyModel(BaseFeatureModel):
    """显著性模型"""
    def compute(self, image: np.ndarray) -> np.ndarray:
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建显著性对象
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        
        # 计算显著性图
        success, saliency_map = saliency.computeSaliency(gray)
        
        if success:
            # 归一化到0-1
            saliency_map = (saliency_map * 255).astype(np.uint8)
            return saliency_map / 255.0
        return np.zeros_like(gray)

class DepthModel(BaseFeatureModel):
    """深度估计模型"""
    def compute(self, image: np.ndarray) -> np.ndarray:
        # 这里使用简单的边缘检测作为示例
        # 实际应用中应该使用更复杂的深度估计模型
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges.astype(np.float32) / 255.0

class OpticalFlowModel(BaseFeatureModel):
    """光流模型"""
    def __init__(self):
        self.prev_gray = None
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def compute(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.flow_params
        )
        
        return flow 