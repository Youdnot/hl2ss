#------------------------------------------------------------------------------
# 特征提取模块
# 负责从视频流中提取各种特征，包括显著性、深度、光流等
#------------------------------------------------------------------------------

import cv2
import numpy as np
from typing import Dict, Any, Optional
from collections import deque

from ..utils.video_processor import VideoStreamProcessor
from ..models.feature_models import SaliencyModel, DepthModel, OpticalFlowModel

class FeatureExtractor(VideoStreamProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化特征模型
        self.models = {
            'saliency': SaliencyModel(),
            'depth': DepthModel(),
            'optical_flow': OpticalFlowModel()
        }
        
        # 特征缓存
        self.feature_buffer = deque(maxlen=2)
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像，提取各种特征
        
        Args:
            frame: 输入图像帧
            
        Returns:
            包含各种特征的字典
        """
        features = {}
        
        # 提取显著性图
        features['saliency'] = self.models['saliency'].compute(frame)
        
        # 提取深度图
        features['depth'] = self.models['depth'].compute(frame)
        
        # 存储特征
        self.feature_buffer.append(features)
        
        return features
    
    def process_frames(self, frames: list) -> Dict[str, Any]:
        """
        处理多帧图像，提取时序特征
        
        Args:
            frames: 输入图像帧列表
            
        Returns:
            包含时序特征的字典
        """
        if len(frames) != 2:
            return None
            
        # 计算光流
        flow_features = self.models['optical_flow'].compute(frames[0], frames[1])
        
        # 合并特征
        features = self.feature_buffer[-1].copy()
        features.update(flow_features)
        
        return features
    
    def get_features(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的特征
        
        Returns:
            最新的特征字典
        """
        if len(self.feature_buffer) > 0:
            return self.feature_buffer[-1]
        return None 