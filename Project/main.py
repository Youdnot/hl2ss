#------------------------------------------------------------------------------
# 主程序入口
# 整合所有模块并运行系统
#------------------------------------------------------------------------------

import cv2
import numpy as np
from typing import Tuple

from core.feature_extractor import FeatureExtractor
from core.field_map import FieldMap, WindowConfig
from core.window_controller import WindowController, SpringConfig

def main():
    # 配置参数
    resolution = (640, 480)  # 处理分辨率
    window_config = WindowConfig(
        position=(0.5, 0.5),  # 初始位置（归一化坐标）
        size=(0.2, 0.2),     # 窗口大小（归一化坐标）
        margin=0.05          # 安全距离
    )
    spring_config = SpringConfig(
        k=1.0,   # 弹性系数
        b=2.0,   # 阻尼系数
        dt=0.1   # 时间步长
    )
    
    # 初始化各个模块
    feature_extractor = FeatureExtractor(
        width=1920,
        height=1080,
        framerate=30,
        buffer_size=2
    )
    
    field_map = FieldMap(resolution, window_config)
    window_controller = WindowController(resolution, spring_config)
    
    try:
        # 初始化视频流
        feature_extractor.initialize()
        
        while True:
            # 获取特征
            features = feature_extractor.get_features()
            if features is None:
                continue
            
            # 更新Field Map
            field_map.update(features)
            monolithic_map = field_map.get_monolithic_map()
            
            # 更新势场
            window_controller.update_potential_field(monolithic_map)
            
            # 更新窗口位置
            new_position = window_controller.update_window_position()
            
            # 显示结果
            cv2.imshow('Field Map', monolithic_map)
            cv2.imshow('Original', feature_extractor.frame_buffer[-1])
            
            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        # 清理资源
        feature_extractor.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 