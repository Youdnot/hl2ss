#------------------------------------------------------------------------------
# 光流特征计算实现
# 使用OpenCV的Farneback方法计算密集光流
#------------------------------------------------------------------------------

import cv2
import numpy as np
from template_feature_calculation import VideoStreamProcessor

class OpticalFlowProcessor(VideoStreamProcessor):
    def __init__(self, **kwargs):
        """
        初始化光流处理器
        
        Args:
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        # 光流计算参数
        self.prev_gray = None
        self.flow_params = dict(
            pyr_scale=0.5,    # 图像金字塔缩放比例
            levels=3,         # 金字塔层数
            winsize=15,       # 窗口大小
            iterations=3,     # 迭代次数
            poly_n=5,         # 多项式展开邻域大小
            poly_sigma=1.2,   # 高斯标准差
            flags=0           # 额外标志
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像
        将彩色图像转换为灰度图
        
        Args:
            frame: 输入图像帧
            
        Returns:
            灰度图像
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def process_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        计算两帧之间的光流
        
        Args:
            frames: 两帧连续图像
            
        Returns:
            可视化后的光流图
        """
        if len(frames) != 2:
            return None
            
        # 转换为灰度图
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, **self.flow_params
        )
        
        # 将光流转换为可视化图像
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 创建HSV图像
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # 色调
        hsv[..., 1] = 255  # 饱和度
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度
        
        # 转换为BGR用于显示
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 将原始图像和光流图并排显示
        combined = np.hstack((frames[1], flow_vis))
        
        return combined

def main():
    processor = OpticalFlowProcessor()
    processor.initialize()
    processor.run()

if __name__ == "__main__":
    main() 