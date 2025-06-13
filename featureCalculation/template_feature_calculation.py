#------------------------------------------------------------------------------
# 特征计算模板文件
# 该文件提供了视频流读取和特征计算的基本框架
#------------------------------------------------------------------------------

from pynput import keyboard
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_utilities

class VideoStreamProcessor:
    def __init__(self, 
                 host: str = "192.168.137.230",
                 width: int = 1920,
                 height: int = 1080,
                 framerate: int = 30,
                 buffer_size: int = 2):
        """
        初始化视频流处理器
        
        Args:
            host: HoloLens的IP地址
            width: 视频宽度
            height: 视频高度
            framerate: 帧率
            buffer_size: 帧缓存大小
        """
        self.host = host
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # 视频流设置
        self.mode = hl2ss.StreamMode.MODE_1
        self.enable_mrc = False
        self.shared = False
        self.profile = hl2ss.VideoProfile.H265_MAIN
        self.bitrate = None
        self.decoded_format = 'bgr24'
        
        # 处理状态
        self.is_running = False
        self.client = None
        self.listener = None

    def initialize(self):
        """初始化视频流连接"""
        hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, 
                                    enable_mrc=self.enable_mrc, shared=self.shared)
        
        self.listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
        self.listener.open()
        
        self.client = hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO,
                                     mode=self.mode, width=self.width, height=self.height,
                                     framerate=self.framerate, profile=self.profile,
                                     bitrate=self.bitrate, decoded_format=self.decoded_format)
        self.client.open()
        self.is_running = True

    def cleanup(self):
        """清理资源"""
        if self.client:
            self.client.close()
        if self.listener:
            self.listener.close()
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)
        cv2.destroyAllWindows()
        self.is_running = False

    def get_frame(self) -> Optional[np.ndarray]:
        """获取一帧图像"""
        if not self.is_running:
            return None
            
        data = self.client.get_next_packet()
        if data is None:
            return None
            
        frame = data.payload.image
        self.frame_buffer.append(frame)
        return frame

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        处理单帧图像
        子类需要重写此方法来实现具体的特征计算
        
        Args:
            frame: 输入图像帧
            
        Returns:
            处理后的图像或特征图
        """
        raise NotImplementedError("Subclasses must implement process_frame method")

    def process_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        处理多帧图像
        子类需要重写此方法来实现需要多帧的特征计算
        
        Args:
            frames: 输入图像帧列表
            
        Returns:
            处理后的图像或特征图
        """
        raise NotImplementedError("Subclasses must implement process_frames method")

    def run(self):
        """运行视频处理循环"""
        try:
            while not self.listener.pressed():
                frame = self.get_frame()
                if frame is None:
                    continue
                
                # 处理单帧
                result = self.process_frame(frame)
                
                # 处理多帧（当缓存中有足够的帧时）
                if len(self.frame_buffer) == self.frame_buffer.maxlen:
                    multi_frame_result = self.process_frames(list(self.frame_buffer))
                
                # 显示结果
                if result is not None:
                    cv2.imshow('Processed Frame', result)
                    cv2.waitKey(1)
                
        finally:
            self.cleanup()

def main():
    processor = VideoStreamProcessor()
    processor.initialize()
    processor.run()

if __name__ == "__main__":
    main() 