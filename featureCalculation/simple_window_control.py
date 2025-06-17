#------------------------------------------------------------------------------
# 简化版窗口控制系统
# 基于显著性图实现简单的窗口避障
#------------------------------------------------------------------------------

import cv2
import numpy as np
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
from pynput import keyboard

# 全局配置
WINDOW_WIDTH = 853   # 窗口宽度（40°视场角）
WINDOW_HEIGHT = 480  # 窗口高度（16:9比例）
FIELD_SMOOTH = 2.0   # 场图平滑参数
ATTRACTION_STRENGTH = -2.0  # 吸引力强度
SPRING_K = 1.0       # 弹簧系数
SPRING_B = 2.0       # 阻尼系数
DT = 0.1            # 时间步长

def calculate_saliency_map(image):
    """计算显著性图"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    
    if success:
        saliency_map = (saliency_map * 255).astype(np.uint8)
        return saliency_map
    return np.zeros_like(gray)

def create_field_map(saliency_map, window_pos):
    """创建场图"""
    # 归一化显著性图
    field_map = saliency_map.astype(np.float32) / 255.0
    
    # 平滑处理
    field_map = cv2.GaussianBlur(field_map, (0, 0), FIELD_SMOOTH)
    
    # 创建吸引力场
    y, x = np.ogrid[:field_map.shape[0], :field_map.shape[1]]
    center_y, center_x = window_pos
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    attraction = ATTRACTION_STRENGTH * np.exp(-dist**2 / (2 * 50**2))
    
    # 合成场图
    field_map = field_map + attraction
    
    return field_map

def calculate_force(field_map, position):
    """计算力"""
    # 计算梯度
    grad_y, grad_x = np.gradient(field_map)
    
    # 获取当前位置的力
    y, x = int(position[1]), int(position[0])
    force_x = grad_x[y, x]
    force_y = grad_y[y, x]
    
    # 限制力的大小
    force = np.array([force_x, force_y])
    force = np.clip(force, -5.0, 5.0)
    
    return force

def update_window_position(position, velocity, force):
    """更新窗口位置"""
    # 计算到中心点的位移
    center = np.array([field_map.shape[1]/2, field_map.shape[0]/2])
    displacement = center - position
    
    # 计算弹簧力和阻尼力
    spring_force = SPRING_K * displacement
    damping_force = -SPRING_B * velocity
    
    # 合力
    total_force = spring_force + damping_force + force
    
    # 更新速度和位置
    velocity += total_force * DT
    position += velocity * DT
    
    # 限制位置在图像范围内
    position[0] = np.clip(position[0], 0, field_map.shape[1]-1)
    position[1] = np.clip(position[1], 0, field_map.shape[0]-1)
    
    return position, velocity

def draw_window(image, position, size=(WINDOW_WIDTH, WINDOW_HEIGHT)):
    """在图像上绘制窗口"""
    x, y = int(position[0]), int(position[1])
    w, h = size
    
    # 计算窗口左上角坐标
    x1 = x - w//2
    y1 = y - h//2
    
    # 绘制窗口边框
    cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
    
    # 绘制窗口中心点
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    
    return image

def main():
    # 初始化视频流
    host = "192.168.137.230"
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
    
    # 设置键盘监听
    listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
    listener.open()
    
    # 创建视频流客户端
    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO,
                            mode=hl2ss.StreamMode.MODE_1,
                            width=1920, height=1080,
                            framerate=30)
    client.open()
    
    # 初始化窗口状态
    window_pos = np.array([1920//2, 1080//2])  # 初始位置在中心
    window_vel = np.zeros(2)  # 初始速度为零
    
    try:
        while not listener.pressed():
            # 获取图像
            data = client.get_next_packet()
            if data is None:
                continue
                
            image = data.payload.image
            
            # 计算显著性图
            saliency_map = calculate_saliency_map(image)
            
            # 创建场图
            field_map = create_field_map(saliency_map, window_pos)
            
            # 计算力
            force = calculate_force(field_map, window_pos)
            
            # 更新窗口位置
            window_pos, window_vel = update_window_position(window_pos, window_vel, force)
            
            # 绘制窗口
            window_vis = image.copy()
            window_vis = draw_window(window_vis, window_pos)
            
            # 可视化场图
            field_vis = cv2.applyColorMap((field_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # 调整图像大小以便显示
            display_size = (640, 360)
            image_resized = cv2.resize(image, display_size)
            saliency_resized = cv2.resize(saliency_map, display_size)
            field_resized = cv2.resize(field_vis, display_size)
            window_resized = cv2.resize(window_vis, display_size)
            
            # 并排显示
            top_row = np.hstack((image_resized, saliency_resized))
            bottom_row = np.hstack((field_resized, window_resized))
            combined = np.vstack((top_row, bottom_row))
            
            cv2.imshow('Original | Saliency | Field Map | Window', combined)
            cv2.waitKey(1)
            
    finally:
        client.close()
        listener.close()
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 