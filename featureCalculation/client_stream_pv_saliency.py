#------------------------------------------------------------------------------
# Cursor实现
# 需要再核对一下使用的算法。目前的效果高亮部分并不能很好地对应障碍位置。
#------------------------------------------------------------------------------

import cv2
import numpy as np
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
from pynput import keyboard

# Settings --------------------------------------------------------------------
host = "192.168.137.230"  # HoloLens address
mode = hl2ss.StreamMode.MODE_1
enable_mrc = False
shared = False

# Camera parameters
width = 1920
height = 1080
framerate = 30
profile = hl2ss.VideoProfile.H265_MAIN
bitrate = None
decoded_format = 'bgr24'

def calculate_saliency_map(image, method='static'):
    """
    Calculate saliency map using OpenCV's Static Saliency Spectral Residual
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create saliency object
    if method == 'static':
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    elif method == 'static_fine_grained':
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    # elif method == 'objectness':
    #     saliency = cv2.saliency.ObjectnessBING_create()
    #     saliency.setTrainingPath(training_path)	# 提取模型文件参数
    #     saliency.setBBResDir("Results")	# 将算法检测结果保存在Results文件夹内
    # elif method == 'motion':
    #     saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
    #     saliency.setImagesize(image.shape[1], image.shape[0])   # 设置图像大小
    #     saliency.init() # 初始化
    else:
        raise ValueError(f"Invalid method: {method}")
    
    # Calculate saliency map
    success, saliency_map = saliency.computeSaliency(gray)
    
    if success:
        # Normalize saliency map to 0-255
        saliency_map = (saliency_map * 255).astype(np.uint8)
        # Apply colormap for better visualization
        saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        return saliency_map
    return None

def main():
    # Start PV subsystem
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, 
                                enable_mrc=enable_mrc, shared=shared)

    # Setup keyboard listener for exit
    listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
    listener.open()

    # Create client
    client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO,
                            mode=mode, width=width, height=height,
                            framerate=framerate, profile=profile,
                            bitrate=bitrate, decoded_format=decoded_format)
    client.open()

    try:
        while not listener.pressed():
            # Get next frame
            data = client.get_next_packet()
            
            # Get image and calculate saliency
            image = data.payload.image
            saliency_map = calculate_saliency_map(image, method='static')
            saliency_map_fine_grained = calculate_saliency_map(image, method='static_fine_grained')

            if saliency_map is not None:
                # Resize images for better display
                display_size = (960, 540)
                image_resized = cv2.resize(image, display_size)
                saliency_resized = cv2.resize(saliency_map, display_size)
                saliency_resized_fine_grained = cv2.resize(saliency_map_fine_grained, display_size)
                
                # Combine images side by side
                combined = np.hstack((image_resized, saliency_resized, saliency_resized_fine_grained))
                
                # Display
                cv2.imshow('Original | Saliency Map | Saliency Map Fine Grained', combined)
                cv2.waitKey(1)
            
    finally:
        # Cleanup
        client.close()
        listener.close()
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 