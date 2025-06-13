Next steps
- [ ] read through data format
- [ ] feature calculation
- [ ] window control using python
  - [ ] can borrow from https://github.com/jdibenes/mrtk_remote_ui/blob/main/client/demo_ui_videos_images_text.py





该文件是调用hololens2的Front Camera并在本主机上进行串流收集数据的一个典型程序实例。详细解释以下问题：

1. 该程序如何实现了串流收集数据？数据的内容结构是怎么样的？
2. hl2ss主组件是怎么集成到各文件的python实现中的？
3. 我需要在实现串流后对返回的数据在流式的情况下进行提取和计算。怎么类似的功能？在本文件的基础上编写一个新的文件，功能是实现串流收集rgb数据可视化的基础上使用opencv实时计算获得的图片中的saliency map并可视化。



该文件是一个典型的示例程序，演示了如何通过 Hololens 2 的 Front Camera（前置摄像头）进行视频数据的串流，并在本地主机上收集这些数据。请基于该程序文件，详细解释并分析以下问题：

问题分析与解释任务：

1. 串流数据收集的实现机制：
    程序是如何连接并从 Hololens 2 获取前置摄像头图像数据的？
    数据是如何通过网络串流传输到本地主机的？是否使用了特定协议（如 TCP/UDP）？
    收集到的数据结构是怎样的？例如图像的分辨率、格式（如 RGB/BGR）、数据打包方式等。
2. hl2ss 组件的集成方式：
    hl2ss 的核心模块在本项目中的作用是什么？
    各 Python 文件中是如何导入并使用 hl2ss 的？请结合关键模块（如 hl2ss_stream_viewer, hl2ss_rx, hl2ss_stream 等）说明其在整体程序结构中的角色。
    是否对 hl2ss 进行了定制或二次封装？

扩展任务：功能拓展与实现

1. 实时图像处理与可视化：
    在原有程序串流收集 RGB 图像并可视化的基础上，扩展一个新 Python 文件，实现以下功能：

读取和显示 RGB 数据流：使用 OpenCV 实时显示前置摄像头采集的图像。
 计算显著性图（Saliency Map）：对每一帧图像使用 OpenCV 的 saliency 模块或其他视觉注意力机制方法计算显著性图。
 结果可视化：将原始图像与对应的显著性图实时并排显示。

要求：
 实现需保证处理的实时性；
 代码结构清晰，逻辑模块分明（数据读取、显著性计算、显示）；
 若使用第三方库（如 OpenCV Contrib），请在注释中说明安装方式与依赖。



 Multiple streams can be active at the same time but only one client per stream is allowed.
 这个问题可能导致每个流在重新获取测试时需要重启设备。