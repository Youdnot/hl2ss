# AR窗口自适应系统

这是一个基于环境感知的AR窗口自适应系统，用于避免AR窗口与真实环境中的障碍物发生遮挡。

## 系统架构

系统包含以下主要模块：

1. **特征提取模块** (`core/feature_extractor.py`)
   - 从视频流中提取各种特征
   - 支持显著性、深度、光流等特征计算
   - 可扩展的特征模型接口

2. **Field Map构建模块** (`core/field_map.py`)
   - 将特征转换为多层次的网格代价图
   - 支持静态障碍物、动态障碍物、显著性等层
   - 可配置的层权重和合成策略

3. **窗口控制模块** (`core/window_controller.py`)
   - 将Field Map转换为势场
   - 使用过阻尼弹簧模型控制窗口运动
   - 平滑、稳定的窗口移动

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：
```bash
python main.py
```

## 配置

可以在`main.py`中调整以下参数：

- 处理分辨率
- 窗口初始位置和大小
- 安全距离
- 弹簧模型参数

## 扩展

1. 添加新的特征：
   - 在`models/feature_models.py`中创建新的特征模型类
   - 继承`BaseFeatureModel`并实现`compute`方法

2. 添加新的Field Map层：
   - 在`core/field_map.py`中创建新的层类
   - 继承`FieldMapLayer`并实现`update`方法

## 注意事项

- 确保HoloLens2设备已正确连接
- 检查网络连接和IP地址配置
- 根据实际需求调整参数 