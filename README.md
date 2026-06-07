# 🎨 ColorEngine V1.0.3: 光影与色彩仿色引擎


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)




---选择你的原始图片和目标图片一键仿色---



## 📂 项目结构 (Repository Structure)



```text
My_Color_Engine/
├── color_utils.py       # 🛠️ RGB/HSV 转换, 直方图匹配等纯数学工具
├── core_engine.py       # 🧠 处理所有的逻辑分流与蒙版
├── main.py              # 🚀 CLI 交互界面，图片读写，参数调优
└── README.md            # 📖 项目说明文档

环境依赖
确保你的 Python 环境中已安装以下核心库：
pip install torch torchvision pillow

在运行 main.py 时，你可以自由配置三个参数（范围 0.0 - 1.0）：

融合强度 (blend_strength): 控制原图向目标图色彩靠拢的力度。

保色阈值 (similarity_threshold): 容差度系统。数值越大，代表相色相似度更低的原图像素会被强制染成目标色。

光影匹配强度 (hist_strength): 控制最终画面明暗分布向目标图靠近的程度。

本项目采用 MIT License 开源协议。
