# ResNet50 图像分类推理优化服务

本项目是 AI 模型部署入门级实战项目，基于 FastAPI 搭建 ResNet50 图像分类服务，完成了从原生 PyTorch 推理到 ONNX Runtime 全流程优化落地，支持单张 / 批量图片推理，附带完整的对等性能基准测试体系，可直接作为生产级图像分类服务的基础模板。

## ✨ 核心特性

1. **双推理引擎兼容**：同时支持原生 PyTorch 与 ONNX Runtime 推理引擎，对外接口完全一致，可一键切换
2. **RESTful 规范接口**：提供单张 / 批量图片上传推理接口，配套 Swagger 可视化文档，开箱即用
3. **全链路推理优化**：通过静态图转换、算子融合、常量折叠、线程对齐等手段，吞吐量提升 158%+
4. **专业性能测试体系**：内置公平基准测试脚本，量化对比延迟、P95 尾部延迟、吞吐量、内存占用多维度指标
5. **工程化项目结构**：遵循 Python 工业级项目规范，模块化封装，支持 Docker 容器化一键部署
6. **动态 Batch 支持**：ONNX 模型支持动态批量大小，适配不同并发场景的推理需求
7. **精度无损优化**：严格对齐预处理与后处理逻辑，优化后推理结果与原生模型误差 < 1e-5
8. ## 🛠️ 技术栈

表格

| 分类 | 技术选型 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| 模型与推理 | PyTorch、TorchVision、ONNX、ONNX Runtime |
| 图像处理 | Pillow |
| 性能分析 | timeit、psutil、cProfile |
| 工程化 | Python 面向对象封装、配置分离、异常统一处理 |
| 部署支持 | Docker / Docker Compose（可扩展） |
## 📂 项目目录结构

```
resnet_onnx_project/
├── app/                        # 核心业务代码
│   ├── __init__.py
│   ├── models/                 # 推理模型封装
│   │   ├── __init__.py
│   │   ├── torch_model.py      # 原生 PyTorch 推理类
│   │   └── onnx_model.py       # ONNX Runtime 优化推理类
│   ├── api/                    # 接口路由
│   │   ├── __init__.py
│   │   └── inference.py        # 推理接口实现
│   └── utils/                  # 工具函数
│       └── __init__.py
├── models/                     # 模型文件目录
│   └── resnet50_dynamic.onnx   # 导出的动态 Batch ONNX 模型
├── tests/                      # 测试与性能脚本
│   ├── export_onnx.py          # ONNX 模型导出脚本
│   ├── precision_verify.py     # 精度对齐验证脚本
│   └── benchmark.py            # 性能对比基准测试脚本
├── docs/                       # 文档与测试报告
│   └── performance_report.md   # 性能测试详细报告
├── main.py                     # FastAPI 服务入口
├── requirements.txt            # 项目依赖清单
├── .env                        # 环境配置文件
└── README.md                   # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.9 ~ 3.11（推荐 3.10）
- Linux / WSL2 /macOS 环境（Windows 也可运行，推荐 WSL2）
- 4 核以上 CPU（GPU 可选，需安装对应版本依赖）

### 1. 克隆项目

```
git clone <你的GitHub仓库地址>
cd resnet_onnx_project
```

### 2. 创建虚拟环境并安装依赖

```
# 创建虚拟环境
python -m venv venv
# Linux/WSL 激活虚拟环境
source venv/bin/activate
# Windows PowerShell 激活
# .\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 导出 ONNX 模型（可选，models 目录已附带基础模型）

```
python tests/export_onnx.py
```

### 4. 启动服务

```
# 开发模式启动
python main.py

# 生产模式启动
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

服务启动后访问：`http://localhost:8000/docs` 即可打开 Swagger 可视化接口文档，在线调试接口。

### 5. 接口调用示例

#### 健康检查

```
curl http://localhost:8000/health
```

#### 单张图片推理

```
curl -X POST "http://localhost:8000/inference/single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

#### 批量图片推理

```
curl -X POST "http://localhost:8000/inference/batch" \
  -F "files=@test1.jpg" \
  -F "files=@test2.jpg"
```

## 📊 性能测试

### 测试环境

表格

| 环境项 | 参数 |
| --- | --- |
| 操作系统 | WSL2 Ubuntu |
| CPU | 4 核心 |
| Python 版本 | 3.10 |
| PyTorch 版本 | 2.4.0+cu121 |
| ONNX Runtime 版本 | 1.19.2 |
| 测试模型 | ResNet50 ImageNet 预训练 |
| 测试输入 | 1x3x224x224 单图推理 |

### 测试方案

- 模型仅全局加载一次，消除初始化开销对测试结果的影响
- 预热 10 轮，排除冷启动、算子编译的一次性开销
- 正式测试 200 轮，统计平均延迟、P95 尾部延迟、吞吐量、内存峰值
- 双引擎对齐 CPU 线程数，保证测试条件完全对等

### 测试结果

表格

| 指标 | 原生 PyTorch | ONNX Runtime | 优化效果 |
| --- | --- | --- | --- |
| 单次平均延迟 | 47.76 ms | 18.46 ms | 降低 61.3% |
| P95 尾部延迟 | 71.82 ms | 18.90 ms | 降低 73.7%（3.8 倍优化） |
| 稳态吞吐量 | 20.94 FPS | 54.18 FPS | 提升 158.7%（2.59 倍优化） |
| 启动加载内存 | 119.87 MB | 178.47 MB | ONNX 略高（算子缓存预分配） |
| 运行峰值内存 | 604.76 MB | 704.16 MB | 高并发场景 ONNX 内存波动更稳定 |

### 复现测试

```
python tests/benchmark.py
```

## 🎯 优化方案说明

1. **静态图转换**：将 PyTorch 动态图模型导出为 ONNX 静态图格式，消除动态图的调度开销
2. **全量图优化**：开启 ONNX Runtime 全量图优化等级，实现常量折叠、算子融合、死代码消除
3. **并行性能优化**：对齐 CPU 物理核心数配置线程数，解决多线程过载导致的性能劣化问题
4. **内存优化**：配置内存池策略，平衡推理速度与内存占用
5. **工程化优化**：单例模式全局加载模型，复用输入张量内存，减少重复分配开销

## 📌 精度验证

严格对齐 PyTorch 与 ONNX 版本的预处理、后处理逻辑，随机输入测试下：

- 平均绝对误差（MAE）< 1e-5
- 最大绝对误差 < 1e-4
- 分类结果完全一致，实现精度无损优化

验证命令：

```
python tests/precision_verify.py
```

## 🔧 进阶配置

### 切换推理引擎

修改 `app/api/inference.py` 中的导入语句，即可切换 PyTorch / ONNX Runtime 引擎，接口完全兼容：

```
# 使用 PyTorch 原生引擎
from app.models.torch_model import TorchResNetClassifier
classifier = TorchResNetClassifier()

# 使用 ONNX Runtime 优化引擎
from app.models.onnx_model import ONNXResNetClassifier
classifier = ONNXResNetClassifier()
```

### 调整 ONNX 性能参数

修改 `app/models/onnx_model.py` 中的会话配置：

- `intra_op_num_threads`：算子内并行线程数，建议设置为 CPU 物理核心数
- `enable_cpu_mem_arena`：内存池开关，关闭可降低内存占用，速度损失 < 5%
- `cpu_mem_limit`：CPU 内存池上限，适配内存受限环境

### GPU 加速支持

安装 GPU 版 ONNX Runtime 即可自动启用 CUDA 加速：

```
pip uninstall onnxruntime -y
pip install onnxruntime-gpu==1.19.2
```

## 📈 扩展方向

1. **容器化部署**：编写 Dockerfile 与 docker-compose.yml，实现服务一键容器化部署
2. **更高性能优化**：接入 TensorRT 执行提供程序，GPU 场景下进一步提升推理速度
3. **多模型支持**：扩展 YOLOv8、SAM2 等视觉模型，统一推理服务框架
4. **生产级特性**：新增限流、鉴权、日志收集、监控告警等功能
5. **并发压测**：接入 JMeter/hey 工具，测试服务高并发下的 QPS 与稳定性
6. **Triton 部署**：迁移至 NVIDIA Triton Inference Server，支持动态批处理、多模型调度

## 📄 许可证

MIT License
