# app/models/onnx_model.py
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from typing import List


class ONNXResNetClassifier:
    """ONNX Runtime 优化版 ResNet50 分类器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, onnx_path: str = "./models/resnet50_dynamic.onnx", use_gpu: bool = False):
        if self._initialized:
            return
        
        print("[ONNX Runtime] 正在加载 ONNX 模型...")
        
        # 1. 配置执行提供程序（CPU/GPU）
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        # 2. 配置会话选项：开启全量图优化，最大化性能
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU 线程数，根据你的CPU核心数调整，一般设为物理核心数
        sess_options.intra_op_num_threads = 24
        
        # 3. 创建推理会话
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # 4. 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 5. 预处理管道：和 PyTorch 版本完全一致，保证结果对齐
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 6. 类别名称：和 PyTorch 一致
        self.class_names = ResNet50_Weights.DEFAULT.meta["categories"]
        
        self._initialized = True
        print(f"[ONNX Runtime] 模型加载完成，优化级别: 全量优化")
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """numpy 实现 softmax，和 PyTorch 结果对齐"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def predict(self, image: Image.Image) -> dict:
        """单张图片推理，接口和 PyTorch 版完全一致"""
        # 1. 预处理
        image = image.convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_np = input_tensor.numpy().astype(np.float32)
        
        # 2. ONNX 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_np})
        logits = outputs[0][0]  # 取第一个batch的结果
        
        # 3. 后处理
        probs = self._softmax(logits)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        
        return {
            "class_name": self.class_names[top_idx],
            "confidence": round(top_prob, 4),
            "class_id": top_idx
        }
    
    def predict_batch(self, image_list: List[Image.Image]) -> List[dict]:
        """批量图片推理，接口和 PyTorch 版完全一致"""
        # 1. 批量预处理
        tensor_list = []
        for img in image_list:
            img = img.convert("RGB")
            tensor_list.append(self.preprocess(img).numpy())
        
        # 2. 拼接 batch
        input_batch = np.stack(tensor_list, axis=0).astype(np.float32)
        
        # 3. 批量推理
        outputs = self.session.run([self.output_name], {self.input_name: input_batch})
        logits = outputs[0]
        
        # 4. 批量后处理
        probs = self._softmax(logits)
        top_indices = np.argmax(probs, axis=1)
        top_probs = np.max(probs, axis=1)
        
        # 组装结果
        results = []
        for idx, prob in zip(top_indices, top_probs):
            results.append({
                "class_name": self.class_names[int(idx)],
                "confidence": round(float(prob), 4)
            })
        
        return results
