# app/models/torch_model.py
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
from typing import List


class TorchResNetClassifier:
    """原生 PyTorch 版本 ResNet50 分类器，作为性能基线"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式：全局只加载一次模型"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("[PyTorch] 正在加载 ResNet50 预训练模型...")
        
        # 1. 加载预训练模型与权重
        self.weights = ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=self.weights)
        
        # 2. 切换评估模式（必须！推理专用）
        self.model.eval()
        
        # 3. 自动选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 4. 官方预处理管道（和训练完全一致）
        self.preprocess = self.weights.transforms()
        
        # 5. ImageNet 类别名称
        self.class_names = self.weights.meta["categories"]
        
        self._initialized = True
        print(f"[PyTorch] 模型加载完成，运行设备: {self.device}")
    
    @torch.no_grad()  # 关闭梯度计算，节省显存+提速
    def predict(self, image: Image.Image) -> dict:
        """单张图片推理"""
        # 1. 预处理
        image = image.convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 2. 推理
        output = self.model(input_tensor)
        
        # 3. 后处理：softmax 转概率
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probs, dim=0)
        
        return {
            "class_name": self.class_names[top_idx.item()],
            "confidence": round(top_prob.item(), 4),
            "class_id": top_idx.item()
        }
    
    @torch.no_grad()
    def predict_batch(self, image_list: List[Image.Image]) -> List[dict]:
        """批量图片推理"""
        # 1. 批量预处理
        tensor_list = []
        for img in image_list:
            img = img.convert("RGB")
            tensor_list.append(self.preprocess(img))
        
        # 2. 拼接 batch
        input_batch = torch.stack(tensor_list, dim=0).to(self.device)
        
        # 3. 批量推理
        output = self.model(input_batch)
        
        # 4. 批量后处理
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.max(probs, dim=1)
        
        # 组装结果
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "class_name": self.class_names[idx.item()],
                "confidence": round(prob.item(), 4)
            })
        
        return results
