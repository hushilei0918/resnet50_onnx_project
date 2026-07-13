# tests/benchmark.py
import time
import numpy as np
import psutil
import os
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import onnxruntime as ort


# ===================== 通用工具函数 =====================
def get_process_memory_mb() -> float:
    """获取当前进程的内存占用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


def benchmark(predict_func, warmup: int = 10, rounds: int = 200) -> dict:
    """
    通用性能测试函数
    返回：平均延迟、P95延迟、最大延迟、吞吐量
    """
    # 预热阶段
    print(f"  预热 {warmup} 轮...")
    for _ in range(warmup):
        predict_func()
    
    # 正式测试
    print(f"  正式测试 {rounds} 轮...")
    time_list = []
    for _ in range(rounds):
        start = time.time()
        predict_func()
        cost_ms = (time.time() - start) * 1000
        time_list.append(cost_ms)
    
    time_array = np.array(time_list)
    return {
        "avg_ms": round(np.mean(time_array), 2),
        "p95_ms": round(np.percentile(time_array, 95), 2),
        "max_ms": round(np.max(time_array), 2),
        "fps": round(1000 / np.mean(time_array), 2)
    }



# ===================== 1. 原生 PyTorch 性能测试 =====================
def test_torch_performance(batch_size: int = 1):
    print("\n" + "="*60)
    print(f"【原生 PyTorch 测试 | batch_size={batch_size}】")
    
    # 记录模型加载前内存
    mem_before = get_process_memory_mb()
    
    # 加载模型
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    model.to("cpu")
    
    # 构造测试输入
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # 模型加载后内存
    mem_after = get_process_memory_mb()
    mem_usage = round(mem_after - mem_before, 2)
    print(f"  模型加载内存占用: {mem_usage} MB")
    
    # 定义推理函数
    @torch.no_grad()
    def infer():
        return model(dummy_input)
    
    # 性能测试
    perf_result = benchmark(infer, warmup=10, rounds=200)
    perf_result["memory_mb"] = mem_usage
    
    print(f"  平均延迟: {perf_result['avg_ms']} ms")
    print(f"  P95 延迟: {perf_result['p95_ms']} ms")
    print(f"  吞吐量: {perf_result['fps']} FPS")
    
    return perf_result


# ===================== 2. ONNX Runtime 性能测试 =====================
def test_onnx_performance(batch_size: int = 1):
    print("\n" + "="*60)
    print(f"【ONNX Runtime 测试 | batch_size={batch_size}】")
    
    # 记录加载前内存
    mem_before = get_process_memory_mb()
    
    # 加载模型，开启全量优化
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 24
    
    session = ort.InferenceSession(
        "./models/resnet50_dynamic.onnx",
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    
    # 构造测试输入
    dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    
    # 加载后内存
    mem_after = get_process_memory_mb()
    mem_usage = round(mem_after - mem_before, 2)
    print(f"  模型加载内存占用: {mem_usage} MB")
    
    # 定义推理函数
    def infer():
        return session.run(None, {input_name: dummy_input})
    
    # 性能测试
    perf_result = benchmark(infer, warmup=10, rounds=200)
    perf_result["memory_mb"] = mem_usage
    
    print(f"  平均延迟: {perf_result['avg_ms']} ms")
    print(f"  P95 延迟: {perf_result['p95_ms']} ms")
    print(f"  吞吐量: {perf_result['fps']} FPS")
    
    return perf_result


# ===================== 3. 对比结果汇总 =====================
def print_comparison(torch_result: dict, onnx_result: dict):
    print("\n" + "="*60)
    print("【性能对比汇总】")
    print(f"{'指标':<15} {'PyTorch':<12} {'ONNX Runtime':<15} {'提升倍数':<10}")
    print("-"*60)
    print(f"{'平均延迟(ms)':<15} {torch_result['avg_ms']:<12} {onnx_result['avg_ms']:<15} {round(torch_result['avg_ms']/onnx_result['avg_ms'], 2):<10}x")
    print(f"{'P95延迟(ms)':<15} {torch_result['p95_ms']:<12} {onnx_result['p95_ms']:<15} {round(torch_result['p95_ms']/onnx_result['p95_ms'], 2):<10}x")
    print(f"{'吞吐量(FPS)':<15} {torch_result['fps']:<12} {onnx_result['fps']:<15} {round(onnx_result['fps']/torch_result['fps'], 2):<10}x")
    print(f"{'内存占用(MB)':<15} {torch_result['memory_mb']:<12} {onnx_result['memory_mb']:<15} {round(torch_result['memory_mb']/onnx_result['memory_mb'], 2):<10}x")
    print("="*60)
    
    # 计算优化比例
    speedup = round(onnx_result['fps']/torch_result['fps'] * 100 - 100, 1)
    mem_reduce = round((1 - onnx_result['memory_mb']/torch_result['memory_mb']) * 100, 1)
    
    print(f"\n📊 核心结论：")
    print(f"1. 推理吞吐量提升 {speedup}%")
    print(f"2. 内存占用降低 {mem_reduce}%")
    
    return speedup, mem_reduce


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=== ResNet50 推理性能对比测试 ===")
    
    # 测试 batch=1 单张推理场景
    torch_res = test_torch_performance(batch_size=1)
    onnx_res = test_onnx_performance(batch_size=1)
    speedup, mem_reduce = print_comparison(torch_res, onnx_res)
    
    # 可选：测试 batch=8 批量推理场景
    # torch_batch_res = test_torch_performance(batch_size=8)
    # onnx_batch_res = test_onnx_performance(batch_size=8)
    # print_comparison(torch_batch_res, onnx_batch_res)
