# app/api/inference.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
from typing import List
#from app.models.torch_model import TorchResNetClassifier
from app.models.onnx_model import ONNXResNetClassifier
# 路由实例
router = APIRouter(prefix="/inference", tags=["推理接口"])

# 全局初始化模型（服务启动时加载一次）
#classifier = TorchResNetClassifier()
classifier = ONNXResNetClassifier()


@router.post("/single", summary="单张图片分类")
async def single_inference(file: UploadFile = File(..., description="上传图片文件")):
    """单张图片推理接口"""
    # 校验文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片格式文件")
    
    try:
        # 读取图片
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # 推理
        result = classifier.predict(image)
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "filename": file.filename,
                **result
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@router.post("/batch", summary="批量图片分类")
async def batch_inference(files: List[UploadFile] = File(..., description="批量上传图片")):
    """批量图片推理接口"""
    image_list = []
    filename_list = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            image_list.append(img)
            filename_list.append(file.filename)
        except:
            continue
    
    if not image_list:
        raise HTTPException(status_code=400, detail="没有有效的图片文件")
    
    # 批量推理
    results = classifier.predict_batch(image_list)
    
    # 补全文件名
    for i, res in enumerate(results):
        res["filename"] = filename_list[i]
    
    return {
        "code": 200,
        "message": "success",
        "total": len(results),
        "data": results
    }
