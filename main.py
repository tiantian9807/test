from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import io
import os
import requests
from PIL import Image
from ultralytics import YOLO
import torch.serialization

# 导入需要的安全全局类（解决PyTorch 2.6+加载问题）
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import Bottleneck, SPPF, C2f, DFL
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from torch.nn.modules.container import Sequential, ModuleList
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.loss import BCEWithLogitsLoss
from dill._dill import _load_type

# 添加安全全局变量（允许PyTorch加载模型中的类）
torch.serialization.add_safe_globals([
    DetectionModel, Bottleneck, SPPF, Concat, C2f,
    MaxPool2d, Upsample, Conv2d, BatchNorm2d, SiLU,
    _load_type, Sequential, Conv, ModuleList, Detect,
    DFL, IterableSimpleNamespace, v8DetectionLoss, BCEWithLogitsLoss
])

# 初始化FastAPI应用
app = FastAPI(title="YOLO Label Studio API")

# 配置CORS（允许Label Studio跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定Label Studio的具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型下载配置
MODEL_URL = "https://raw.githubusercontent.com/tiantian9807/my-yolo-weights/main/best.pt"
MODEL_PATH = "best.pt"

# 下载模型（服务启动时执行）
def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"开始下载模型: {MODEL_URL}")
        try:
            response = requests.get(MODEL_URL, timeout=30)
            response.raise_for_status()  # 检查下载是否成功
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print(f"模型下载完成，保存至: {MODEL_PATH}")
        except Exception as e:
            print(f"模型下载失败: {str(e)}")
            raise  # 下载失败时终止服务启动

# 加载模型（服务启动时执行）
download_model()
model = YOLO(MODEL_PATH)  # 加载下载好的模型


# 定义Label Studio请求格式
class LabelStudioRequest(BaseModel):
    tasks: List[Dict[str, Any]]
    version: str = "1.0.0"

# 定义Label Studio响应格式（预测结果）
def format_prediction(result, task_id: str) -> Dict[str, Any]:
    """将YOLO预测结果转换为Label Studio兼容格式"""
    predictions = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 边界框坐标 (xmin, ymin, xmax, ymax)
        score = box.conf[0].item()  # 置信度
        cls = int(box.cls[0].item())  # 类别ID
        cls_name = model.names[cls]  # 类别名称（从模型中获取）

        # Label Studio的边界框格式：[xmin, ymin, width, height]（相对坐标，0-1范围）
        img_width, img_height = result.orig_shape[1], result.orig_shape[0]
        x = x1 / img_width
        y = y1 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        predictions.append({
            "value": {
                "x": x * 100,  # 转换为百分比
                "y": y * 100,
                "width": width * 100,
                "height": height * 100,
                "rotation": 0,
                "rectanglelabels": [cls_name]
            },
            "score": score,
            "id": f"pred_{len(predictions)}",
            "from_name": "label",  # 需与Label Studio标注配置中的"from_name"一致
            "to_name": "image",    # 需与Label Studio标注配置中的"to_name"一致
            "type": "rectanglelabels"
        })

    return {
        "task_id": task_id,
        "result": predictions,
        "score": max([p["score"] for p in predictions]) if predictions else 0.0
    }


# 健康检查接口
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    return {"status": "healthy"}

# 预测接口（供Label Studio调用）
@app.post("/predict", response_model=List[Dict[str, Any]])
async def predict(request: LabelStudioRequest):
    try:
        results = []
        for task in request.tasks:
            task_id = task["id"]
            # 从Label Studio任务中提取base64编码的图片
            image_data = task["data"]["image"].split(",")[-1]  # 去除前缀 "data:image/jpeg;base64,"
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # 用YOLO模型预测
            yolo_results = model(image, conf=0.3)  # conf=0.3 过滤低置信度结果

            # 格式化结果并添加到响应
            formatted = format_prediction(yolo_results[0], task_id)
            results.append(formatted)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# 启动服务（仅本地运行时使用）
if __name__ == "__main__":
    import uvicorn
    # 部署时使用环境变量PORT（如Render），本地默认9090
    port = int(os.getenv("PORT", 9090))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)