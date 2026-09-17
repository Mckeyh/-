# ============================================================
# 【模块说明】害虫识别接口：接收前端拍照的 base64 图片，返回识别结果
# ============================================================
from fastapi import APIRouter,HTTPException
import base64
from utils.common_utils import default_logger
from fastapi import Request
from services.classify_service import classify_services

router=APIRouter(prefix="/api/classify")

# 把 base64 图片字符串（允许带 data:image/xxx;base64, 前缀）解码为字节
def decode_base_image(base64_str:str)->bytes:
    if base64_str.startswith("data:image"):
        base64_str=base64_str.split(",",1)[-1]

    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        default_logger.error(f"base64字符串解码失败:{e}")
        raise ValueError(f"base64字符串解码失败:{e}")
# POST /api/classify/ 害虫识别：解析 JSON → 推理 → 返回 top1 与 top5 及置信度
@router.post("/")
async def classify_image(request:Request):
    content_type=request.headers.get("content-type","")
    try:
        if "application/json" in content_type:
            body=await request.json()
            image_data=body.get("image")
            if not image_data:
                default_logger.error("未上传图片")
                raise HTTPException(status_code=400,detail="请上传图片")
            img_bytes=decode_base_image(image_data)
            result=classify_services.predict(img_bytes)
            return result
        else:
            default_logger.error("不支持的content-type")
            raise HTTPException(status_code=400,detail="不支持的content-type")
    except HTTPException:
        raise
    except Exception as e:
        default_logger.error(f"图片处理失败:{e}")
        raise HTTPException(status_code=500,detail="图片分类失败")
