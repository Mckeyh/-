from fastapi import APIRouter,HTTPException
import base64
from utils.common_utils import default_logger
from fastapi import Request
from services.classify_service import classify_services

router=APIRouter(prefix="/api/classify")

def decode_base_image(base64_str:str)->bytes:
    if base64_str.startswith("data:image"):
        base64_str=base64_str.split(",",1)[-1]

    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        default_logger.error(f"base64字符串解码失败:{e}")
        raise ValueError(f"base64字符串解码失败:{e}")
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
