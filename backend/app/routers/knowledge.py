# ============================================================
# 【模块说明】知识库接口：上传防治资料（PDF/DOCX/TXT）并向量化入库
# ============================================================
from fastapi import APIRouter,HTTPException,File,UploadFile
from pathlib import Path
import re
import uuid
from config.config import settings
from utils.common_utils import default_logger
from services.rag_service import rag_service




router = APIRouter(prefix="/api/knowledge")
# POST /api/knowledge/upload：校验类型与大小 → 落盘 → 解析分块 → 写入向量库
@router.post("/upload")
async def upload_document(file:UploadFile=File(...)):
    file_name=Path(file.filename or "upload.bin").name
    file_name=re.sub(r'[\\/:*?"<>|]+','_',file_name).strip() or "upload.bin"
    ext=Path(file_name).suffix.lower()
    if ext.lstrip('.').lower() not in settings.ALLOWED_EXTENSIONS:
        default_logger.error(f"文件扩展名错误:{ext}")
        raise HTTPException(status_code=400,detail="文件扩展名错误")
    temp_dir=Path(settings.UPLOAD_FILE_DIR)
    unique_name=f"{uuid.uuid4().hex}_{file_name}"
    save_path=temp_dir/unique_name
    try:
        content=await file.read()
        if len(content) > settings.MAX_UPLOAD_FILE_SIZE:
            default_logger.error(f"文件大小超过最大限制:{settings.MAX_UPLOAD_FILE_SIZE}")
            raise HTTPException(status_code=400,detail="文件大小超过最大限制")
        with open(save_path,"wb") as f:
            f.write(content)
        default_logger.info(f"保存文件:{save_path}")

        result=rag_service.add_document(str(save_path))


        if not result["success"]:
            default_logger.error(f"添加文件到RAG服务失败:{result['message']}")
            raise HTTPException(status_code=500,detail="添加文件到RAG服务失败")
        return  {
            "message":"文件上传成功",
            "success":True,
            "document":str(save_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        default_logger.error(f"文件上传失败:{e}")
        raise HTTPException(status_code=500,detail="文件上传失败")
