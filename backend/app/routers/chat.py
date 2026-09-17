from fastapi import APIRouter,HTTPException
from services.llm_service import llm_service
from pydantic import BaseModel
from services.rag_service import rag_service
import json
from utils.common_utils import default_logger
from fastapi.responses import StreamingResponse
from pathlib import Path



router=APIRouter(prefix="/api/chat")

class ChatRequest(BaseModel):
    message:str
    temperature:float=0.7
    max_tokens:int =512

def sse_message(data:dict)->str:
    """按 SSE 规范格式化一条消息：data: 之后需要空行分隔"""
    return f"data: {json.dumps(data,ensure_ascii=False)}\n\n"

async def generate(request:ChatRequest):
    try:
        result=rag_service.query(request.message) or {}
        sources=result.get("source") or result.get("sources") or []
        if isinstance(sources,str):
            sources=[sources]
        answer=result.get("answer","") or "未找到相关答案"
        source_names=[Path(str(s)).name for s in sources[:10]] if sources else []
        default_logger.info(f"生成完成:{answer} | 来源:{source_names}")
        # 回答正文不再拼"来源文档"，来源单独放在 source 字段里（前端按需展示）
        yield sse_message({
            "chunk":answer,
            "source":source_names,
            "done":False
        })
        yield sse_message({
            "chunk":"",
            "source":source_names,
            "done":True
        })
    except Exception as e:
        default_logger.error(f"回答失败:{e}")
        yield sse_message({
            "error":str(e),
            "done":True
        })

@router.post("/stream")
async def chat_stream(request:ChatRequest):
    default_logger.info(f"开始处理:{request.message}")
    return StreamingResponse(
        generate(request),
        media_type="text/event-stream",
        headers={"Connection":"keep-alive","Cache-Control":"no-cache"}
    )
