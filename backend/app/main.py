import multiprocessing
multiprocessing.freeze_support()
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_APP_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gc
import asyncio
from routers import test, test2,train,classify,knowledge,chat
from utils.common_utils import default_logger
from config.config import settings
from services.classify_service import load_resnet50_from_local_safetensors
from safetensors.torch import load_file
from services.classify_service import ClassifyService
from services.train_service import train_service
from pathlib import Path
app=FastAPI(
    title="智慧农技助手",
    version="1.0.0",
    description="智慧农技助手API接口"

)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup_event():
    #load_resnet50_from_local_safetensors()
    #ClassifyService().fintune(Path(settings.UPLOAD_DATASET_UNZIPED_DIR))
    train_service.set_main_loop(asyncio.get_running_loop())
    default_logger.info("应用启动完成")


@app.on_event("shutdown")
async def shutdown_event():
    gc.collect()
    default_logger.info("关闭应用完成")



#app.include_router(test.router)
#app.include_router(test2.router)
app.include_router(train.router)
app.include_router(classify.router)
app.include_router(knowledge.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)