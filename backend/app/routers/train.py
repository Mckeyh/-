from fastapi import WebSocket,APIRouter,HTTPException,Request,WebSocketDisconnect
import asyncio
from services.train_service import train_service
from utils.common_utils import default_logger
from config.config import settings
from pathlib import Path
import shutil
import base64
from PIL import Image
import io
import re
class ConnectManager:
    def __init__(self):
        self.active_connections:list[WebSocket]=[]
    async def connect(self,websocket:WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self,websocket:WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def broadcast(self,message:dict):
        if not self.active_connections:
            return
        dead=[]
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)
manager=ConnectManager()


def broadcast_status(status:dict):
    loop=train_service._main_loop
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(status),loop)
train_service.set_broadcat_callback(broadcast_status)

router=APIRouter(prefix="/api/train")
@router.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json(train_service.get_status())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        default_logger.info("ws客户端断开连接")
    except Exception as e:
        default_logger.error(f"处理ws消息失败:{e}")
    finally:
        manager.disconnect(websocket)
@router.get("/status")
async def get_train_status():
    return train_service.get_status()

@router.post("/stop")
async def stop_trainning():
    if train_service.status != settings.TRAIN_STATUS_RUNNING:
        raise HTTPException(status_code=400,detail="训练未进行")
    success=train_service.request_stop()
    if not success:
        raise HTTPException(status_code=400,detail="终止训练失败")
    return {"messsage":"训练已终止"}
@router.post("/finetune")
async def finetune_resnet50(request:Request):
    content_type=request.headers.get("content-type","")
    if "application/json" in content_type:
        body=await request.json()
        label=body.get("label")
        clear_old=body.get("clear_old",False)
        images_base64=body.get("images",[])
        if not images_base64:
            raise HTTPException(status_code=400,detail="请上传图片")
        if not label:
            raise HTTPException(status_code=400,detail="请上传标签")
        label=str(label).strip()
        if not re.match(r'^[\w\u4e00-\u9fa5-]{1,32}$', label):
            raise HTTPException(status_code=400,detail="标签只能包含中文/字母/数字/下划线/短横线,长度1-32")
        default_logger.info(f"开始微调ResNet50,标签:{label},是否清除旧模型:{clear_old}")
        train_dir=Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
        if clear_old:
            default_logger.info("清除旧数据")
            for item in train_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        class_dir=train_dir/label
        class_dir.mkdir(parents=True,exist_ok=True)
        saved_count=0
        for idx,b64_data in enumerate(images_base64):
            try:
                if "," in b64_data:
                    b64_data=b64_data.split(",",1)[-1]
                b64_data += "="*(4-len(b64_data)%4) if len(b64_data)% 4 else ""
                img_bytes=base64.b64decode(b64_data)
                img=Image.open(io.BytesIO(img_bytes))
                img.verify()
                ext=".jpg"
                if img.format=="PNG":
                    ext=".png"
                elif img.format=="GIF":
                    ext=".gif"
                elif img.format=="WEBP":
                    ext=".webp"
                safe_name=f"{label}_{idx+1}{ext}"
                file_path=class_dir / safe_name
                with open(file_path,"wb") as f:
                    f.write(img_bytes)
                saved_count+=1
            except Exception as e:
                default_logger.error(f"保存图片失败:{e}")

        default_logger.info(f"保存成功,共有{saved_count}张图片到目录:{class_dir}")    
        success=train_service.start_training(train_dir,settings.FULL_EPOCHS)
        if not success:
            raise HTTPException(status_code=409,detail="启动失败")
        return{
            "message":"训练已启动",
            "status":settings.TRAIN_STATUS_RUNNING
        }
    else:
        raise HTTPException(status_code=415,detail="请上传json")