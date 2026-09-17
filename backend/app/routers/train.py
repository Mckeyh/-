# ============================================================
# 【模块说明】训练接口 + WebSocket 进度推送：样本上传、启动/停止训练、状态查询与实时广播
# ============================================================
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
# WebSocket 连接管理器：维护所有在线客户端，用于广播训练进度
class ConnectManager:
    def __init__(self):
        self.active_connections:list[WebSocket]=[]
    # 接受 WebSocket 连接并登记到连接池
    async def connect(self,websocket:WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    # 把指定连接从连接池移除
    def disconnect(self,websocket:WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    # 向所有在线连接广播消息（发送失败的连接会被自动剔除，防止连接池无限增长）
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


# 训练线程的状态回调：用 run_coroutine_threadsafe 投递回主事件循环再广播（跨线程安全）
def broadcast_status(status:dict):
    loop=train_service._main_loop
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(status),loop)
train_service.set_broadcat_callback(broadcast_status)

router=APIRouter(prefix="/api/train")
# WS /api/train/ws：连上先推一次当前状态，之后保持连接；断开时清理连接池
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
# GET /api/train/status：返回当前训练状态（WebSocket 断线时前端可用它兜底同步）
@router.get("/status")
async def get_train_status():
    return train_service.get_status()

# POST /api/train/stop：请求停止训练（只置标志，训练线程会在下一个 epoch 边界退出）
@router.post("/stop")
async def stop_trainning():
    if train_service.status != settings.TRAIN_STATUS_RUNNING:
        raise HTTPException(status_code=400,detail="训练未进行")
    success=train_service.request_stop()
    if not success:
        raise HTTPException(status_code=400,detail="终止训练失败")
    return {"messsage":"训练已终止"}
# POST /api/train/finetune：上传样本图片 → 按类别存盘 → 启动后台微调（可清除旧数据）
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
        # IP102 的类别名可能是学名（含空格、点、括号，如 "Chilo suppressalis (Walker)"），
        # 因此放宽字符集；字符集里不含 / 和 \，并单独拦截 "." 与 ".."，
        # 避免 label 被当作路径片段写到数据集目录之外
        if label in ('.', '..') or not re.match(r"^[\w\u4e00-\u9fa5 \-\.\(\)',]{1,48}$", label):
            raise HTTPException(status_code=400,detail="标签只能是中文/字母/数字/空格/下划线/短横线/点/括号,长度1-48")
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