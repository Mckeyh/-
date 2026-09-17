import threading
import asyncio
from typing import Callable, Optional
from utils.common_utils import default_logger
from pathlib import Path
from config.config import settings
from services.classify_service import classify_services


class TrainService:
    _instance=None
    _lock=threading.Lock()


    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
             cls._instance=super().__new__(cls)
        return cls._instance




    def __init__(self):
        if hasattr(self,'_initialized'):
            return
        self._initialized=True
        self.status=settings.TRAIN_STATUS_IDLE
        self.logs=[]
        self.result=None
        self.current_task=None
        self.stop_requested=False
        self._brodcast_callback:Optional[Callable]=None
        self._main_loop:Optional[asyncio.AbstractEventLoop]=None
        self._epochs=settings.FULL_EPOCHS
        self._data_dir=Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
    def set_main_loop(self,main_loop:asyncio.AbstractEventLoop):
        self._main_loop=main_loop
    def set_broadcat_callback(self,callback=None):
        self._brodcast_callback=callback
    def _broadcast(self):
        if self._brodcast_callback:
            try:
                self._brodcast_callback(self.get_status())
            except Exception as e:
                default_logger.info(f"广播训练状态失败:{e}")
    def get_status(self):
        return {
            "status":self.status,
            "result":self.result
        }
    def start_training(self, data_dir: Path, epochs: Optional[int] = 10):
        if self.status==settings.TRAIN_STATUS_RUNNING:
            default_logger.info("训练进行中")
            return False
        self.status=settings.TRAIN_STATUS_RUNNING
        self.result=None
        self.stop_requested=False
        if epochs is None:
            self._epochs=settings.FULL_EPOCHS
        else:
            self._epochs=epochs
        if data_dir is None:
            self._data_dir=Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
        else:
            self._data_dir=data_dir
        self._broadcast()
        self.current_task=threading.Thread(target=self._run,daemon=True)
        self.current_task.start()
        return True
    def request_stop(self):
        if self.status==settings.TRAIN_STATUS_RUNNING:
            self.stop_requested=True
            default_logger.info("终止训练")
            return True
        return False
    def _run(self):
        default_logger.info(f"开始训练模型,支持{self._epochs}个epoch")
        try:
            success=classify_services.fintune(date_dir=self._data_dir,epoch=self._epochs,stop_check=lambda x: self.stop_requested)
            if self.stop_requested:
                self.status=settings.TRAIN_STATUS_FAILED
                self.result={"success":False,"error":"中途结束训练"}
                default_logger.info("中途结束训练")
            elif success:
                self.status=settings.TRAIN_STATUS_DONE
                self.result={"success":True,"message":"训练完成"}
                default_logger.info("训练完成")
            else:
                self.status=settings.TRAIN_STATUS_FAILED
                self.result={"success":False,"error":"训练失败"}
                default_logger.info("训练失败")
        except Exception as e:
            self.status=settings.TRAIN_STATUS_FAILED
            self.result={"success":False,"error":str(e)}
            default_logger.warning(f"训练模型失败:{e}")
        finally:
            self._broadcast()
train_service=TrainService()



