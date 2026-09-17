# ============================================================
# 【模块说明】全局配置：目录路径、模型路径、训练与推理超参、设备（CPU/GPU）自动检测
# ============================================================
from pathlib import Path
from pydantic_settings import BaseSettings
import torch
from utils.common_utils import default_logger



BASE_DIR=Path(__file__).resolve().parent.parent.parent

# 配置类（pydantic-settings）：字段带类型标注，可通过环境变量或 .env 覆盖默认值
class Settings(BaseSettings):
    DATA_DIR: str = f"{BASE_DIR}/data"
    MODELS_DIR: str = f"{BASE_DIR}/data/models"
    MODELS_PTH_DIR: str = f"{BASE_DIR}/pth"
    MODELS_GGUF_DIR: str = f"{MODELS_DIR}/gguf"
    MODELS_SAFETENSORS_DIR: str = f"{MODELS_DIR}/safetensors"
    RESNET50_MODEL_ID: str = "microsoft/resnet-50"
    RESNET50_MODEL_PATH: str = f"{MODELS_SAFETENSORS_DIR}/{RESNET50_MODEL_ID}"
    RESNET50_FINETUNED_PTH_DIR: str = f"{MODELS_PTH_DIR}/resnet_50_finetuned"
    RESNET50_FINETUNED_PTH_PATH: str = f"{RESNET50_FINETUNED_PTH_DIR}/resnet_50_finetuned.pth"
    JSON_DIR: str = f"{BASE_DIR}/json"
    RESNET50_FINETUNED_JSON_DIR: str = f"{JSON_DIR}/resnet_50_finetuned"
    RESNET50_FINETUNED_CLASSNAMES_PATH: str = f"{RESNET50_FINETUNED_JSON_DIR}/class_names.json"
    UPLOAD_DATASET_UNZIPED_DIR:str=f"{DATA_DIR}/upload/dataset/unzipped"
    DEVICE: str = "cpu"
    LLM_GPU_LAYERS: int = 0
    GPU_MEMORY_THRESHOLD_GB: float =5.9
    BATCH_SIZE: int = 16
    CONFIDENCE_THRESHOLD: float = 0.25
    FULL_EPOCHS: int = 10
    TRAIN_STATUS_RUNNING:str = "running"
    TRAIN_STATUS_IDLE:str="idle"
    TRAIN_STATUS_FAILED:str="failed"
    TRAIN_STATUS_DONE:str="done"

    CHROMA_PERSIST_DIR:str=f"{DATA_DIR}/chroma_db"
    EMBEDDING_MODEL_ID:str="sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_PATH:str=f"{MODELS_SAFETENSORS_DIR}/{EMBEDDING_MODEL_ID}"

    K:int=4

    ALLOWED_EXTENSIONS:list=["pdf","docx","doc","txt"]

    UPLOAD_FILE_DIR:str=f"{DATA_DIR}/uploads/files"
    MAX_UPLOAD_FILE_SIZE:int=1024*1024*10

    LLM_MODEL_ID:str="LoveSeaW/Qwen2.5-1.5b-instruct-gguf"
    LLM_MODEL_FILE:str="qwen2.5-1.5b-instruct-q4_k_m.gguf"
    LLM_MODEL_PATH:str=f"{MODELS_GGUF_DIR}/{LLM_MODEL_ID}/{LLM_MODEL_FILE}"
    LLM_CONTEXT_SIZE:int=4096
    LLM_THREADS:int=4
    # 检测 GPU：显存达到阈值才把 DEVICE 设为 cuda，并设置全局默认设备
    def _detect_and_configure_device(self):
        self.DEVICE='cpu'
        self.LLM_GPU_LAYERS=0
        if torch.cuda.is_available():
            device_idx=torch.cuda.current_device()
            total_memory_bytes=torch.cuda.get_device_properties(device_idx).total_memory
            total_memory_gb=total_memory_bytes/(1024**3)
            if total_memory_gb>=self.GPU_MEMORY_THRESHOLD_GB:
                self.DEVICE='cuda'
                self.LLM_GPU_LAYERS=-1
                default_logger.info(f"GPU detected with {total_memory_gb:.2f} GB")
            else:
                default_logger.warning(f"显存不足，使用CPU")
        else:
            default_logger.warning("CUDA不可用, using CPU")
        torch.set_default_device(self.DEVICE)
        default_logger.info(f"默认设备:{self.DEVICE}")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._detect_and_configure_device()

settings=Settings()

for d in [
    settings.MODELS_DIR,
    settings.RESNET50_FINETUNED_PTH_DIR,
    settings.RESNET50_FINETUNED_JSON_DIR,

    settings.UPLOAD_DATASET_UNZIPED_DIR,
    settings.CHROMA_PERSIST_DIR,
    settings.UPLOAD_FILE_DIR
]:
    Path(d).mkdir(parents=True,exist_ok=True)
default_logger.info(f"配置初始化完成,默认设备:{settings.DEVICE},LLM_GPU_LAYERS:{settings.LLM_GPU_LAYERS}")  