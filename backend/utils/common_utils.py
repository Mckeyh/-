# ============================================================
# 【模块说明】日志工具：控制台 + 滚动文件双输出（文件固定 UTF-8，避免中文日志乱码）
# ============================================================
from pathlib import Path
import logging
import sys
from logging.handlers import RotatingFileHandler
BASE_DIR=Path(__file__).resolve().parent.parent.parent
_LOG_FILE=f"{BASE_DIR}/data/logs/app.log"
_LOGGER_CACHE={}
# 创建并配置 logger：控制台输出全部级别，文件只记 INFO 以上且按大小自动轮转
def setup_logger(
        name: str="智慧农机",
        log_file: str= None,
        level: str="INFO",
        max_bytes: int=10*1024*1024,
        backup_count: int=5,
)->logging.Logger:
    logger=logging.getLogger(name)
    logger.setLevel(getattr(logging,level.upper()))
    if logger.handlers:
        return logger
    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if log_file:
            try:
                 log_path=Path(log_file)
                 log_path.parent.mkdir(parents=True,exist_ok=True)
                 file_handler=RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                 )
                 file_handler.setLevel(logging.INFO)
                 file_formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
                 file_handler.setFormatter(file_formatter)
                 logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"进入失败: {e}")
    return logger
# 获取 logger（带缓存，同名只初始化一次）
def get_logger(name: str="智慧农机") -> logging.Logger:
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger=setup_logger(name=name,log_file=_LOG_FILE,level="INFO")
    _LOGGER_CACHE[name] = logger
    return logger
default_logger=get_logger()
    
