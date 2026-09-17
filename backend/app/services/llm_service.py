import threading
from typing import Dict, List,Optional
import torch
from llama_cpp import Llama
from config.config import settings
from utils.common_utils import default_logger





class LLMService:
    _instance=None
    _lock=threading.Lock()
        
        
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                            cls._instance=super().__new__(cls)
        return cls._instance
        
        
        
        
    def __init__(self):
        if hasattr(self,'_initialized'):
            return
        self._initialized=True
        self.model_path=settings.LLM_MODEL_PATH
        self.llm=None
        self._loaded=False
    def _load_model(self):
        if self._loaded:
            return
        n_gpu_layers=settings.LLM_GPU_LAYERS
        try:
             self.llm=Llama(
                model_path=self.model_path,
                n_ctx=settings.LLM_CONTEXT_SIZE,
                n_gpu_layers=n_gpu_layers,
                n_threads=settings.LLM_THREADS,
                verbose=False
             )
             self._loaded=True
             default_logger.info(f"模型加载完成(设备:{settings.DEVICE})")

        except Exception as e:
            default_logger.error(f"模型加载失败:{e}")
            self.llm=None
            self._loaded=False
    def is_ready(self)->bool:
        return self._loaded and self.llm is not None
    def _ensure_model(self)->bool:
        if not self.is_ready():
            default_logger.error("模型未加载，重新载入")
            self._load_model()
        if not self.is_ready():
            default_logger.error("模型未加载完成，请检查模型路径是否正确")
            return False
        return True
    def chat(self,messages:List[Dict[str,str]],temperature:float=0.7,max_tokens:int=512,top_p:float=0.9)->str:
        """非流式对话：一次性返回完整回答"""
        if not self._ensure_model():
            return "模型未加载完成，请检查模型路径是否正确"
        def stream_generator():
            try:
                response=self.llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    repeat_penalty=1.1,
                    stream=False,
                    stop=["<|im_end|>","<|endoftext|>"]
                )
                choices=response.get('choices') or []
                if choices:
                    return choices[0]['message']['content']
                default_logger.error("模型返回为空")
                return "模型返回为空,请重试"
            except Exception as e:
                default_logger.error(f"LLM模型调用失败:{e}")
                return f"模型调用失败:{str(e)},请重试"
        return stream_generator()
    def chat_stream(self,messages:List[Dict[str,str]],temperature:float=0.7,max_tokens:int=512,top_p:float=0.9):
        """流式对话：逐段产出模型生成的内容"""
        if not self._ensure_model():
            yield "模型未加载完成，请检查模型路径是否正确"
            return
        try:
             response=self.llm.create_chat_completion(
                  messages=messages,
                  temperature=temperature,
                  max_tokens=max_tokens,
                  top_p=top_p,
                  repeat_penalty=1.1,
                  stream=True,
                  stop=["<|im_end|>","<|endoftext|>"]
             )
             for chunk in response:
                  choices=chunk.get('choices') or []
                  if not choices:
                       continue
                  delta=choices[0].get('delta') or {}
                  content=delta.get('content','')
                  if content:
                       yield content
        except Exception as e:
             default_logger.error(f"LLM流式调用失败:{e}")
             yield f"模型调用失败:{str(e)},请重试"
    def generate(self,user_prompt:str,system_prompt:Optional[str]=None,temperature:float=0.7,max_tokens:int=512)->str:
         messages=[]
         if system_prompt:
              messages.append({
                   "role":"system",
                   "content":system_prompt
              })
         messages.append({
              "role":"user",
              "content":user_prompt
         })
         return self.chat(messages,temperature,max_tokens)

llm_service=LLMService()
