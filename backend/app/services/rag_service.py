# ============================================================
# 【模块说明】RAG 服务：文档解析入库 + 向量检索 + 交给本地大模型作答
# ============================================================
import threading
import asyncio
import json
import re
from pathlib import Path
from langchain_chroma import Chroma
from config.config import settings
from utils.common_utils import default_logger
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.llm_service import llm_service


# 系统提示词模板：设定「农田害虫识别与防治专家」人设与回答结构
class PromptTemplate:
     # 系统提示词：把大模型设定为「农田害虫识别与防治专家」，并约束回答必须覆盖
     # 「怎么防 / 用什么药 / 天敌是什么」三个维度，同时强制安全用药提示
     PEST_EXPERT: str="""
你是一位农田害虫识别与防治专家，熟悉水稻、玉米、小麦、蔬菜、果树等作物的常见害虫及其综合防治技术。
回答用户问题时请遵循以下规则：
1. 先说明该害虫的种类（常用中文名/学名）、危害特征与发生规律
2. 按三个方面给出建议：
   「怎么防」——农业防治、物理防治、生物防治等非化学手段优先
   「用什么药」——只给出农药的「有效成分通用名」（如吡虫啉、氯虫苯甲酰胺），说明适用虫态与最佳施药时期
   「天敌是什么」——列出主要天敌昆虫或病原微生物
3. 语言通俗易懂，给出可操作的具体步骤，避免过于学术化
4. 严格以「知识库内容」为依据作答：
   - 知识库没有提到的农药、天敌、害虫一律不要写；
   - 知识库没有给出具体用量、稀释倍数、安全间隔期时，直接说明「知识库未提供具体剂量，请按产品标签使用」，严禁编造任何数字、商品名或天数；
   - 不确定时明确说「知识库中没有相关信息」，不要猜测。
5. 涉及农药时务必提醒：严格按标签剂量使用、注意安全间隔期与个人防护、轮换用药避免抗性
6. 回答中用「怎么防 / 用什么药 / 天敌是什么」三个小标题分段，不要重复输出同一段内容
请以专业、耐心的态度回答用户的问题。
     """

# RAG 服务（单例）：向量库、嵌入模型、检索器三者懒加载并全局复用
class RagService: 
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
            self.embeddings=None
            self.vectorstore=None
            self.retriever=None
            self._embedding_loaded=False
            self._embedding_model_dir=None
            # 关键词表：用于「名称精确命中」的混合检索（害虫名 + 知识库小标题）
            self._index_terms=set()
            self._init_chromadb()
    # 初始化 ChromaDB 持久化向量库（首次使用注入本地嵌入模型）
    def _init_chromadb(self):
        try:
            self.vectorstore=Chroma(
                collection_name="saa_knowledge",
                embedding_function=None,
                persist_directory=settings.CHROMA_PERSIST_DIR

            )
            self.retriever=None
            default_logger.info("向量数据库初始化完成")
        except Exception as e:  
            default_logger.error(f"向量数据库初始化失败:{e}")
            self.vectorstore=None
    # 加载本地句向量模型 all-MiniLM-L6-v2（把文本转向量）
    def _load_embeddings(self):
        if self._embedding_loaded:
            return
        try:
             self.embeddings=HuggingFaceEmbeddings(
                  model_name=settings.EMBEDDING_MODEL_PATH,
                  model_kwargs={"device":settings.DEVICE},
                  encode_kwargs={"normalize_embeddings":True}
             )
             self._embedding_loaded=True
             default_logger.info("嵌入模型加载完成")
        except Exception as e: 
             default_logger.error(f"嵌入模型加载失败:{e}")
             self.embeddings=None
             self._embedding_loaded=False
    # 确保三个组件就绪（向量库 / 嵌入模型 / 检索器），并绑定嵌入函数与检索参数
    def _ensure_components(self):
         if self.vectorstore is None:
             default_logger.info("向量数据库未初始化,开始初始化")
             return False
         if not self._embedding_loaded:
              self._load_embeddings()
         if not self._embedding_loaded:
              default_logger.error("嵌入模型加载失败")
              return False
         if self.vectorstore._embedding_function is None and self.embeddings is not None:
              self.vectorstore._embedding_function=self.embeddings

         if self.retriever is None and self.embeddings is not None:
              self.retriever = self.vectorstore.as_retriever(search_kwargs={"k":max(settings.K*3,12)})
              # 重启后（向量库里已有历史文档）也要重建关键词表，否则混合检索的关键词为空
              try:
                   if not self._index_terms:
                        data=self.vectorstore.get(include=["documents"],limit=500)
                        self._refresh_index_terms(data.get("documents") or [])
              except Exception as e:
                   default_logger.warning(f"关键词表初始化失败:{e}")
         default_logger.info("所有组件初始化完成")

         return True
    # 收集「关键词表」：识别模型的害虫类别名 + 知识库里的【小标题】，
    # 提问时若命中这些词就做一次精确检索并优先注入，
    # 弥补纯向量检索对「蛴螬/蝼蛄」这类具体虫名不敏感的问题（混合检索）。
    def _refresh_index_terms(self,texts=None):
         terms=set(self._index_terms or set())
         try:
              p=Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
              if p.exists():
                   with open(p,'r',encoding='utf-8') as f:
                        for name in json.load(f):
                             if isinstance(name,str) and 2<=len(name)<=16:
                                  terms.add(name)
         except Exception as e:
              default_logger.warning(f"类别名加载失败:{e}")
         for text in texts or []:
              for seg in re.findall(r"【([^】]{1,40})】",text or ""):
                   for name in re.split(r"[、,，/]",seg):
                        name=name.strip()
                        if 2<=len(name)<=16:
                             terms.add(name)
         self._index_terms=terms
         default_logger.info(f"关键词表更新完成,共{len(terms)}个词")
    # 文档入库：按扩展名选解析器 → 分块 → 写入向量库 → 重建检索器
    def add_document(self,file_path:str)->dict:
         if not self._ensure_components():
              default_logger.error("向量化模型或向量数据库未加载，无法添加文档")
              return {
                        "success":False,
                        "trunk_count":0,
                        "message":"向量化模型或向量数据库未加载"
             }
         _file_path=Path(file_path)
         suffix=_file_path.suffix.lower()
         loader=None

         try:
              if suffix==".pdf":
                   loader=PyPDFLoader(file_path)
              elif suffix==".docx":
                   loader=Docx2txtLoader(file_path)
              elif suffix==".txt":
                   encoding=["utf-8","gbk","gb2312",'latin-1']
                   for enc in encoding:
                        try:
                             loader=TextLoader(file_path,encoding=enc)
                             loader.load()
                             break
                        except Exception as e:
                             default_logger.error(f"编码{enc}加载失败:{e}")
                             continue
                   if loader is None:
                        default_logger.error(f"所有编码都加载失败:{file_path}")
                        return{
                             "success":False,
                             "trunk_count":0,
                             "message":"无法解析txt文件"
                        }    
              elif suffix==".doc":
                    try:
                        loader = TextLoader(file_path,encoding="latin-1")
                        loader.load()
                    except Exception as e:
                        default_logger.error(f"doc文件加载失败:{e}")
                        return {
                        "success":False,
                        "trunk_count":0,
                        "message":"doc文件加载失败"
                    }
              else:
                   default_logger.error(f"不支持文件格式:{suffix}")
                   return {
                        "success":False,
                        "trunk_count":0,
                        "message":f"不支持文件格式:{suffix}"
                   }
              documents=loader.load()
              if not documents:
                   default_logger.error(f"文档加载失败:{file_path}")
                   return{
                        "success":False,
                        "trunk_count":0,
                        "message":"文档加载失败"
                    }
              # 分块策略：优先按「空行」切（本知识库每个害虫段落之间用空行分隔），
              # 段内按行/句继续切，块控制在 350 字左右。
              # 注意：句向量模型 bge-small-zh-v1.5 最大输入 512 token（中文约 1 字≈1 token），
              # 块过大（如 800 字）会被截断，只剩共有的标题部分参与计算，导致各块互相区分不开、检索跑偏。
              text_splitter=RecursiveCharacterTextSplitter(
                   chunk_size=350,
                   chunk_overlap=60,
                   separators=["\n\n","\n","。","；","，"]
              )
              chunks=text_splitter.split_documents(documents)
              if not chunks:
                   default_logger.error(f"文档分割失败:{file_path}")
                   return{
                            "success":False,
                            "trunk_count":0,
                            "message":"文档分割失败"
                    }
              for chunk in chunks:
                   chunk.metadata['source']=file_path
              self.vectorstore.add_documents(chunks)
              self.retriever=self.vectorstore.as_retriever(search_kwargs={"k":max(settings.K*3,12)})
              self._refresh_index_terms([c.page_content for c in chunks])
              default_logger.info(f"文档{file_path}添加到向量数据库,共{len(chunks)}个段落")
              return{
                        "success":True,
                        "trunk_count":len(chunks),
                        "message":"文档添加成功"
                }
         except Exception as e:
              default_logger.error(f"添加文档{file_path}失败:{e}")
              return{
                        "success":False,
                        "trunk_count":0,
                        "message":f"添加文档{file_path}失败"
                }
    # 问答主流程：检索相关片段（按内容去重）→ 拼上下文 → 调大模型 → 返回答案与来源
    def query(self,question:str)->dict:
         if not self._ensure_components():
              default_logger.error("向量化模型未加载或向量数据库未初始化，无法查询")
              return{
                   "answer":"RAG无法查询",
                   "source":[]
              }
         try:
              if self.retriever is None:
                   default_logger.error("检索器未初始化，无法查询")
                   return{
                        "answer":"检索器未初始化",
                        "source":[]
                   }
              docs=self.retriever.get_relevant_documents(question)
              # 混合检索：问题里出现知识库/类别表里的害虫名时，用该名称再精确检索一次并放到最前，
              # 保证「问哪种虫就答哪种虫」，再由后面的向量结果补充通用防治原则
              matched=[t for t in (self._index_terms or set()) if t in question]
              if matched:
                   boost=[]
                   for name in matched[:3]:
                        try:
                             boost.extend(self.vectorstore.similarity_search(name,k=2))
                        except Exception as e:
                             default_logger.warning(f"关键词检索失败:{name}:{e}")
                   if boost:
                        default_logger.info(f"关键词命中{matched[:3]},优先注入{len(boost)}段")
                        docs=boost+docs
              default_logger.info(f"查询到{len(docs)}个相关文档落")
              source=[]
              seen=set()
              unique_docs=[]
              for i,doc in enumerate(docs):
                   text=(doc.page_content or "").strip()
                   if not text or text in seen:
                        continue
                   seen.add(text)
                   unique_docs.append(doc)
                   src=doc.metadata.get('source')
                   if src and str(src) not in source:
                        source.append(str(src))
              default_logger.info(f"检索到{len(docs)}段,去重后{len(unique_docs)}段")
              if not unique_docs:
                   return{
                        "answer":"查询到0个相关文档",
                        "source":[]
                   }
              # 送入大模型的段落数：块变小后多给几段，保证「怎么防/用什么药/天敌」三方面都有依据
              context="\n\n".join([
                   doc.page_content.strip() for doc in unique_docs[:max(settings.K,5)]
              ])
              if len(context)>2000:
                   context=context[:2000] + "..."
              system_prompt=PromptTemplate.PEST_EXPERT
              user_prompt=f"""
请根据以下知识库内容回答用户问题，如果知识库中没有相关信息，请如实告知，不要编造。
回答时请围绕「怎么防治」「用什么药」「天敌是什么」三方面展开；
农药只写有效成分通用名，知识库未给出的用量、稀释倍数、安全间隔期一律不要编造。

知识库内容：
{context}

用户问题：
{question}

回答：
"""
              answer=llm_service.generate(
                   system_prompt=system_prompt,
                   user_prompt=user_prompt,
                   temperature=0.5,
                   max_tokens=512,
              )
              if "模型未加载" in answer:
                   default_logger.error("LLM模型未加载，无法回答问题")
                   return{
                        "answer":f"SAA_LLM模型未加载，以下是检索到的相关内容:{context}",
                        "source":source
                   }


              default_logger.info(f"查询到{len(source)}个相关文档，经过LLM处理，回答为{source}")
              return{
                   "answer":answer,
                   "source":source
              }






         except Exception as e:
              default_logger.error(f"查询失败:{e}")
              return {
                   "answer":f"查询失败:{e}",
                   "source":[]
              }
rag_service=RagService()