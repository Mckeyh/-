import threading
import asyncio
from langchain_chroma import Chroma
from config.config import settings
from utils.common_utils import default_logger
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.llm_service import llm_service


class PromptTemplate:
     SAA_EXPERT: str="""
你是一个专业的农业技术专家，拥有丰富的农作物种植，病虫害防治，土壤管理的经验，请遵循以下规则回答用户的问题：
1.回答专业，准确，基于科学知识
2.使用通俗易懂的语言，避免过于学术化
3.提供具体的操作的建议和步骤
4.如果超出你的知识范围，请如实告诉用户
5.设计农药使用，务必提醒用户注意事项
请以专业，耐心的态度，回答用户的问题。
     """

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
            self._init_chromadb()
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
         default_logger.info("所有组件初始化完成")

         return True
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
              text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
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
              context="\n\n".join([
                   doc.page_content.strip() for doc in unique_docs[:settings.K]
              ])
              if len(context)>2000:
                   context=context[:2000] + "..."
              system_prompt=PromptTemplate.SAA_EXPERT
              user_prompt=f"""
请根据以下知识库内容回答用户问题，如果知识库中没有相关信息，请如实告知。

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