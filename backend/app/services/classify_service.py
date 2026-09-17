# ============================================================
# 【模块说明】害虫识别核心服务：模型构建、迁移学习微调、推理预测
# ============================================================
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import re
from utils.common_utils import default_logger
from config.config import settings
from pathlib import Path
from safetensors.torch import load_file
import torchvision.models as models
import threading
import torchvision.transforms as transforms
import gc
import json
from torchvision.models import ResNet50_Weights
from torchvision import datasets
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy
import io
from PIL import Image


# 标签平滑交叉熵损失：缓解过拟合，改善小样本类别表现
class LabelSmoothingCrossEntropy(nn.Module):
     def __init__(self, smoothing:float=0.1):
          super().__init__()
          self.smoothing=smoothing
     def forward(self,pred,target):
          n_classes=pred.size(1)
          one_hot=torch.zeros_like(pred).scatter(1,target.unsqueeze(1),1)
          one_hot=one_hot*(1-self.smoothing)+self.smoothing/n_classes
          log_prob=F.log_softmax(pred,dim=1)

          loss=-(one_hot*log_prob).sum(dim=1).mean()
          return loss








# 余弦分类头：特征归一化后用余弦相似度分类，类别数可动态扩展
class ConsineClassifier(nn.Module):
    def __init__(self, in_features:int, num_classes:int, scale:float=1.0):
        super().__init__()
        self.in_features=in_features
        self.num_classes=num_classes
        self.scale=nn.Parameter(torch.tensor(scale,dtype=torch.float))
        self.weight=nn.Parameter(torch.empty(num_classes, in_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
    def forward(self, x):
        weight_norm=F.normalize(self.weight, dim=1,p=2)
        x_norm=F.normalize(x, dim=1,p=2)
        cos_sim=F.linear(x_norm, weight_norm)
        return self.scale * cos_sim
# 权重 key 映射：把 safetensors 里的 key 转成 torchvision ResNet50 的命名
def adapt_timm_resnet_state_dict(state_dict):
    """把 safetensors 权重 key 映射为 torchvision resnet50 的 key 格式

    实际模型 key 形如:
      resnet.embedder.embedder.convolution.weight            -> conv1.weight
      resnet.embedder.embedder.normalization.*               -> bn1.*
      resnet.encoder.stages.{S}.layers.{L}.layer.{B}.convolution.weight -> layer{S+1}.{L}.conv{B+1}.weight
      resnet.encoder.stages.{S}.layers.{L}.layer.{B}.normalization.*     -> layer{S+1}.{L}.bn{B+1}.*
      resnet.encoder.stages.{S}.layers.{L}.shortcut.convolution.weight   -> layer{S+1}.{L}.downsample.0.weight
      resnet.encoder.stages.{S}.layers.{L}.shortcut.normalization.*      -> layer{S+1}.{L}.downsample.1.*
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # stem: conv1 / bn1
        m = re.match(r"^resnet\.embedder\.embedder\.convolution\.(weight|bias)$", k)
        if m:
            new_state_dict["conv1." + m.group(1)] = v
            continue
        m = re.match(r"^resnet\.embedder\.embedder\.normalization\.(weight|bias|running_mean|running_var|num_batches_tracked)$", k)
        if m:
            new_state_dict["bn1." + m.group(1)] = v
            continue
        # bottleneck 块: layer{S+1}.{L}.conv{B+1} / bn{B+1}
        m = re.match(r"^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.layer\.(\d+)\.convolution\.(weight|bias)$", k)
        if m:
            new_state_dict["layer%d.%d.conv%d.%s" % (int(m.group(1)) + 1, int(m.group(2)), int(m.group(3)) + 1, m.group(4))] = v
            continue
        m = re.match(r"^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.layer\.(\d+)\.normalization\.(weight|bias|running_mean|running_var|num_batches_tracked)$", k)
        if m:
            new_state_dict["layer%d.%d.bn%d.%s" % (int(m.group(1)) + 1, int(m.group(2)), int(m.group(3)) + 1, m.group(4))] = v
            continue
        # shortcut -> downsample
        m = re.match(r"^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.shortcut\.convolution\.(weight|bias)$", k)
        if m:
            new_state_dict["layer%d.%d.downsample.0.%s" % (int(m.group(1)) + 1, int(m.group(2)), m.group(3))] = v
            continue
        m = re.match(r"^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.shortcut\.normalization\.(weight|bias|running_mean|running_var|num_batches_tracked)$", k)
        if m:
            new_state_dict["layer%d.%d.downsample.1.%s" % (int(m.group(1)) + 1, int(m.group(2)), m.group(3))] = v
            continue
        if k == "classifier.1.weight":
            new_state_dict["fc.weight"] = v
            continue
    default_logger.info(f"映射完成，共 {len(new_state_dict)} 个 key")
    return new_state_dict
# 从本地 safetensors 加载 ResNet50 预训练权重（迁移学习的起点）
def load_resnet50_from_local_safetensors():
    model_path=Path(settings.RESNET50_MODEL_PATH)
    try:
         raw_state_dict=load_file(f"{settings.RESNET50_MODEL_PATH}/model.safetensors")
    except Exception as e:
         raise RuntimeError(f"加载失败:{e}")
    sample_keys=list(raw_state_dict.keys())
    #if any(k.startswith("resnet.encoder.stages") for k in sample_keys):
    #     default_logger.info(f"非标准key")
    #else:
    #     default_logger.info(f"标准key")
    adapted_state_dict=adapt_timm_resnet_state_dict(raw_state_dict)
    model=models.resnet50(weights=None)
    old_fc_weight=None
    if 'fc.weight' in adapted_state_dict:
         old_fc_weight=adapted_state_dict['fc.weight']
         del adapted_state_dict['fc.weight']
    if 'fc.bias' in adapted_state_dict:
         del adapted_state_dict['fc.bias']
    missing_key,unexpected_key=model.load_state_dict(adapted_state_dict,strict=False)
    essential_keys=['conv1.weight','bn1.weight','layer1.0.conv1.weight']
    missing_essential=[k for k in essential_keys if k in missing_key]
    if missing_essential:
         default_logger.error(f"缺少key:{missing_essential}")
         raise RuntimeError(f"缺少key:{missing_essential}")
    in_features=model.fc.in_features
    num_classes=1000

    model.fc=ConsineClassifier(in_features,num_classes)
    if old_fc_weight is not None:
         model.fc.weight.data=old_fc_weight
         default_logger.info(f"旧FC已加载")
    else:
         default_logger.info(f"没有旧FC权重")
    return model

# 害虫识别服务（单例）：持有模型与类别名，提供加载/卸载/微调/推理
class ClassifyService:
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
        self.device=torch.device(settings.DEVICE)
        self.model=None
        self.class_names=[]
        self.model_ready=False
        self.transform=transforms.Compose([
               transforms.Resize((224,224)),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
          ])
        self._load_model()
     # 卸载模型：删除引用并回收内存与显存
     def unload_model(self):
         if self.model is not None:
              del self.model
              self.model=None
              gc.collect()
              if torch.cuda.is_available():
                   torch.cuda.empty_cache()
              default_logger.info(f"模型已卸载")
         else:
              default_logger.info(f"模型未加载")
         self.model_ready=False
     # 加载模型：优先加载微调后的 .pth（含类别表），否则退回预训练权重
     def _load_model(self): 
          self.unload_model()
          pth_path=Path(settings.RESNET50_FINETUNED_PTH_PATH)
          class_path=Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
          if pth_path.exists() and class_path.exists():
            try:
                with open(class_path,'r',encoding='utf-8') as f:
                    class_names=json.load(f)
                num_classes=len(class_names)
                model=load_resnet50_from_local_safetensors()
                in_features=model.fc.in_features
                model.fc=ConsineClassifier(in_features,num_classes)
                state_dict=torch.load(pth_path,map_location=self.device,weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model=model
                self.class_names=class_names
                self.model_ready=True
                default_logger.info(f"模型已加载")
                return
            except Exception as e:
                 default_logger.error(f"加载失败:{e}")
          default_logger.error("开始预训练模型")
          try:
               model=load_resnet50_from_local_safetensors()
               self.model=model
               self.class_names=self._get_imagenet_classes()
               self.model_ready=True
               default_logger.info(f"模型已加载")
          except Exception as e:
               default_logger.error(f"加载失败:{e}")
               raise RuntimeError(f"无法加载:{e}")
     # 兜底类别名：没有本地类别表时用（微调后会由 class_names.json 覆盖）
     def _get_imagenet_classes(self):
          try:
               weight=ResNet50_Weights.IMAGENET1K_V1
               return weight.meta['categories']
          except Exception as e:
               default_logger.warning(f"获取ImageNet失败:{e}")
               return [f"class_{i}" for i in range(1000)]
     #def _load_class_name_from_data(self):
     # 分类头扩容/裁剪：类别数变化时保留已学权重，并为新类别初始化参数
     def _expand_fc_layer(self,model,old_num_classes,new_num_classes):
          fc=model.fc
          if not isinstance(fc,ConsineClassifier):
               raise TypeError("模型")
          in_features=fc.in_features
          old_scale=fc.scale.data.item()
          new_scale=math.sqrt(new_num_classes)

          if old_num_classes==0:
               new_fc=ConsineClassifier(in_features,new_num_classes,scale=new_scale)
               nn.init.kaiming_uniform_(new_fc.weight,a=math.sqrt(5))
               model.fc=new_fc
               default_logger.info(f"fc层以扩展到{new_num_classes}")
               return model
          old_weight=fc.weight.data
          new_fc=ConsineClassifier(in_features,new_num_classes,scale=new_scale)
          if new_num_classes>old_num_classes:
               new_fc.weight.data[:old_num_classes]=old_weight[:old_num_classes]
               nn.init.kaiming_uniform_(new_fc.weight.data[old_num_classes:,:],a=math.sqrt(5))
               default_logger.info(f"fc已扩展")
          else:
               new_fc.weight.data[:,:]=old_weight[:new_num_classes,:]
               default_logger.info(f"fc已扩展")
          model.fc=new_fc
          return model
     # 微调训练主流程：准备数据 → 构建模型 → 逐 epoch 训练与验证 → 保存最优模型
     def fintune(self,date_dir:Path,epoch=settings.FULL_EPOCHS,stop_check=None):
          default_logger.info(f"开始微调")
          try:
               train_transform=transforms.Compose([
               transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
               transforms.RandomHorizontalFlip(p=0.5),
               transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.1),
               transforms.RandomRotation(20),
               transforms.ToTensor(),
               transforms.Normalize(
                    mean=[0.485,0.465,0.406],
                    std=[0.229,0.224,0.225]
               )
               ])
               val_transform=transforms.Compose([
               transforms.Resize(224),
               transforms.ToTensor(),
               transforms.Normalize(
                    mean=[0.485,0.465,0.406],
                    std=[0.229,0.224,0.225]
               )
               ])

               full_dataest=datasets.ImageFolder(root=str(date_dir),transform=train_transform)
               class_names=full_dataest.classes
               num_classes=len(class_names)
               if num_classes==0:
                    default_logger.error(f"")
                    return False
               total_len=len(full_dataest)
               indices=list(range(total_len))
               np.random.seed(42)
               np.random.shuffle(indices)
               val_len=int(0.2*total_len)
               train_indices=indices[val_len:]
               val_indices=indices[:val_len]
               val_dataest=datasets.ImageFolder(root=str(date_dir),transform=val_transform)

               val_dataset=torch.utils.data.Subset(val_dataest,val_indices)
               train_dataset=torch.utils.data.Subset(full_dataest,train_indices)
               train_labels=[full_dataest.targets[i] for i in train_indices]
               class_counts=np.bincount(train_labels,minlength=num_classes)
               class_counts=np.maximum(class_counts,1)
               class_weight=1.0/torch.tensor(class_counts,dtype=float)
               sample_weight=class_weight[train_labels]
               sample=WeightedRandomSampler(sample_weight,len(sample_weight),replacement=True)
               batch_size=settings.BATCH_SIZE
               train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,sampler=sample,shuffle=False,num_workers=0)
               val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
          except Exception as e:
               default_logger.error(f"数据加载失败:{e}")
               return False
     
          pth_path = Path(settings.RESNET50_FINETUNED_PTH_PATH)
          class_path = Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
          old_num_class = 0
          model = None
          if pth_path.exists() and class_path.exists():
               try:
                    with open(class_path,'r',encoding='utf-8') as f:
                         old_num_class=json.load(f)
                    old_num_class=len(old_num_class)
                    model=models.resnet50(weights=None)
                    in_feature=model.fc.in_features
                    model.fc=ConsineClassifier(in_feature,old_num_class)
                    state_dict=torch.load(pth_path,map_location=self.device,weights_only=True)
                    model.load_state_dict(state_dict)
                    default_logger.info(f"模型加载成功")
               except Exception as e:
                    default_logger.error(f"模型加载失败:{e}")
                    model=None
          if model is None:
               try:
                    model=load_resnet50_from_local_safetensors()
                    old_num_class=0
                    default_logger.info(f"模型加载成功")
               except Exception as e:
                    default_logger.error(f"模型加载失败")
                    raise RuntimeError(f"模型加载出错")
          if old_num_class !=num_classes:
               model=self._expand_fc_layer(model,old_num_class,num_classes)
               default_logger.info(f"")
          else:
               current_scale=model.fc.scale.data.item()
               target_scale=math.sqrt(num_classes)
               if abs(current_scale -target_scale)> 1e-3:
                    model.fc.scale.data.fill_(target_scale)
                    default_logger.info("scale 已更新")
               default_logger.info("模型分类数一致，保持原分类头")
          model=model.to(self.device)
          criterion=LabelSmoothingCrossEntropy(smoothing=0.1)
          lr_fc=5e-4
          optimizer=optim.Adam(model.fc.parameters(),lr=lr_fc,weight_decay=0.01)
          scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch,eta_min=1e-6)
          best_acc=0.0
          best_model_state=None
          for e in range(1,epoch+1):
               if stop_check and stop_check(0):
                    default_logger.info(f"微调模型结束")
                    return False
               model.train()
               running_loss,correct,total=0.0,0,0
               for images,labels in train_loader:
                    images=images.to(self.device)
                    labels=labels.to(self.device)
                    optimizer.zero_grad()
                    outputs=model(images)
                    loss=criterion(outputs,labels)
                    loss.backward()
                    clip_grad_norm_(model.fc.parameters(),1.0)
                    optimizer.step()
                    running_loss+=loss.item()*images.size(0)
                    _,pred=torch.max(outputs,1)
                    total+=labels.size(0)
                    correct+=(pred==labels).sum().item()
               train_loss=running_loss/total
               train_acc=correct/total
               model.eval()
               val_loss,val_correct,val_total=0.0,0,0
               with torch.no_grad():
                    for images,labels in val_loader:
                         images=images.to(self.device)
                         labels=labels.to(self.device)
                         optimizer.zero_grad()
                         outputs=model(images)
                         loss=criterion(outputs,labels)
                         val_loss+=loss.item()*images.size(0)
                         _,pred=torch.max(outputs,1)
                         val_total+=labels.size(0)
                         val_correct+=(pred==labels).sum().item()
               val_loss /=val_total
               val_acc =val_correct / val_total
               default_logger.info(f"Epoch {e}/{epoch} done, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}")
               scheduler.step()
               current_lr=optimizer.param_groups[0]['lr']
               if val_acc > best_acc:
                    best_acc=val_acc
                    best_model_state=copy.deepcopy(model.state_dict())
                    default_logger.info(f"验证准确率提升，当前epoch为{e},准确率为{val_acc:.4f}")
          if best_model_state is None:
               default_logger.info(f"微调失败")
               return False
          try:
               torch.save(best_model_state,settings.RESNET50_FINETUNED_PTH_PATH)
               default_logger.info(f"微调后保存模型成功:{settings.RESNET50_FINETUNED_PTH_PATH}")
               with open(Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH),'w',encoding='utf-8') as f:
                    json.dump(class_names,f,ensure_ascii=False,indent=2)
               default_logger.info(f"微调后的模型所识别的类型列表已保存:{settings.RESNET50_FINETUNED_CLASSNAMES_PATH}")
               del model
               gc.collect()
               if torch.cuda.is_available():
                    torch.cuda.empty_cache()
               self._load_model()
               default_logger.info(f"微调后的模型载入成功:{settings.RESNET50_FINETUNED_CLASSNAMES_PATH}")
               return True

          except Exception as e:
               default_logger.error(f"保存微调后的模型失败:{e}")
               return False

     # 推理：图片字节 → 预处理 → 前向计算 → top-k 类别与置信度（低于阈值判为未知）
     def predict(self,image_bytes:bytes,top_k:int=5)->dict:
          if self.model is None:
               default_logger.info(f"微调失败")
               return {
                    "top1":"模型未加载",
                    "top1_confidence":0.0,
                    "top5":[{
                         "class":"模型未加载",
                         "confidence":0.0
                    }]
               }
          if not self.class_names:
               self.class_names=["未知类型"]*1000
          try:
               image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
               input_tensor=self.transform(image).unsqueeze(0).to(self.device)
               with torch.no_grad():
                    outputs=self.model(input_tensor)
                    probs=torch.softmax(outputs,dim=1).cpu().numpy()[0]
               top_indices=np.argsort(probs)[-top_k:][::-1]
               top_probs=probs[top_indices]
               top5=[]
               for idx,prob in zip(top_indices,top_probs):
                    class_name=self.class_names[idx] if idx < len(self.class_names) else f"未知类型_{idx}"
                    top5.append({
                         "class":class_name,
                         "confidence":float(prob)
                    })
               top1_confidence=top5[0]["confidence"] if top5 else 0.0
               threshold=getattr(settings,"CONFIDENCE_THRESHOLD",0.25)

               if top1_confidence < threshold:
                    return {
                         "top1":"未知类型",
                         "top1_confidence":0.0,
                         "top5":[]
                    }
               return{
                    "top1":top5[0]["class"],
                    "top1_confidence":top1_confidence,
                    "top5":top5
               }
          except Exception as e:
               default_logger.error(f"预测失败:{e}")
               return {
                    "top1":"预测失败",
                    "top1_confidence":0.0,
                    "top5":[{
                         "class":"预测失败",
                         "confidence":0.0
                    }]
               }


classify_services = ClassifyService()
