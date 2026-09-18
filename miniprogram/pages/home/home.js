// pages/home/home.js
const app=getApp()


Page({
  data: {
    features:[
      {icon:'🧠',title:'模型训练',desc:'上传样本 在线微调模型',url:'/pages/train/train'},
      {icon:'📷',title:'害虫识别',desc:'拍照识别害虫种类',url:'/pages/classify/classify'},
      {icon:'📚',title:'知识库管理',desc:'上传文档 构建向量库',url:'/pages/knowledge/knowledge'},
      {icon:'💬',title:'智慧问答',desc:'基于知识库的AI问答',url:'/pages/chat/chat'}
    ]
  },
  navigateTo(e){
    const url=e.currentTarget.dataset.url
    if (!url) return 
    const tabBarPages=[
      '/pages/home/home',
      '/pages/train/train',
      '/pages/classify/classify',
      '/pages/knowledge/knowledge',
      '/pages/chat/chat'
    ]

    if(tabBarPages.includes(url)){
      wx.switchTab({
        url
      })
    }
    else{
      wx.navigateTo({
        url
      })
    }

  }

})