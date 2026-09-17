// pages/home/home.js
const app=getApp()


Page({
  data: {
    features:[
      {title:'模型训练',url:'/pages/train/train'},
      {title:'病害识别',url:'/pages/classify/classify'},
      {title:'知识库管理',url:'/pages/knowledge/knowledge'},
      {title:'智慧问答',url:'/pages/chat/chat'}
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