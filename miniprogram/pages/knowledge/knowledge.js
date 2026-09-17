// pages/knowledge/knowledge.js
const app=getApp()
const BASE_URL=app.apiBaseUrl
Page({

  /**
   * 页面的初始数据
   */
  data: {
    loading:false,
    result:null
  },

  chooseFile(){
    wx.chooseMessageFile({
      count:1,
      type:'file',
      extension:['txt','doc','docx','pdf'],
      success:(res)=>{
        const file=res.tempFiles[0]
        wx.showLoading({
          title:'上传中...'
        })
        wx.uploadFile({
          url:`${BASE_URL}/api/knowledge/upload`,
          filePath:file.path,
          name:'file',
          success:(res)=>{
            wx.hideLoading()
            try{
              const body=JSON.parse(res.data)
              this.setData({result:body})
              if(res.statusCode===200){
                wx.showToast({
                  title:'上传成功',
                  icon:'success'
                })
              }else{
                wx.showToast({
                  title:(body&&body.message)||'上传失败',
                  icon:'none'
                })
              }
            } catch(e){
              wx.hideLoading()
              wx.showToast({
                title:'上传失败',
                icon:'none'
              })
            }
          },
          fail:(res)=>{
            wx.hideLoading()
            wx.showToast({
              title:'无法与服务通信',
              icon:'none'
            })
          }
        })
      }
    })
  }
})