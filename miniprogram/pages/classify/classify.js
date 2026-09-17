// pages/classify/classify.js
const app=getApp()
const USER_DATA_PATH=wx.env.USER_DATA_PATH
const fs=wx.getFileSystemManager()


Page({

  /**
   * 页面的初始数据
   */
  data: {
    result:null,
    isProcessing:false,
    image:{
      path:'',
      name:'',
      size:0,
      url:''
    }

  },
  takePhoto(){
    wx.chooseMedia({
      count:1,
      mediaType:['image'],
      sourceType:['camera'],
      camera:'back',
      sizeType:['original'],
      success:(res)=>{
        const tempFile=res.tempFiles[0]
        this.copyAndAddImage(tempFile.tempFilePath,tempFile.size)
      }
    })
  },
  copyAndAddImage(tempPath,size){
    const ext=(tempPath.split('.').pop() || 'jpg').toLowerCase()
    const fileName=`img_${Date.now()}_${Math.random().toString(36).slice(2,6)}.${ext}`
    const cachePath=`${USER_DATA_PATH}/${fileName}`
    try{
      fs.copyFileSync(tempPath,cachePath)
      const name=tempPath.split('/').pop() || 'image.jpg'
      this.setData({
        image:{
          path:cachePath,
          name,
          size,
          url:cachePath
        },
        result:null
      })
      console.log('复制图片成功',cachePath)
    }catch(error){
      console.error('复制图片失败',error)
      wx.showToast({
        title:'复制图片失败',
        icon:'none'
      })
    }

  },

  startRecognize(){
    const imagePath=this.data.image && this.data.image.path
    if(!imagePath){
      wx.showToast({title:'请先拍照或选择图片',icon:'none'})
      return
    }
    if(this.data.isProcessing){
      return
    }
    this.setData({isProcessing:true,result:null})
    wx.showLoading({title:'识别中...',mask:true})

    const finish=()=>{
      wx.hideLoading()
      this.setData({isProcessing:false})
    }
    const toPercent=(v)=>`${(Number(v||0)*100).toFixed(2)}%`

    fs.readFile({
      filePath:imagePath,
      encoding:'base64',
      success:(res)=>{
        const ext=(imagePath.split('.').pop() || 'jpg').toLowerCase()
        const mime=ext==='jpg' ? 'jpeg' : ext
        const base64Image=`data:image/${mime};base64,${res.data}`

        wx.request({
          url:`${app.apiBaseUrl}/api/classify/`,
          method:'POST',
          header:{'content-type':'application/json'},
          data:{image:base64Image},
          success:(resp)=>{
            finish()
            if(resp.statusCode===200 && resp.data){
              const top5=(resp.data.top5 || resp.data.top_5 || []).map((item)=>({
                class:item.class,
                confidence:item.confidence,
                confidence_text:toPercent(item.confidence)
              }))
              const result={
                top1:resp.data.top1 || '无法识别',
                top1_confidence:resp.data.top1_confidence || 0,
                top1_confidence_text:toPercent(resp.data.top1_confidence),
                top5:top5
              }
              this.setData({result})
              console.log('识别结果',result)
            }else{
              const msg=(resp.data && (resp.data.detail || resp.data.message)) || '识别失败'
              wx.showToast({title:msg,icon:'none'})
            }
          },
          fail:(err)=>{
            finish()
            console.error('调用识别接口失败',err)
            wx.showToast({title:'网络错误,请检查后端服务',icon:'none'})
          }
        })
      },
      fail:(err)=>{
        finish()
        console.error('读取图片失败',err)
        wx.showToast({title:'读取图片失败',icon:'none'})
      }
    })
  },

  retake(){
    this.setData({result:null})
    this.takePhoto()
  }

})
