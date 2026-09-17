// pages/train/train.js
const app = getApp()
Page({

  data: {
    imageList: [],
    label: '',
    clearold: false,
    isTraining: false,
    status: { status: 'idle', result: null },
    ws: null,
    canStart: false
  },

  onLoad() {
    this._closed = false
    this.initWebSocket()
    this.updateCanStart()
  },

  initWebSocket() {
    try {
      const wsUrl = app.apiBaseUrl.replace(/^http/, 'ws') + '/api/train/ws'
      const ws = wx.connectSocket({
        url: wsUrl,
        success: () => { console.log('ws连接成功') }
      })
      ws.onMessage((res) => {
        try {
          const msg = JSON.parse(res.data)
          console.log('ws接收到消息', msg)
          const isTraining = msg && msg.status === 'running'
          this.setData({
            status: msg,
            isTraining: isTraining
          }, this.updateCanStart)
        } catch (e) {
          console.log('解析数据失败', e)
        }
      })
      ws.onClose(() => {
        console.log('ws连接关闭,3秒后重连')
        if (this._closed) return; this._reconnectTimer = setTimeout(() => { this.initWebSocket() }, 3000)
      })
      ws.onError((res) => {
        console.log('ws连接错误', res); this.syncStatus()
      })
      this.ws = ws
    } catch (e) {
      console.log('ws初始化失败', e)
    }
  },

  updateCanStart() {
    const can = this.data.imageList.length > 0 &&
      this.data.label.trim() !== '' &&
      !this.data.isTraining
    this.setData({ canStart: can })
  },

  onShow() {
    this.updateCanStart()
    this.syncStatus()
  },

  onUnload() {
    this._closed = true
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer)
      this._reconnectTimer = null
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  },

  chooseImages() {
    const remaining = 9 - this.data.imageList.length
    if (remaining <= 0) {
      wx.showToast({ title: '最多只能选择9张图片', icon: 'none' })
      return
    }
    wx.chooseMedia({
      count: Math.min(9, remaining),
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const newFiles = res.tempFiles.map(f => ({
          path: f.tempFilePath,
          name: f.tempFilePath.split('/').pop() || 'image.jpg',
          size: f.size
        }))
        const added = newFiles.slice(0, remaining)
        if (newFiles.length > remaining) {
          wx.showToast({ title: `仅能添加前${remaining}张图片`, icon: 'none' })
        }
        this.setData({
          imageList: [...this.data.imageList, ...added]
        }, this.updateCanStart)
      }
    })
  },

  clearImages() {
    this.setData({ imageList: [] }, this.updateCanStart)
  },

  removeImage(e) {
    const index = e.currentTarget.dataset.index
    const list = this.data.imageList
    list.splice(index, 1)
    this.setData({ imageList: list }, this.updateCanStart)
  },

  onLabelInput(e) {
    this.setData({ label: e.detail.value }, this.updateCanStart)
  },

  onClearOldChange(e) {
    this.setData({ clearold: !!e.detail.value }, this.updateCanStart)
  },

  startFull() {
    if (!this.data.canStart) return
    this.setData({ isTraining: true }, this.updateCanStart)
    wx.showLoading({ title: '准备图片中...', mask: true })

    const fs = wx.getFileSystemManager()
    const tasks = this.data.imageList.map((item) => {
      return new Promise((resolve, reject) => {
        fs.readFile({
          filePath: item.path,
          success: (res) => {
            const base64 = wx.arrayBufferToBase64(res.data)
            const ext = item.path.split('.').pop() || 'jpg'
            const mimeType = `image/${ext === 'jpg' ? 'jpeg' : ext}`
            resolve(`data:${mimeType};base64,${base64}`)
          },
          fail: reject
        })
      })
    })

    Promise.all(tasks).then((images) => {
      wx.hideLoading()
      wx.showLoading({ title: '开始训练...', mask: true })
      const payload = {
        images: images,
        label: this.data.label.trim(),
        clear_old: this.data.clearold
      }
      wx.request({
        url: `${app.apiBaseUrl}/api/train/finetune`,
        method: 'POST',
        data: payload,
        success: (res) => {
          wx.hideLoading()
          if (res.statusCode === 200) {
            wx.showToast({ title: '训练启动成功', icon: 'success' })
            this.setData({
              imageList: [],
              label: '',
              clearold: false,
              isTraining: true
            }, this.updateCanStart)
          } else {
            const body = res.data || {}
            wx.showToast({ title: (body && body.message) || '训练失败', icon: 'none' })
            this.setData({ isTraining: false }, this.updateCanStart)
          }
        },
        fail: () => {
          wx.hideLoading()
          wx.showToast({ title: '网络错误', icon: 'none' })
          this.setData({ isTraining: false }, this.updateCanStart)
        }
      })
    }).catch((err) => {
      wx.hideLoading()
      wx.showToast({ title: '训练失败', icon: 'none' })
      this.setData({ isTraining: false }, this.updateCanStart)
    })
  },

  // 用 HTTP 接口同步训练状态（WS 断开时的兜底）
  syncStatus() {
    wx.request({
      url: `${app.apiBaseUrl}/api/train/status`,
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200 && res.data) {
          const msg = res.data
          this.setData({
            status: msg,
            isTraining: msg.status === 'running'
          }, this.updateCanStart)
        }
      },
      fail: () => { }
    })
  },

  stopTraining() {
    if (!this.data.isTraining && this.data.status.status !== 'running') return
    wx.request({
      url: `${app.apiBaseUrl}/api/train/stop`,
      method: 'POST',
      success: (res) => {
        if (res.statusCode === 200) {
          wx.showToast({ title: '停止训练成功', icon: 'none' })
          this.setData({ isTraining: false }, this.updateCanStart)
        } else {
          const body = res.data || {}
          wx.showToast({ title: (body && body.message) || '停止训练失败', icon: 'none' })
          this.syncStatus()
        }
      },
      fail: () => {
        wx.showToast({ title: '停止训练失败', icon: 'none' })
        this.syncStatus()
      }
    })
  }
})