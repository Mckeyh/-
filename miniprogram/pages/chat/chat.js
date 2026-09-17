// pages/chat/chat.js
const app = getApp()
const BASE_URL = app.apiBaseUrl

Page({

  /**
   * 页面的初始数据
   */
  data: {
    messages: [],
    inputText: '',
    isStreaming: false,
    scrollIntoView: ''
  },

  onUnload() {
    this._cleanup()
  },

  onHide() {
    this._cleanup()
  },

  // 清理定时器并中断请求，避免页面卸载后回调已销毁的页面
  _cleanup() {
    if (this._chunkTimer) {
      clearTimeout(this._chunkTimer)
      this._chunkTimer = null
    }
    if (this._requestTask) {
      try { this._requestTask.abort() } catch (e) { /* ignore */ }
      this._requestTask = null
    }
  },

  onInput(e) {
    this.setData({ inputText: e.detail.value })
  },

  getTime() {
    const now = new Date()
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`
  },

  // scroll-into-view 只在值变化时生效，所以先清空再设置
  scrollBottom() {
    this.setData({ scrollIntoView: '' }, () => {
      this.setData({ scrollIntoView: 'bottom' })
    })
  },

  // 重置换段超时定时器，返回新的定时器 id
  resetChunkTimeout(timeoutId) {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    const timer = setTimeout(() => {
      if (this.data.isStreaming) {
        this.setData({ isStreaming: false })
        wx.showToast({ title: '请求超时', icon: 'none', duration: 3000 })
      }
    }, 120 * 1000)
    this._chunkTimer = timer
    return timer
  },

  sendMessage() {
    const text = this.data.inputText.trim()
    if (!text || this.data.isStreaming) return

    const userMsg = {
      id: `u_${Date.now()}`,
      role: 'user',
      content: text,
      time: this.getTime()
    }
    const aiMsg = {
      id: `a_${Date.now()}`,
      role: 'ai',
      content: '',
      time: this.getTime()
    }
    this.setData({
      messages: [...this.data.messages, userMsg, aiMsg],
      inputText: '',
      isStreaming: true
    })
    this.scrollBottom()

    let timeoutId = this.resetChunkTimeout(null)

    const finish = () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
      this._chunkTimer = null
      this.setData({ isStreaming: false })
      this.scrollBottom()
    }

    const appendToLastAi = (chunk) => {
      const messages = this.data.messages
      if (!messages.length) return
      const last = messages[messages.length - 1]
      if (last.role !== 'ai') return
      last.content += chunk
      this.setData({ messages })
      this.scrollBottom()
    }

    const requestTask = wx.request({
      url: `${BASE_URL}/api/chat/stream`,
      method: 'POST',
      data: {
        message: text,
        temperature: 0.7,
        max_tokens: 512
      },
      header: { 'content-type': 'application/json' },
      enableChunked: true,
      responseType: 'arraybuffer',
      success: (res) => {
        if (res.statusCode !== 200) {
          appendToLastAi(`请求失败（${res.statusCode}）`)
        }
        finish()
      },
      fail: (err) => {
        appendToLastAi('无法与服务通信')
        console.error('交互失败', err)
        finish()
      }
    })
    this._requestTask = requestTask

    // 流式数据必须用 onChunkReceived（onHeadersReceived 只有响应头）
    let buffer = ''
    requestTask.onChunkReceived((res) => {
      timeoutId = this.resetChunkTimeout(timeoutId)
      const bytes = new Uint8Array(res.data)
      buffer += this.utf8ArrayToStr(bytes)
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data:')) continue
        const jsonStr = line.substring(5).trim()
        if (!jsonStr) continue
        try {
          const data = JSON.parse(jsonStr)
          // 注意：后端错误帧也带 done:true，必须先判 error
          if (data.error) {
            appendToLastAi(`\n[出错了] ${data.error}`)
            finish()
          } else if (data.done) {
            finish()
          } else if (data.chunk !== undefined) {
            appendToLastAi(data.chunk)
          }
        } catch (e) {
          console.error('解析JSON失败', e)
        }
      }
    })
  },

  utf8ArrayToStr(uint8Array) {
    let out = ''
    let i = 0
    const len = uint8Array.length
    let c, char2, char3
    while (i < len) {
      c = uint8Array[i++]
      switch (c >> 4) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
          out += String.fromCharCode(c)
          break
        case 12:
        case 13:
          char2 = uint8Array[i++]
          out += String.fromCharCode((c & 0x1F) << 6 | (char2 & 0x3F))
          break
        case 14:
          char2 = uint8Array[i++]
          char3 = uint8Array[i++]
          out += String.fromCharCode((c & 0x0F) << 12 | (char2 & 0x3F) << 6 | (char3 & 0x3F))
          break
        default:
          break
      }
    }
    return out
  }

})
