class Draw {
  frames = []
  keyframes = []
  idx = 0
  lastIdx = -1

  constructor(speed) {
    this.speed = speed
  }

  add(options, frame) {
    const array = options.array ? [...options.array] : []
    const data = options.data ? options.data.clone() : {}
    const pointers = new Pointers(options.pointers ? [...options.pointers] : [])
    const highlights = options.highlights ? [...options.highlights] : []
    const lineNumber = options.lineNumber ?? 1000
    const unsorted = options.unsorted // 未排序个数，用在冒泡排序3
    const sorted = options.sorted // 已排序索引，用在插入排序
    const cache = options.cache ? [...options.cache] : [] // 用在缓存行
    const node = options.node ? [...options.node] : [] // 用在链表
    const code = options.code ?? ''
    const keyframe = options.keyframe ?? false
    const cloned = options.cloned ?? {}
    const curr = options.curr ?? '' // 用在非递归遍历二叉树
    const left = options.left ?? '' // 用在非递归遍历二叉树
    const right = options.right ?? '' // 用在非递归遍历二叉树
    const pop = options.pop ?? '' // 用在非递归遍历二叉树
    const ancestorFromLeft = options.ancestorFromLeft ?? '' // 用在二叉搜索树
    const ancestorFromRight = options.ancestorFromRight ?? '' // 用在二叉搜索树
    const gap = options.gap ?? 1 // 用在希尔排序
    const sorts = options.sorts ?? [] // 用在快排，表示已排序索引集合
    const highlightVisited = options.highlightVisited ?? -1 // 用在非重复全排列
    this.frames.push(() =>
      frame({
        array,
        data,
        pointers,
        highlights,
        lineNumber,
        unsorted,
        sorted,
        cache,
        node,
        code,
        cloned,
        curr,
        left,
        right,
        pop,
        ancestorFromLeft,
        ancestorFromRight,
        gap,
        sorts,
        highlightVisited,
      })
    )
    this.keyframes.push(keyframe)
  }

  // 绘制当前帧
  draw(beforeDraw, force) {
    if ((force && force()) || this.idx != this.lastIdx) {
      beforeDraw && beforeDraw()
      this.frames[this.idx]()
      this.lastIdx = this.idx
    }
  }

  // 播放每一帧
  play() {
    if (this.idx >= this.frames.length - 1) {
      return
    }
    this.idx++
    setTimeout(() => this.play(), this.speed)
  }

  // 更新当前帧、播放等按钮
  updateFrameButtons() {
    // console.log(this.frames)
    document.querySelector('section.frames').innerHTML = ''
    for (let i = 0; i < this.frames.length; i++) {
      this.addFrameButton(i)
    }
    this.addPrevButton()
    this.addNextButton()
    this.addPlayButton()
    this.addCodeButton()
  }

  addCodeButton() {
    const btn = document.createElement('input')
    btn.type = 'button'
    btn.value = 'code'
    btn.onclick = function () {
      document.querySelectorAll('div.code').forEach((e) => {
        if (e.style.display === 'block') {
          e.style.display = 'none'
        } else if (e.style.display === 'none') {
          e.style.display = 'block'
        } else {
          e.style.display = 'none'
        }
      })
    }
    const div = document.createElement('div')
    document.querySelector('section.frames').appendChild(div)
    div.appendChild(btn)
  }

  addFrameButton(i) {
    const btn = document.createElement('input')
    btn.type = 'button'
    btn.value = this.keyframes[i] ? `* ${i}` : i
    btn.onclick = () => (this.idx = i)
    const div = document.createElement('div')
    document.querySelector('section.frames').appendChild(div)
    div.appendChild(btn)
  }

  addPrevButton() {
    const btn = document.createElement('input')
    btn.type = 'button'
    btn.value = '<'
    btn.onclick = () => {
      if (this.idx > 0) {
        this.idx--
      }
    }
    const div = document.createElement('div')
    document.querySelector('section.frames').appendChild(div)
    div.appendChild(btn)
  }

  addNextButton() {
    const btn = document.createElement('input')
    btn.type = 'button'
    btn.value = '>'
    btn.onclick = () => {
      if (this.idx < this.frames.length - 1) {
        this.idx++
      }
    }
    const div = document.createElement('div')
    document.querySelector('section.frames').appendChild(div)
    div.appendChild(btn)
  }

  addPlayButton() {
    const btn = document.createElement('input')
    btn.type = 'button'
    btn.value = 'play'
    btn.onclick = () => this.play()
    const div = document.createElement('div')
    document.querySelector('section.frames').appendChild(div)
    div.appendChild(btn)
  }
}

function minAndMax(array) {
  return [Math.min(...array), Math.max(...array)]
}

// 按比例计算每个值矩形的高度
function calculateRectHeight(array, minHeight, maxHeight) {
  const [min, max] = minAndMax(array)
  const map = new Map()
  if (min == max) {
    for (let i = 0; i < array.length; i++) {
      map.set(array[i], minHeight)
    }
    return map
  }
  for (let i = 0; i < array.length; i++) {
    map.set(
      array[i],
      minHeight + ((array[i] - min) / (max - min)) * (maxHeight - minHeight)
    )
  }
  return map
}

// {index: , text: }
class Pointers {
  static WIDTH = 30
  static BASE_HEIGHT = 150
  pointers = []
  constructor(pointers) {
    this.pointers = pointers
  }
  // 横向
  draw2(currentIndex, y, lineStartWidth) {
    const results = []
    for (let i = 0; i < this.pointers.length; i++) {
      results[i] = 0
      for (let j = i + 1; j < this.pointers.length; j++) {
        const pi = this.pointers[i]
        const pj = this.pointers[j]
        if (pi.index === pj.index) {
          results[i]++
        }
      }
    }
    for (let i = 0; i < this.pointers.length; i++) {
      this.pointers[i].width = results[i] * Pointers.WIDTH
    }
    for (const p of this.pointers) {
      if (currentIndex === p.index) {
        fill(255)
        line(
          lineStartWidth,
          y + Pointers.WIDTH / 2,
          lineStartWidth + Pointers.BASE_HEIGHT,
          y + Pointers.WIDTH / 2
        )
      }
    }

    for (const p of this.pointers) {
      if (currentIndex === p.index) {
        fill(255)
        circle(
          lineStartWidth + Pointers.BASE_HEIGHT + p.width,
          y + Pointers.WIDTH / 2,
          Pointers.WIDTH
        )
        fill(0)
        text(
          p.text,
          lineStartWidth + Pointers.BASE_HEIGHT + p.width,
          y + Pointers.WIDTH / 2 + 4
        )
      }
    }
  }
  // 纵向
  draw(currentIndex, x, lineStartHeight) {
    const results = []
    for (let i = 0; i < this.pointers.length; i++) {
      results[i] = 0
      for (let j = i + 1; j < this.pointers.length; j++) {
        const pi = this.pointers[i]
        const pj = this.pointers[j]
        if (pi.index === pj.index) {
          results[i]++
        }
      }
    }
    for (let i = 0; i < this.pointers.length; i++) {
      this.pointers[i].height =
        Pointers.BASE_HEIGHT + results[i] * Pointers.WIDTH
    }
    for (const p of this.pointers) {
      if (currentIndex === p.index) {
        fill(255)
        line(x, height - lineStartHeight, x, height - p.height)
      }
    }

    for (const p of this.pointers) {
      if (currentIndex === p.index) {
        stroke('black')
        fill(255)
        circle(x, height - p.height, Pointers.WIDTH)
        fill(0)
        noStroke()
        text(p.text, x, height - p.height + 4)
      }
    }
  }
}

class Cache {
  index = 0
  array = []
  add(i) {
    const exists = this.array.map((c) => Number(c.split('_')[0])).includes(i)
    const line = [i, exists]
    for (let j = 0; j < 14; j++) {
      line.push(`[${i},${j}]`)
    }
    this.array[i % 8] = line.join('_')
    return this
  }
}

// 按起点与终点画箭头
function arrow1(x1, y1, x2, y2, color) {
  push()
  stroke(color)
  line(x1, y1, x2, y2)
  const offset = 4
  const th = 0.5
  const angle = atan2(y1 - y2, x1 - x2)
  translate(x2, y2)
  rotate(angle - HALF_PI)
  // line(-offset * th, offset, 0, -offset * th)
  // line(offset * th, offset, 0, -offset * th)
  fill(color)
  // noStroke()
  // -2 4,  0,-2,  2, 4
  // triangle(-offset * th, offset, 0, -offset * th, offset * th, offset)
  triangle(-1, 3, 0, 0, 1, 3)
  pop()
}

// 按起点与长度角度画箭头
function arrow2(x, y, length, angle, color) {
  push()
  stroke(color)
  const x2 = x + cos(angle) * length,
    y2 = y + sin(angle) * length
  line(x, y, x2, y2)
  // const offset = 5
  // const th = 0.5
  translate(x2, y2)
  rotate(angle + HALF_PI)
  fill(color)
  // noStroke()
  // triangle(-offset * th, offset, 0, -offset * th, offset * th, offset)
  triangle(-1, 3, 0, 0, 1, 3)
  pop()
}

function loadOptionsFromDom() {
  const elements = document.querySelectorAll('.saveable')
  const options = {}
  for (let i = 0; i < elements.length; i++) {
    const e = elements[i]
    if (e.id) {
      options[e.id] = e.value
      if (e.classList.contains('int')) {
        options[e.id] = Number(e.value)
      } else if (e.classList.contains('array')) {
        options[e.id] = e.value.split(',').map((e) => Number(e))
      } else if (e.classList.contains('narray')) {
        options[e.id] = e.value.split(',').map((e) => {
          if (e === 'null' || e === 'n') {
            return null
          } else {
            return e
          }
        })
      }
      if (e.type == 'checkbox' && e.classList.contains('bool')) {
        options[e.id] = e.checked ?? false
      }
    }
  }
  return options
}
function onSave(name) {
  const options = loadOptionsFromDom()
  const json = JSON.stringify(options)
  localStorage.setItem(name, json)
  window.location.reload()
}
function loadOptionsFromStorage(name) {
  let json = localStorage.getItem(name)
  if (json) {
    const options = JSON.parse(json)
    for (let key in options) {
      const e = document.getElementById(key)
      if (e) {
        e.value = options[key]
        if (e.classList.contains('array')) {
          e.value = options[key].join(',')
        } else if (e.classList.contains('narray')) {
          e.value = options[key].map((e) => (e == null ? 'n' : e)).join(',')
        }
        if (e.type == 'checkbox' && e.classList.contains('bool')) {
          e.checked = options[key]
        }
      }
    }
    return options
  } else {
    return loadOptionsFromDom()
  }
}

function clone(tree) {
  return JSON.parse(JSON.stringify(tree))
}
