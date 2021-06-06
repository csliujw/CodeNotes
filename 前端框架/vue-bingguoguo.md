# 安装

VSCode开发安装插件

- Vetur  -- Pine Wu
- Vue 3 Snippets -- hollowtree

#  基本用法

## 基本使用

## 双向数据绑定

只要vm监听到data中任何一条数据的变化，都会重新执行el区域的所有指令！！

## 插值表达式

> {{表达式}}

> v-cloak 基本不用

- 解决：插值表达式闪烁的问题（v-cloak指令来解决闪烁问题）

- 应用：网络比较卡是，可以为最外层的元素添加v-cloak，防止用户看到插值表达式

- ```html
  <style>
      [v-cloak]{
          display: none;
      }
  </style>
  <body>
      <div id="app" v-cloak>
          <h3>{{array[0]}}</h3>
      </div>
  </body>
  <script>
      const vm = new Vue({
          el: '#app',
          data: {
              array: [1, 2, 3, 4, 5]
          }
      });
  </script>
  ```

## 常见指令

vue中的指令，只有=={{}}==是用在内容节点中的，其它所有的指令，都是用在==属性节点==中的！！

- 内容节点`<div>{{msg}}</div>`
- 属性节点`<div v-html='msg'></div>`

> ==数据模板==

```html
<script>
    const vm = new Vue({
        el: '#app',
        data: {
            msg: "hello",
            array: [1, 2, 3, 4, 5],
            elem: "<span style='color:red'>Hello</span>"
        },
        method: {
            show: function(){
                console.log("ok");
            }
        }
    });
</script>
```

> v-text 基本不用

会把原来的内容清空。插值表达式只会把占位符处的数据进行解析替换。

```html
<h3 v-text="msg">
    12313
</h3>
// 显示 hello。12313会被覆盖掉的。
```

v-text中使用简单的语句

```html
<h3 v-text="msg + 666">
    12313
</h3>
// 显示hello666

<h3 v-text="msg + 'abc' ">
</h3>
// 显示 hellabc    
```

v-text不存在闪烁问题。

**场景：向元素的内容区域中，渲染指定的文本。**

> ==v-html==

```html
<h3 v-html="elem">
    // 可以解析html标签
</h3>
```

> ==v-bind：==属性绑定；用的很频繁

为html属性节点动态绑定数据的，如：

`<buttuon v-bind:title="mytitle">按钮</button>`

应用场景：如果元素的属性值，需要动态地进行绑定，则需要使用v-bind：指令

简写形式：

- v-bind可以简写为==:==，如
- `<buttuon :title="mytitle">按钮</button>`
- `<img :src="boo ？img1:img2" />` boo是布尔值，img1/2 是图片链接，以此动态切换图片路径

> ==v-on：==  事件绑定

`<div v-on:click="show">按钮</div>`   绑定事件不传参

`<div v-on:click="show('hello')">按钮</div>`   绑定事件传参

`<div @click="show('hello')">按钮</div>`   **==简写==**

> ==v-model==

<a href="https://segmentfault.com/a/1190000006599500">几种实现双向绑定的做法</a>

<a href="https://www.cnblogs.com/kidney/p/6052935.html">csdn</a>

> ==v-bind==

> ==v-for==

> ==v-if /v-show==

## 答疑

var let const

# 常规用法

定义使用过滤器：处理文本显示格式

了解实例`生命周期`和`生命周期函数`

使用axios发起Ajax请求

- ES6中的Promise
- ES7中的async和await

带数据交互的案例

Vue常见的过渡动画（不重要）

## 过滤器

- 过滤器的作用示例：“2020-01-23T:10:02.945Z” => 2020-01-23
- 概念：过滤器本质上是一个函数，可被用作一些常见的文本格式化。
- 过滤器只可以用在两个地方：mustache插值表达式和 v-bind表达式
- 过滤器应该被添加在JavaScript表达式的尾部，由管道符指示；

### 全局过滤器

- 使用全局过滤器语法

  - `<span>{{dt | 过滤器名称}</span>`

- 定义全局过滤器语法

  - ```js
    Vue.filter('过滤器名称',function(originVal){
    	// doing something 对数据进行处理
    	return 处理结果;
    })
    ```

- 使用过滤器的注意事项

  - 如果想拿管道符前面的值，通过function的第一个形参来拿。
  - 过滤器中，一定要返回一个处理的结果，否则就是一个无效的过滤器
  - 在调用过滤器的时候，直接通过()调用就能传参；从过滤器处理函数的第二个形参开始接收传递过来的参数。
  - 可多次使用 | 管道符，一次调用多个过滤器

> 全局过滤器代码示例

```javascript
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="js/vue.js"></script>
</head>

<body>
    <div id="app">
        <h3>{{time | dataFormat}}</h3>
    </div>
</body>
<script>
    Vue.filter('dataFormat', function (originVal) {
        const date = new Date(originVal);
        let years = date.getFullYear()
        let month = date.getMonth() + 1
        let day = date.getDay()
        // 魔法字符串${}是占位符
        return `${years}-${month}-${day}`;
    });

    const vm = new Vue({
        el: '#app',
        data: {
            time: '2020-01-22 23:11:23'
        }
    });

</script>

</html>
```



## 铺垫知识

## 实例生命周期

### 什么是生命周期

**生命周期**：实例的生命周期，就是一个阶段，从创建到运行，再到销毁的阶段。
**生命周期函数**：在实例的生命周期钟，在特定阶段执行的一些特定的事件，这些事件，叫做生命周期函数；

- 生命周期函数 = 生命周期钩子 = 生命周期事件

### 主要的生命周期函数分类

- 创建期间的生命周期函数：（特点：每个实例一辈子只执行一次）
    - `beforeCreate`：创建之前，此时data和methods尚未初始化
    - ==created==（第一个重要的函数，此时，data和methods已经创建好了，可以被访问了，首页数据的请求一般在这里发起！）
    - `beforeMount`：挂在模板结构之前，此时，页面还没有被渲染到浏览器中（如果想初始化一些第三方的JS插件，必须在mounted中进行初始化。比如echarts，它需要在初始化完毕的dom中进行操作）
    - ==mounted==（第二个重要的函数，此时，页面刚被渲染出来；如果需要操作DOM元素，最好在这个阶段；如使用三方插件，该插件需要DOM初始化完毕！）

- 运行期间的生命周期函数：（特点：按需被调用至少0次，最多N次）
  - beforeUpdate：数据是最新的，页面是旧的。
  - updated：页面和数据都是最新的。
- 销毁期间的生命周期函数：（特点：每个实例一辈子只执行一次）
  - beforeDestory：销毁之前，实例还是正常可用。
  - destoryed：销毁之后，实例已经不在工作了。

<img src="https://cn.vuejs.org/images/lifecycle.png" />

## Promise、async、await

### Promise

> 概念：

ES6中的新语法，Promise是一个构造函数；每个new出来的Promise实例对象，都代表一个异步操作。

JS解析引擎是单线程的；宿主环境（浏览器、Node环境）是多线程的。

异步的任务会放到异步回调函数的队列中。当js把自己栈中的任务执行完后，才会执行异步回调函数队列中的任务。

> 作用

解决了回调地狱的问题；

- 回调地狱，指的是回调函数中，嵌套回调函数的代码形式；如果嵌套的层级很深，就是回调地狱。
- 回调地狱，不利于代码的阅读、维护和后期的扩展。

### Promise用法

异步代码回顾

```js
/**
JS解析引擎是单线程的；宿主环境（浏览器、Node环境）是多线程的。

异步的任务会放到异步回调函数的队列中。当js把自己栈中的任务执行完后，才会执行异步回调函数队列中的任务。
*/

```

回调地狱代码示例：`node.js`

```js
const fs = require('fs')

fs.readFile('./files/1.txt', 'utf-8', (err, dataStr1) => {
    if (err) return console.log(err.message);
    console.log(dataStr1);
    fs.readFile('./files/2.txt', 'utf-8', (err, dataStr1) => {
        if (err) return console.log(err.message);
        console.log(dataStr1);
        fs.readFile('./files/3.txt', 'utf-8', (err, dataStr1) => {
            if (err) return console.log(err.message);
            console.log(dataStr1);
        })
    })
})

```

----

Promise不会减少代码量，但是可以解决回调地狱的问题。

创建形式上的异步操作

```js
const p = new Promise()
```

创建具体的异步操作；只要new了就会立即执行！

```js
// 只要new了，就会立即执行！
const p = new Promise(function(successCb,errorCb){
    // 定义具体的异步操作
})
// 定义成功和失败的回调
p.then(successCallback,errorCallback);
```

---

查看下Promise的原型链

Promise

- prototype
  - ==catch==：function catch()
  - constructor：function Promise()
  - finally：function finally()
  - ==then==：function then()  为Promise示例对象，指定 成功 和 失败的回调函数

```js
const fs = require('fs')

//==================无效写法================
function getContentByPath(fPath) {
    // js主线程只负责new出这个Promise，具体的执行交给浏览器执行了
    const p = new Promise(function () {
        fs.readFile(fPath, 'utf-8', (err, dataStr1) => {
            if (err) return console.log(err.message);
            console.log(dataStr1);
            // return dataStr1; 所以这个返回值是无效的。
        })
    })
}
getContentByPath('./files/1.txt')
//==================无效写法================


//==================有效写法================
function getContentByPath2(fPath) {
    // js主线程只负责new出这个Promise，具体的执行交给浏览器执行了.回调函数从哪里来？
    const p = new Promise(function (successCallback, errorCallback) {
        fs.readFile(fPath, 'utf-8', (err, dataStr1) => {
            if (err) return errorCallback(err);
            successCallback(dataStr1)
        })
    });
    return p;
}

const r1 = getContentByPath2('./files/1.txt')
// 成功回调  失败回调
r1.then(function (info) { console.log(info); console.log("success"); }, function (err) { console.log(err); });
//==================有效写法================

```

实际我们不会自己封装Promise，会使用其他人封装的方法。

### async和await

ES7中async和await可以简化Promise调用，提高Promise代码的阅读性和理解性

## axios

## 案例

