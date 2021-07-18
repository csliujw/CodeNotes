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
  - ==then==：function then()  为Promise示例对象，.then()方法最多需要两个参数，成功 和 失败的回调函数；它的返回值是Promise对象。
  - catch：function catch() 捕获前面所有.then()中发生的错误，集中处理。

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

> ES7中async和await可以简化Promise调用，提高Promise代码的阅读性和理解性。

- 如果某个方法的返回值是 Promise 对象，那么，就可以用 await关键字，来修饰 promise 实例
- 如果一个方法内部用了 await 那么这个方法必须修饰为 async 异步方法
  - 精简：await 只能用在被 async 修饰的方法中

```js
function getContentByPath(fpath){
    return new Promise(function(successCb,errorCb){
        fs.readFile(fpath, 'utf-8',(err,data)=>{
            if(err) return errorCb(err)
            successCb(data)
        })
    })
}

const data = await getContentByPath("./fs.txt")

// 如果一个方法内部用了await那么这个方法必须修饰为async
async function test(){
	const data = await getContentByPath("./fs.txt")
}
```

## axios

> 之前发起请求的方式

- 最开始封装XMLHttpRequest对象发起Ajax请求。
- 使用Jquery中提供的工具函数：
  - `$.ajax({配置对象})`
  - `$.post(url地址, function(){})`
  - `$.get(url地址，处理函数)`
- 现在，用axios发起Ajax请求。
  - 只支持get和post请求，无法发起JSONP请求。
  - 如果设计到JSONP请求，让后端启用cors跨域资源共享即可。
- 在Vue中使用 vue-resource 发起数据请求
  - 支持get post jsonp ，vue官方不推荐。

### axios的使用

- 测试数据地址
  - get 测试地址 http://www.liulongbin.top:3005/api/get
  - post 测试地址 http://www.liulongbin.top:3005/api/post
- 使用axios.get() 和 axios.post() 发起请求。
- 使用拦截器实现loading效果
- 使用 async 和 await 结合 axios 发起 Ajax请求 

#### get请求

> 使用axios发起get请求

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="js/vue.js"></script>
    <script src="js/axios.js"></script>
</head>

<body>
    <div id="app">
        <button @click='getInfo'>GET</button>
    </div>
    <script>
        const vm = new Vue({
            el: '#app',
            methods: {
                getInfo() {
                    const result = axios.get('http://www.liulongbin.top:3005/api/get', {
                        params: {
                            name: 'zs',
                            age: 20
                        }
                    });
                    result.then(function (res) {
                        console.log(res);
                    })
                }
            }
        });
    </script>
</body>
</html>
```

> 结合async await

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="js/vue.js"></script>
    <script src="js/axios.js"></script>
</head>

<body>
    <div id="app">
        <button @click='getInfo'>GET</button>
    </div>
    <script>
        const vm = new Vue({
            el: '#app',
            methods: {
                async getInfo() {
                    const result = await axios.get('http://www.liulongbin.top:3005/api/get', {
                        params: {
                            name: 'zs',
                            age: 20
                        }
                    });
                    console.log(result);
                }
            }
        });
    </script>

</body>

</html>
```

> 解构赋值

```js
const user = {
    name: 'zs',
    age: 20,
    gender: 'man'
}

// 把name属性解放出来，当作常量去使用。
// const { name } = user
// console.log(name);

// 给age取别名：userage
const { name, age: userage } = user
console.log(name, userage);
```

==这样我们获取数据的时候，就可以用解构赋值，得到我们想要的那部分数据了！==

```js
async function getInfo() {
    const {data:retVal} = await axios.get('http://www.liulongbin.top:3005/api/get', {
        params: {
            name: 'zs',
            age: 20
        }
    });
    console.log(result);
}
```

#### post请求

```js
async postInfo() {
    const { data: retVal } = await axios.post('http://www.liulongbin.top:3005/api/post', { name: 'ls', gender: 'man' })
    console.log(retVal.data);
}
```

#### Vue推荐用法

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="js/vue.js"></script>
    <script src="js/axios.js"></script>
</head>

<body>
    <div id="app">
        <button @click='getInfo'>GET</button>
        <button @click='postInfo'>POST</button>
    </div>
    <script>
        // 通过这个属性，全局设置 请求的 根路径。
        axios.defaults.baseURL = 'http://www.liulongbin.top:3005'
        Vue.prototype.$http = axios;
        const vm = new Vue({
            el: '#app',
            methods: {
                async getInfo() {
                    // 请求数据的时候会。 baseURL + 路径 = 'http://www.liulongbin.top:3005' + '/api/get'
                    const { data: retVal } = await this.$http.get('/api/get', {
                        params: {
                            name: 'zs',
                            age: 20
                        }
                    });
                    console.log(retVal.data);
                },
                async postInfo() {
                    const { data: retVal } = await this.$http.post('/api/post', { name: 'ls', gender: 'man' })
                    console.log(retVal.data);
                }
            }
        });
    </script>

</body>

</html>
```

### axois的传参

```js
this.$http.get('/user/10',{params:{name:'zs',age:22}}) // ===> http://127.0.0.1:8080/user/10?name=zs&age=2
```

## 案例

带有数据库的品牌管理案例

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="js/vue.js"></script>
    <script src="js/axios.js"></script>
    <link rel="stylesheet" href="css/bootstrap.css">
</head>

<body>
    <div id="app">

        <div class="panel panel-primary inline">
            <div class="panel-heading inline">
                <h3 class="panel-title">添加新品牌</h3>
            </div>

            <div class="panel-body form-inline">
                <div class="input-group">
                    <div class="input-group-addon">品牌名称</div>
                    <input type="text" class="form-control" v-model='name'>
                </div>

                <div class="input-group">
                    <button class="btn btn-primary" @click='add'>添加</button>
                </div>

                <div class="input-group">
                    <div class="input-group-addon">按名称搜索</div>
                    <input type="text" class="form-control" v-model='keywords'>
                </div>
            </div>
        </div>

        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>name</th>
                    <th>time</th>
                    <th>operate</th>
                </tr>
            </thead>
            <tbody>
                <!-- 很巧妙啊 in search() search用来过滤 -->
                <tr v-for="item in search()" :key='item.id'>
                    <td>{{item.id}}</td>
                    <td>{{item.name}}</td>
                    <td>{{item.ctime | dataFormat}}</td>
                    <td><a href="#" @click.prevent="remove(item.id)">删除</a></td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
<script>
    axios.defaults.baseURL = 'http://liulongbin.top:3005';
    Vue.prototype.$http = axios;

    // 定义全局过滤器
    Vue.filter('dataFormat', function (originVal) {
        const dt = new Date(originVal);
        const y = dt.getFullYear();
        const m = (dt.getMonth() + 1 + '').padStart(2, '0');
        const d = (dt.getDay() + '').padStart(2, '0');
        return `${y}-${m}-${d}`
    })


    const vm = new Vue({
        el: '#app',
        data: {
            brandList: [],
            name: '',
            keywords: ''
        },
        created() {
            //在created中发起首屏数据的请求
            this.getBandList()
        },
        methods: {
            async getBandList() {
                const { data: res } = await this.$http.get('/api/getprodlist');
                // console.log(res);
                // return res.message; 返回的是一个promise对象。
                // 应该这么写
                this.brandList = res.message;
            },
            async add() {
                const { data: res } = await this.$http.post('/api/addproduct', { name: this.name });
                if (res.status !== 0) return alert('添加失败！');
                this.getBandList();
                this.name = '';
            },
            search() {
                return this.brandList.filter(item=>item.name.includes(this.keywords))
            },
            async remove(id) {
                const { data: res } = await this.$http.get('/api/delproduct/' + id);
                if (res.status !== 0) return alert('删除失败');
                else this.getBandList();
            }
        }
    });
</script>

</html>
```

# Vue中的动画

## 主要内容

- Vue.js中的过渡动画
- webpack的基本配置和使用
- ES6模块化导入和导出
  - CommonJS ==> 必须有 require，exports，module
    - 导入模块：const fs = require('fs')
    - 暴露模块：module.exports={}
  - ES6模块化规范：
    - import $ from 'jquery' ==> 从jquery包中导入 `$`

## Vue中的动画

- 都是简单的过渡动画。

### 基本介绍

> **每个动画分为两部分**

- 入场动画：从不可见（flag=false）-> 可见（flag=true）
- 离场动画：可见（flag=true）-> 不可见（flag=false）

> 入场动画：两个时间点，一个时间段

- v-enter：入场前的样式。(class名)
- v-enter-to：入场完成以后的样式。(class名)
- v-enter-active：入场的时间段，即中间过渡的时间段。(class名)

>离场动画：两个时间点，一个时间段

- v-leave：离场前的样式。(class名)
- v-leave-to：离场完成以后的样式。(class名)
- v-leave-active：离场的时间段，即中间过渡的时间段。(class名)

<img src="../pics/vue/transition.png">

### Demo

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="js/vue.js"></script>
</head>
<style>
    /* 定义入场之前和离场之后的样式 */
    .v-enter,
    .v-leave-to {
        transform: translateX(150px);
    }

    /* 定义入场阶段和离场阶段的样式 */
    .v-enter-active,
    .v-leave-active {
        transition: all 0.8s ease;
    }
</style>

<body>
    <div id="app">
        <button @click='flag=!flag'>toggle</button>
        <!-- 1.使用vue提供的transition标签 包裹需要添加动画的元素 name默认以v为前缀。 -->
        <transition name='v'>
            <h3 v-if='flag'>asfaf</h3>
        </transition>
    </div>
    <script>
        const vm = new Vue({
            el: '#app',
            data: {
                flag: true
            },
            methods: {

            }
        })
    </script>
</body>

</html>
```

### 三方动画库

==Vue不支持animate4.0==

- 把需要添加动画的元素，使用v-if或v-show进行控制。
- 把需要添加动画的元素，使用Vue提供的元素`<transition></transition>`包裹起来
- 为`<transition></transition>`添加两个属性类`enter-active-class,leave-active-class`
- 把需求添加动画的元素，添加一个class='animated'

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="css/animate.min.css">
    <script src="js/vue.js"></script>
</head>

<body>
    <div id="app">
        <button @click='flag=!flag'>toggle</button>
        <!-- 1.使用vue提供的transition标签 包裹需要添加动画的元素 -->
        <transition enter-active-class="bounceInDown" leave-active-class="bounceOutDown">
            <h3 v-show='flag' class="animated">aasffasfsasfasfsfaf</h3>
        </transition>
    </div>
    <script>
        const vm = new Vue({
            el: '#app',
            data: {
                flag: true
            }
        })
    </script>
</body>

</html>
```

### v-for的列表过渡

- 把v-for循环渲染的元素，添加`:key`属性【注意：如果为列表项添加动画效果，一定要指定key，并且，key的值不能为索引】

- 在v-for循环渲染的元素外层，包裹`<transition-group>`标签

- 添加两组类即可：

  ```css
  .v-enter,
  .v-leave-to{
      opacity:0,
      transform:translateY(100px);
  }
  
  .v-enter-active,
  .v-leave-active{
      transition:all 0.8s ease;
  }
  ```

> 具体Demo

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="css/animate.min.css">
    <script src="js/vue.js"></script>
</head>

<body>
    <div id="app">
        <input v-model="name"> <button @click="add">添加</button>
        <!-- 默认会用span 包裹 li。我们指定tag的话，就会用我们指定的tag包裹 -->
        <transition-group tag='ul' enter-active-class="bounceInDown" leave-active-class="bounceOutDown">
            <li v-for="item in list" :key="item.id" @click="del(item.id)" class="animated">{{item.name}}</li>
        </transition-group>
    </div>

</body>
<script>
    const vm = new Vue({
        el: "#app",
        data: {
            list: [
                { "id": 1, "name": 'd1' },
                { "id": 2, "name": 'd2' },
                { "id": 3, "name": 'd3' }
            ],
            newId: 4,
            name: "123"
        },
        methods: {
            add() {
                const newInfo = { "id": this.newId++, "name": this.name }
                console.log(name);
                this.list.push(newInfo);
                this.name = ''
            },
            // 有问题，不过没事，就了解一下。
            del(id) {
                const i = this.list.findIndex(item=>item.id===id);
                this.list.splice(i, 1);
            }
        },
    })
</script>

</html>
```

# Webpack

## 网页中常见静态资源

- 样式表
  - .css .less .scss
- js文件
  - .js	.ts	.coffee
- 图片
  - .jpg/.jpeg	.png	.gif	.bmp	.webp
- 字体文件
  - .ttf	.eot	.woff	.woff2	.svg
- 模板文件
  - .vue	.jsx

> 引入静态资源多了？

- 对网页性能不友好：要发起很多静态资源请求，降低页面的加载效率，用户体验差；
- 对程序开发不友好：前端程序员要处理复杂的文件之间的依赖关系；

> 如何解决上述问题？

- 对于JS或CSS，可以压缩和合并；小图片适合转Base64格式的编码。
  - 合并：减少发送请求次数
  - 压缩：减小文件的传输量
  - Base：把图片转换成了字符串，无需为图片发起请求。
- 通过一些工具，让工具自动维护文件之间的依赖关系。

## Webpack

**Webpack：**前端项目的构建工具；前端的项目都是基于webpack进行构建和运行的。

**为什么用Webpack：**

- 1、如果项目使用webpack进行构建，我们可以书写高级的ES代码，且不用考虑兼容性。
- 2、webpack能够优化项目的性能，比如合并、压缩文件等；
- 3、基于webpack，程序员可以把自己的开发重心，放到功能上；

**什么项目适合使用webpack：**

- 单页面应用程序
  - vue、react、angular只要用前端三大框架开发项目，必然会使用webpack工具。
- 不太适合与多页面的普通网站结合使用
- <a href="http://webpack.github.io/">官网</a>

## webpack 流程

<img src="..\pics\redis\image-20210627210139677.png">

vscode快捷键==li{这是第$个li}*9==

## 安装和配置webpack

> 安装

1、新建一个项目的空白目录，并在终端中，cd到项目根目录，指向==npm init -y==初始化项目

2、装包：运行==npm i webpack webpack-cli -D== 安装项目构建所需要的webpack。

- 由于存在版本问题，所以推荐使用指定版本的工具==npm i webpack@4.17.1 webpack-cli==

3、打开 package.json 文件，在scripts节点中，==新增一个dev的节点==：

```shell
"scripts":{
	"test" : "echo \"Error: no test specified\" && exit 1",
	"dev" : "webpack"
}
```

4、在项目根目录中，新建一个webpack.config.js 配置文件，内容如下：

```js
// 这里用的 Node 语法， 向外导出一个 配置对象
module.exports = {
	mode: 'production' // production development
}
```

5、在项目根目录中，新增一个src目录，并且在src目录中，新建一个 index.js 文件，作为 webpack 构建的入口；会把打包好的文件输出到 dist->main.js

6、在终端中，==直接运行 npm run dev==启动 webpack进行项目构建；

webpack学习失败。我直接上极客时间的Vue了。

# 组件化Vue

## 模块化和组件化

模块化：从代码角度分析，把可复用的代码，抽离为单独的模块；==实现代码的复用。==

组件化：从页面UI的角度分析，把页面中可复用的 UI 结构，抽离为单独的组件；==实现 UI 的复用。==

## ES6导入导出新语法

一个模块可以同时使用按需导出和默认导出

### 默认导入和导出

```js
//xxx.js
let a = 10
```

----

```js
// 默认导入
import m1 from './js/xxx.js'
// 在webpack中，每个js文件都是独立的模块
// 每个模块都有独立的作用域
// 其他模块，默认无法直接访问当前模块中定义的成员。
console.log(m1)
```

----

```js
//xxx.js
let a = 10
let b = 20
// 这个export default{} 语法叫做默认导出。
// 在一个模块中，仅允许导出一次
export default {
    a: a,
    // 属性值和属性名一直可以简写。
    b,
    say(){
        console.log("hello")
    }
}
```

### 按需导入和导出

按需导入语法

```js
import { 成员名称 } from '模块名'

//eg
import m2,{xx} from "xxx.js"

// as 取别名
import m2,{test1 as myTest} from "xx.js"
```

按需导出语法

```js
export var a = 10
```

## 定义Vue组件

### 模块化、组件化

模块化：实现代码的复用；把可复用的代码 抽离为单独的模块；

- 提供模块作用域的概念，防止全局变量污染
- 提高了代码的复用率，方便了程序员之间共享代码

组件化：把页面中可复用的UI结构，抽离为单独的组件。

- 方便UI结构的重用
- 可以直接使用第三方封装好的组件库
- 更专注于业务逻辑

### 定义全局组件

> 语法

- `Vue.component('组件名称',{组件的配置对象})`
- 在组件的配置对象中：可以使用template属性指定当前组件要渲染的模板结构

> 使用语法组件

- 把组件的名称，以标签的形式，引入到页面上就行。

> 注意

- 从更抽象的角度来说，每个组件，就相对于一个自定义的元素；
- 组件中的DOM结构，有且只能有唯一的根元素来进行包裹。
- 组件其实就是封装了一些HMTL。

> 完整代码示例

```js
import Vue from "./js/vue.js"
// 名称尽量小写，中间用-隔开
Vue.component("my-test", {
    template: `<div> 这是我定义的组件 </div>`
})
const vm = new Vue({
    el: '#app',
    data: {
        msg: 'hello '
    }
})
```

----

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>
    <div id="app">
        <h5>{{msg}}</h5>
        <my-test></my-test>
    </div>
</body>

</html>
```

### 定义私有组件

```js
const vm = new Vue({
	el: '#app2',
	data: {
		info: '000'
	},
	components: {
		// '组件名称':{/* 组件配置对象 */}
		'my-test2':{
			tenplate: `<div>这是私有组件my-test2</div>`
		}
	}
})
```

### 私有组件的data

- 私有组件的data必须是一个function，且必须return一个数据对象。

- vm 实例中，data 即可以方法 也可以是对象

  ```js
  const vm = new Vue({
  	el: '#app2',
  	data: {
  		info: '000'
  	}
  }
  ```

```js
Vue.component("my-test", {
    template: `<div> 这是我定义的组件 {{d1}} </div>`,
    data(){
        return {
            d1:'111'
        }
    }
})
```

### 组件定义data、methods以及生命周期函数

可以认为：**组件是特殊的Vue实例**

组件和实例的相同和区别：

- 组件的 data 必须是一个 function 并 return 一个 字面量对象； 在 Vue 实例中，实例的 data 既可以是对象，也可以是方法；
- 组件中，直接通过 template 属性来指定组件的 UI 结构； 在 Vue 实例中，通过 el 属性来指定实例控制的区域；但是实例也可以使用 template；
- 组件和实例，都有自己的生命周期函数，私有的过滤器，methods 处理函数；

> 为什么组件中的 data 必须定义为一个方法并返回一个对象

因为这样，能够保证每次创建的组件实例，都有自己的一块唯一的数据内存，防止组件之间数据的干扰。

## Vue 文件结构说明

每个 .Vue 文件，都是一个 vue 组件（叫做 单文件组件），它由三部分组成：

- template 结构
- script 行为
- style 样式

> 定义组件Demo

```js
<template>
  <div>
    <h3>这是组件Home {{ msg }}</h3>
  </div>
</template>

<script>
export default {
  name: "Home",
  data() {
    return {
      msg: "hello vue"
    }
  },
  methods: {},
  filters: {}
}
</script>

<style scoped>
h3 {
  color: bisque;
}
</style>
```

## Vue文件中定义组件

- 声明一个Vue文件

  ```js
  <template>
    <div>
      <h3>这是Son 组件</h3>
    </div>
  </template>
  
  <script>
  export default {
    name: "Son"
  }
  </script>
  
  <style scoped>
  
  </style>
  ```

- 把这个组件 注册为全局组件或私有组件

  **全局组件**

  ```js
  import Home from "@/components/Home";
  import Vue from "vue";
  
  Vue.component('Home', Home)
  ```

  **私有组件**

  ```js
  import Son from "./Son"
  
  console.log(Son.name);
  export default {
    name: "Home",
    data() {
      return {
        msg: "hello vue"
      }
    },
    methods: {},
    filters: {},
    // 定义私有组件
    components: {
      'my-son': Son
    }
  }
  ```

## 组件样式控制

父组件的样式会影响子组件，如何解决？

默认情况下，组件中定义的样式是全局生效的。如何样式只在当前组件内生效？

给 style 加上 scope 属性，即可。如何做到的？只要为组件添加了 scope 那么当前组件（不包括引入的组件）所有的 标签 都会使用同一个属性。

```css
<style scope>
</style>
```

## 组件数据通信

### 父传子

在父组件中，以标签形式使用子组件时，可以通过属性绑定，为子组件传递数据。

在子组件中，如果向父组件传递过来的数据，必须先定义 props 数组来接收

接收完 props 数据，可以直接在子组件的 template 区域使用

----

> 代码

**子组件**

```js
<template>
  <div>
    <br>

    <button @click="objFromParent.a++">a自增</button>
    <h1>子组件---->{{ infoFormParent }}-----> {{ objFromParent }}</h1>
  </div>
</template>

<script>
import _ from 'loadsh'

export default {
  name: "Son",
  // 而 data 中的数据 可读可写
  data() {
    // 建议使用转存的数据，以便满足修改的请求。
    // 对于对象类型的数据， 存储的是地址值，我们需要把数据拷贝一份，不修改源数据。
    // 深拷贝 安装 lodash npm install lodash -S
    return {
      infoFormParent: this.pmsg,
      objFromParent: _.cloneDeep(this.obj)
    }
  },
  // 子组件需要使用 props 数组，接收外界传递过来的数据，接收到的数据可以直接在Son中使用
  // 通过 props 接收的数据，是只读的。不要为它们重新赋值。
  props: ['pmsg', 'obj']
}
</script>
```

**父组件**

```js
<template>
  <div>
    <h1>父组件</h1>
    <button @click="sendData">发送数组给子组件</button>
    <!--在使用组件的时候，通过 属性绑定，把数据传递给子组件-->
    <my-son :pmsg="parentMsg" :obj="obj"></my-son>
  </div>
</template>

<script>
import Son from "@/components/Father2Son/Son";

export default {
  name: "Parent",
  data() {
    return {
      parentMsg: '继承我的花呗',
      obj: {
        a: 10,
        b: 20
      }
    }
  },
  methods: {
    sendData() {}
  },
  components: {
    'my-son': Son
  }
}
</script>
```

**渲染调用**

```js
<template>
  <div id="app">
    <Parent></Parent>
  </div>
</template>

<script>
import Parent from "@/components/Father2Son/Parent";
import Vue from "vue";
Vue.component('Parent', Parent)

export default {
  components: {Parent}
}

</script>
```

### 子传父

> 通过事件绑定机制，子传数据给父

父为子绑定事件，然后子把自己的数据传递过去。

父亲调用方法会接收到子的数据，这时候就得到了子的数据。

> 代码Demo

子组件

```js
<template>
  <div>
    <h1>子组件</h1>
    <button @click="btnHandler">触发func事件</button>
    <button @click="btnHandler2">触发func2事件，带参数</button>
  </div>
</template>

<script>
export default {
  name: "Son",
  data() {
    return {
      msg: ': 我是子组件的值'
    }
  },
  methods: {
    btnHandler() {
      //$emit表示触发事件 , 在子组件中，通过  this.$emit() 触发父组件 为子组件绑定的 func 事件。
      // func 是父组件为子组件绑定的事件。
      this.$emit('func') // 调用父组件给子组件的事件 func
    },
    btnHandler2() {
      this.$emit('func2', this.msg) // 调用父组件给子组件的事件 func
    }
  }
}
</script>
```

父组件

```js
<template>
  <div>
    <h1>父组件</h1>
    <!--在使用组件的时候，通过 属性绑定，把数据传递给子组件-->
    <my-son @func="show" @func2="show2"></my-son>
  </div>
</template>

<script>
import Son from "@/components/Son2Father/Son";

export default {
  name: "Parent",
  data() {
    return {}
  },
  methods: {
    show(){
      console.log("有人调用了父组件的show方法！")
    },
    show2(args){
      console.log("父组件的 show2 带有参数"+args)
    }
  },
  components: {
    'my-son': Son
  }
}
</script>
```

### 兄弟传兄弟

> 思路

定义一个公共的Vue实例，如 bus.js 实例名称为 bus。

数据发送方，调用 `bus.$emit()` 触发 bus 上的某个事件，从而把数据发送出去。

在数据接收方，使用 `bus.$on()` 自定义事件，并指定事件处理函数。

----

> 代码示例

**公共Vue实例 bus.js**

```js
import Vue from 'vue'

const bus = new Vue()
export default bus
```

**发送数据方**

```vue
<template>
  <div>
    <h1>哥哥</h1>
    <button @click="sendMsgToDD">哥哥给弟弟数据</button>
  </div>
</template>

<script>
import bus from './bus'

export default {
  name: "GG",
  data() {
    return {
      msg: '哥哥有糖给弟弟'
    }
  },
  methods: {
    sendMsgToDD() {
      // 在数据发送方，调用bus.$emit() 触发 bus 上的某个事件，从而把数据发送出去
      bus.$emit('ooo', this.msg)
    }
  }
}
</script>
```

**接收数据方**

```vue
<template>
  <div>
    <h3>弟弟</h3>
  </div>
</template>

<script>
import bus from './bus'

export default {
  name: "DD",
  data() {
    return {}
  },
  created() {
    // 在数据接收方 使用 bus.$on 自定义事件，并指定事件处理函数
    bus.$on('ooo', data => {
      console.log("弟弟拿到了哥哥的数据，哥哥的数据是："+data)
    })
  }
}
</script>
```

# 操作DOM

## 使用 this.$refs 来获取元素和组件

> 基本使用

1.把要获取的 DOM 元素，添加 ref 属性，创建一个 DOM 对象的引用，指定的值，就是引用的名称

```html
<p ref="myP">这是父组件</p>
```

2.如果要获取某个引用所对应的 DOM 对象，则直接使用 `this.$refs.引用名称`

```js
console.log(this.$refs.myP)
```

3.也可使用 ref 为组件添加引用；可以使用 `this.$refs.组件名称`，拿到组件的引用，从而调用组件上的方法和获取组件 data 上的数据

> this.$refs 获取 DOM

```vue
<template>
  <div>
    <!--  通过 ref 获取到的 DOM 元素的引用，就是一个元素的 DOM 对象  -->
    <h3 id="h3" @click="getContent" ref="myh3">123</h3>
  </div>
</template>

<script>
export default {
  name: "Home1",
  methods: {
    getContent() {
      // 不要在vue中操作 DOM
      // console.log(document.getElementById("h3").innerHTML);
      console.log(this.$refs.myh3)
    }
  }
}
</script>
```

> ref 直接引用组件并调用组件的方法和数据 ★★★★★

==可以使用 ref 属性直接调用子组件的方法属性！==

实现父调用子的方法

```vue
<template>
  <div>
    <!--  通过 ref 获取到的 DOM 元素的引用，就是一个元素的 DOM 对象  -->
    <h3 id="h3" @click="getContent" ref="myh3">123</h3>
    <my-son ref="son"></my-son>
  </div>
</template>

<script>
import Son from "@/components/GetDocumnet/Son"

export default {
  name: "Home1",
  methods: {
    getContent() {
      // 不要在vue中操作 DOM
      // console.log(document.getElementById("h3").innerHTML);
      console.log(this.$refs.myh3)
      this.$refs.son.add()
    }
  },
  components: {
    "my-son": Son
  }
}
</script>
```

---

```vue
<template>
  <div>
    <h3>这是Son组件 {{ sonMsg }}</h3>
  </div>
</template>

<script>
export default {
  name: "Son",
  data() {
    return {
      sonMsg: 0
    }
  },
  methods: {
    add() {
      this.sonMsg++;
    }
  }
}
</script>
```

## 使用霸道的 render 函数渲染组件

只要在 vm 实例中，只当了 render 函数来渲染组件，那么 el 区域，就会被 render 中渲染的组件替换掉。

```js
import App from "@/components/App.vue"

const vm = new Vue({
	el: '#app',
	data: {
		msg: 'hello'
	},
    template: '<h6>{{msg}}<h6>',
    // createElements 形参是一个方法，专门用于渲染一个组件，并替换掉 el 区域。
    render: function(createElements){
        return createElemens(App)
    }
})
```

render 简化写法

```js
const vm = new Vue({
    el: '#app',
    render: h=>{
        return h(App)
    }
})
```

再简化

```js
const vm = new Vue({
    el: '#app',
    // render 终极形式 lambda 超级简化写法
    render: h => h(App)
})
```

**render 渲染的组件叫做根组件！**

# SPA

## component组件

通过 component 的 is 属性，动态指定要渲染的组件。

```vue
<template>
  <div id="app">
    <h1>App 根组件</h1>
    <!--  注意 是字符串 'GG' 按字符串名称来搜索的！  -->
    <component :is="'GG'"></component>
  </div>
</template>

<script>
// import Parent from "@/components/Father2Son/Parent";
import GG from '@/components/Son2Son/GG'
import DD from '@/components/Son2Son/DD'
import Vue from "vue";

Vue.component('GG', GG)
Vue.component('DD', DD)
export default {}
</script>
```

----

```vue
<template>
  <div id="app">
    <h1>App 根组件</h1>
    <button @click="comName='GG'">GG</button> &nbsp;&nbsp;&nbsp;
    <button @click="comName='DD'">DD</button>
    <!--  注意 是字符串 'GG' 按字符串名称来搜索的！  -->
    <component :is="comName"></component>
  </div>
</template>

<script>
// import Parent from "@/components/Father2Son/Parent";
import GG from '@/components/Son2Son/GG'
import DD from '@/components/Son2Son/DD'
import Vue from "vue";

Vue.component('GG', GG)
Vue.component('DD', DD)
export default {
  data() {
    return {
      comName: 'GG'
    }
  },
  methods: {}
}
</script>
```

## 锚链接及常规url的区别

1.普通的 URL 地址：会刷新整个页面；会追加浏览历史记录；

2.锚链接：不会触发页面的整体刷新；会追加浏览历史记录；（锚链接时页面内的跳转）

## SPA相关概念

- 概念定义：SPA --> Single Page Application，单页面应用程序；即只有一个Web页面的网站，网站的所有功能都在这个唯一的页面上进行展示与切换。
- 特点
  - 只有一个页面
  - 浏览器一开始请求这个页面，必须加载对应的 HTML，CSS，JavaScript
  - 用户的所有操作，都在这唯一的页面上完成
  - 页面数据都是用Ajax请求回来的
- 好处
  - 实现了前后端分离开发，各司其职；提高了开发效率；
  - 用户体验好、快，内容的改变无需重新加载整个页面；

## 原始实现SPA



# 路由

# watch computed



# vue-cli



# 极客时间Vue

