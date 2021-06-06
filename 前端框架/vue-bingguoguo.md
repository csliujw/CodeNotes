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

## 生命周期

## 异步

## axios

## 案例

