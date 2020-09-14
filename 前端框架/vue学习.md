# 引言

先安装node.js

<h2 style="color:red">创建vue项目的时候千万别选严格代码模式！！！</h2>



# Vue基础知识请看官网

## 官网内容

<h3><a href='https://cn.vuejs.org/v2/guide/'>vue官网</a></h3>

## 自行整理的内容

----

# Vue组件化开发

## 安装必备环境。

- 安装一个VSCode 或 HBuilder
- 安装node.js
- npm安装vue，若安装过慢，请百度npm更换镜像源。

## 创建vue项目

**以VSCode为例**

- 命令行创建

在终端输入`vue create 项目名`

选择需要的选项

创建即可。

- 图形界面创建

在终端输入 vue ui

赋值下方显示的地址，在浏览器中访问

自己倒腾界面的功能。

## Vue项目内容概述【单页面】

### 文件目录概述

单页面配置

<img src="..\pics\vue\single_page.png">

<img src="..\pics\vue\single_page2.png">

### 启动Vue项目

进入项目（cd demo1）

输入启动命令（npm run serve）

### 部分代码解释

main.js中的代码

```js
// 这句话不知道啥意思
import Vue from 'vue'
// 导入我们自己写的vue代码
import App from './App.vue'

Vue.config.productionTip = false
// 把App.vue中的代码 挂载到index.html中的<div id="app"></div> 中
new Vue({
  render: h => h(App),
}).$mount('#app')
```

App.vue中的代码

```vue
<template>
  <div id="app">
    <img alt="Vue logo" src="./assets/logo.png">
    <HelloWorld msg="Welcome to Your Vue.js App"/>
  </div>
</template>

<script>
// 导入我们写的子组件 HelloWorld.vue
import HelloWorld from './components/HelloWorld.vue'

export default {
  name: 'App',
  // 把HelloWorld组件作为App的子组件
  components: {
    HelloWorld
  }
}
</script>
```

index.html中的代码

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <link rel="icon" href="<%= BASE_URL %>favicon.ico">
    <title><%= htmlWebpackPlugin.options.title %></title>
  </head>
  <body>
    <noscript>
      <strong>We're sorry but <%= htmlWebpackPlugin.options.title %> doesn't work properly without JavaScript enabled. Please enable it to continue.</strong>
    </noscript>
    <div id="app"></div>
    <!-- built files will be auto injected -->
  </body>
</html>
```

## Vue的多页面应用配置

### 初始化的注意

<a href="https://www.cnblogs.com/guiltyWay/p/10320653.html">我参考的这篇博客</a>

初始化一个最简单的默认vue项目

<img src="..\pics\vue\mutil_pages.png" style="margin-left:0px">

为项目添加vue.config.js配置文件，如下

<img src="..\pics\vue\mutil_pages2.png" style="margin-left:0px">

vue.config.js的内容如下：

```vue
module.exports = {
  publicPath: './'
}
```

### 具体的多页面配置解释

<img src="..\pics\vue\mutil_pages3.png" style="margin-left:0px">

**vue.config.js**中的配置文件

```js
module.exports = {
    publicPath: '',
    pages: {
        index: {
            // page 的入口 js文件的路径
            entry: "src/index/main.js",
            // 模板来源 把代码渲染到的页面？
            template: "public/index.html", 
            // 在 dist/index.html 的输出  build项目的时候 输出的html的名称
            filename: "index.html",
            // 当使用 title 选项时，
            // template 中的 title 标签需要是 <title><%= htmlWebpackPlugin.options.title %></title>
            title: "Index Page",
            // 在这个页面中包含的块，默认情况下会包含
            // 提取出来的通用 chunk 和 vendor chunk。
            chunks: ["chunk-vendors", "chunk-common", "index"]
        },
        user: {
            // page 的入口  
            entry: "src/user/user.js",
            // 模板来源 把代码渲染到的页面？
            template: "public/user.html", 
            // 在 dist/index.html 的输出 build项目的时候 输出的html的名称
            filename: "user.html",
            // 当使用 title 选项时，
            // template 中的 title 标签需要是 <title><%= htmlWebpackPlugin.options.title %></title>
            title: "user Page",
            // 在这个页面中包含的块，默认情况下会包含
            // 提取出来的通用 chunk 和 vendor chunk。
            chunks: ["chunk-vendors", "chunk-common", "user"]
        }
    }
}
```



# Vue与less

## less的安装和基本案例

重复利用`CSS`样式【`CSS`代码复用】

less写完后需要转换为`CSS`

<a href="https://less.bootcss.com/">less的安装【官网】</a>

> **基本案例**

```less
@red:red;
// 全局修改。妙啊。代码复用。 
// 可以作为一门语言来学了。【我后端，不学太多】
@other:#FFAA;
body{
    background-color: @other;
    color: @other;
}
h1{
    color:@red;
}
```

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Document</title>
    <!-- link 和 script要引用，如果想直接用less的话，一般是会把less编译 -->
    <link rel="stylesheet/less" href="less.less">
    <script src="//cdnjs.cloudflare.com/ajax/libs/less.js/3.8.1/less.min.js"></script>
</head>

<body>
    <h1>12312</h1>
</body>

</html>
```

## less基础语法（后端人员）

### CSS基础选择器回顾

```css
/* id选择器 */
#div{}
/* class选择器 */
.div{}
/* tag选择器 */
body{}

.div 的用法
<a class="div"></a>
```



一个后端，为什么要学less？？？

<a href="[https://less.bootcss.com/#%E6%A6%82%E8%A7%88](https://less.bootcss.com/#概览)">官方文档，最好的教材</a>

学点变量复用就好，函数那些就不学了。

- 变量

```less
@width: 10px;
@height: @width + 10px;

#header {
  width: @width;
  height: @height;
}

// 编译为
#header {
  width: 10px;
  height: 20px;
}

------------------------------------------------
可以通过`@{变量名}`取出变量中的值
// Variables
@my-selector: banner;

// Usage
.@{my-selector} {
  font-weight: bold;
  line-height: 40px;
  margin: 0 auto;
}
// 编译为
.banner {
  font-weight: bold;
  line-height: 40px;
  margin: 0 auto;
}
```

<hr>

- 混合：一个样式调用另一个样式

```less
@font:50px;
@bgcolor:white;
@margin:10px,0px;

@color:red;
// less的复用的写法 推荐加上括号，避免和css混淆？
.div(){
    font-size: @font;
    background-color: @bgcolor;
    margin: @margin;
}
// 调用其他的样式，样式中可包含定义的变量
.li{
    .div();
    color: @color;
}
```

<hr>

- 嵌套：简化了一些css的写法，如为一个标签 设置伪类【大概就这个意思】

```less
@entercolor:red;
li{
    color: greenyellow;
    // & 表示当前选择器的父级即li
    &:hover{
        color: @entercolor;
    }
}
```

```less
#header {
  color: black;
}
// 选择 id="header" 的所有class包含 navigation的元素。
#header .navigation {
  font-size: 12px;
}
#header .logo {
  width: 300px;
}

// less 写法
#header {
  color: black;
  .navigation {
    font-size: 12px;
  }
  .logo {
    width: 300px;
  }
}
```



<hr>

- 冒泡  不写

<hr>

- 参数？？

```less
// 事先定义一个  单独用我的代码没出效果
.demo(@c:red){
    width:20px;
    color: @c;
}
// 在div tag中调用才有效
div{
    .demo();
}
div{
    .demo(blue);
}
```

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Document</title>
    <link rel="stylesheet/less" href="less.less">
    <script src="//cdnjs.cloudflare.com/ajax/libs/less.js/3.8.1/less.min.js"></script>
</head>

<body>
    <h1 class="demo">demo</h1>
    <hr>
    <div><span class="demo">123</span></div>

</body>

</html>
```



## `vue`中less的使用

- 语法检测好严格

就是刚刚的那些语法那样用。具体的style标签引入 有模板 看App.vue即可



# Vue之router

## 创建项目

Use history mode for router? (Requires proper server setup for index fallback in production) (Y/n) 

选yes，后端每次都需要返回index页面，选n，就不需要返回index了，直接给数据就行。

## 路由

映射到对应的组件。组件中内容的显示用

```vue
<router-view></router-view>
```

## 嵌套路由

一个路由里可以嵌套其他路由。意义如图：

<img src="..\pics\vue\vue_router01.png">

## 小结

<img src="..\pics\vue\vue_router02.png" >

```html
<!-- 官方代码 -->
<script src="https://unpkg.com/vue/dist/vue.js"></script>
<script src="https://unpkg.com/vue-router/dist/vue-router.js"></script>

<div id="app">
  <h1>Hello App!</h1>
  <p>
    <!-- 使用 router-link 组件来导航. -->
    <!-- 通过传入 `to` 属性指定链接. -->
    <!-- <router-link> 默认会被渲染成一个 `<a>` 标签 -->
    <router-link to="/foo">Go to Foo</router-link>
    <router-link to="/bar">Go to Bar</router-link>
  </p>
  <!-- 路由出口 -->
  <!-- 路由匹配到的组件将渲染在这里 -->
  <router-view></router-view>
</div>
```

## 路由基本案例

### 案例简介

> **该案例包含的内容如下**

- 静态路由匹配
- 动态路由匹配
- 单个路由视图的渲染
- 路由嵌套
  - user
    - user/center
  - article
    - article/detail
- 多个路由视图的渲染
  - 有若干个router-view 同时被渲染
- 重定向
- 路由组件传参

### 案例场景

<img src="..\pics\vue\vue.png">

**左边的大方框**

导航栏为一个组件

文章列表为一个组件。最新 |  热门 | xx | xx 用静态路由实现点击后显示不同类型的文章。

**右边的大方框**

导航栏，文章详细数据，快速导航栏均用多视图实现。

 

# Vue案例知识

## 子组件调用父组件的方法。

子组件自定义一个方法，vue事件绑定子组件的方法，然后子组件方法的内部调用父组件的方法

```vue
<template>
  <nav>
    <span>首页</span> |
    <!-- 绑定子组件的方法 -->
    <span v-if="user==null" @click="login">登录 |</span>
    <span v-if="user==null">注册 |</span>
    <span v-if="user!=null">欢迎你：{{user.username}} |</span>
    <span v-if="user!=null">
      <router-link to="/user/center">个人中心</router-link>
    </span>
  </nav>
</template>

<script>
export default {
  name: "nav",
  props: {
    user: Object,
  },
  methods: {
      login: function(){
          // 子组件的login调用父组件的login方法
          this.$parent.login();
      }
  }
};
</script>
```

