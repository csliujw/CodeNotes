# 引言

SpringBoot开发后端。后端注意要处理跨域问题。也可以直接用我的接口`www.baobaoxuxu.cn/api`

Vue开发前端。

# 技术组成

后端

- SpringBoot + MyBatis + Druid

  或

- SpringBoot + Spring JDBC Template + Drudi（有空就给）

前端

- Vue + axios + ElementUI

# 环境搭建

## 创建vue项目

- 在命令行中输入 vue create 项目名
- 选择必要的插件（用space键勾选）不要勾选严格模式 ESLin
  - 选择vuerouter
  - vue就可以了
  - css预编译看你用不用，不用就不选。我们用elementui基本不自己写css了。
- 回车

<img src="..\pics\sb_vue\vue_dev_01.png" style="margin:left">

<img src="..\pics\sb_vue\vue_dev_02.png" style="margin:left"><img src="..\pics\sb_vue\vue_dev_03.png" style="margin:left">

## vue中引入axios

<a href="http://www.axios-js.com/">axios参考文档</a>

我们使用axios发起ajax请求。也可用vue的resource替代。

在创建好的项目（manager）中安装axios

```cmd
npm install --save axios vue-axios
```

<img src="..\pics\sb_vue\vue_dev_04.png" style="margin:left">

在main.js中引入axios，并注册到vue上。

```js
import Vue from 'vue'
import axios from 'axios'
import VueAxios from 'vue-axios'
Vue.use(VueAxios, axios)
```

简单的使用axios进行测试.

在任意组件中进行了一下测试，发现没有问题。（是我自己写的一个接口）

```js
created() {
    this.$http.get("http://localhost:8080/api/").then(
        (body) => {
            console.log(body);
        }, (error) => {
            console.log("error")
        });
}
```

## 安装elementui

<a href="https://element.eleme.cn/#/zh-CN/component/installation">elementui官方文档</a>

- 在项目中安装`npm i element-ui -S`

- main.js中引入代码

  ```JS
  import ElementUI from 'element-ui';
  import 'element-ui/lib/theme-chalk/index.css';
  
  Vue.use(ElementUI);
  ```

## 创建SpringBoot项目

大家都会，我就不写了。

## 可跨域访问

- 配置类

```java
package com.baobaoxuxu.config;

/**
 * @author payphone
 * @date 2020/7/8 17:13
 * @Description 项目跨域配置。
 */

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CrossConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins("*")
                .allowedMethods("GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS")
                .allowCredentials(true)
                .maxAge(3600)
                .allowedHeaders("*");
    }
}
```

- 在每个Controller类上，加上类级别的注解@CrossOrigin
- PS：我不知道是不是两个要一起用，还是任意一个就可以 - -。当时浏览器有缓存，浏览器控制台一直报跨域的错误。后面就没深究了~~~

# Vue项目结构

## 大致介绍

<img src="..\pics\sb_vue\vue_dev_05.png" style="margin:left">

个人理解

组件是那些可以被重复利用的才封装成组件。

views就是各种各样的页面。如CSDN的首页是一个页面。文章的详细信息是一个页面。页面中可以用各种各样的组件。

所有的views都会挂到App.vue上，通过用户的点击，切换不同的views页面。

App.vue代码是不会被浏览器识别的，所以最后需要变成html，渲染到某个html页面，index.html就是这个页面。

----

App.vue index.html这些是脚手架（vue-cli）为我们提供的默认配置，我们可以修改的，修改方式请看<a href="https://cli.vuejs.org/zh/config/#filenamehashing">Vue Cli</a>

# 项目介绍

## 内容

写一个简单的CRUD。包含登录页面，注册页面，数据展示页面，数据的增删改查。

所有组件都在一个页面。登录成功后，导航栏显示用户名。注册成功是，消息提示。

用户登录后才可进行数据的增加，删除，修改。

## 组件分配

- 导航栏一个组件。
  - 包含登录，注册模态框。

- 页面数据一个组件。
  - 包含数据查询，新增的按钮
  - 包含数据展示页面，各条数据的删除，修改操作。

# 项目代码

## 边学习，边Coding

vue的基本语法和脚手架的结构我都看了一遍。所以不带着看了。但是vue的父子组件传值，子父传值我还是会着重说明的（因为我也不熟悉）。elementui我是没学过，一起看文档学。hahahahaha。

## 导航栏页面

### <a href="https://element.eleme.cn/#/zh-CN/component/layout">基本布局</a> 

**elememtui分为24栏布局**

- gutter 表示每格的间距
- span表示占几栏
- offset表示偏移多少栏

```html
    <el-row :gutter="20">
        <el-col :span="16">
            <div>首页</div>
        </el-col>
        <el-col :span="8">
            <el-col :span="4" :offset="10" v-show="user==null"><a>登录</a></el-col>
            <el-col :span="4" v-show="user==null"><a>注册</a></el-col>
            <el-col :span="10" :offset="6" v-show="user!=null"><a>欢迎用户</a></el-col>
            <el-col :span="6" v-show="user!=null"><a>退出登录</a></el-col>
        </el-col>
    </el-row>
```

### <a href="https://element.eleme.cn/#/zh-CN/component/dialog">对话框</a>

- 登录模态框

  ```html
  asdf
  ```

  

- 注册模态框



