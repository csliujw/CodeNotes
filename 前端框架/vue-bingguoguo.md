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

> ==v-bind==

> ==v-for==

> ==v-if /v-show==

