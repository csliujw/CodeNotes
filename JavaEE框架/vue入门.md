# Vue基础
## Vue入门用法
### 最简单的例子
绑定的方式支持id(#xx) class(.xx) 元素(xx)

el是element的缩写，是元素的挂载点

data是数据的存放点，用json格式。（应该说是对象，以为json是字符串，而data是一个对象！）
```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>
    <div id="app">
        {{ message }} 模板引擎语法 把data中的message的值输出在这里
    </div>

    <!-- 开发环境版本，包含了有帮助的命令行警告 -->
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        var app = new Vue({
            el: '#app',
            data: {
                message: 'Hello Vue！hhaha！!'
            }
        })
    </script>
</body>

</html>
```

### el挂载点
**vue实例的作用范围**
除了body标签和html标签其他都可

### data数据对象
> 复杂数据绑定的访问，遵循js语法即可。

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>

    <div id="multi">
        <div id="simpleData">{{name}}</div>
        <div id="array">{{hobby[0]}}</div>
        <div id="object">{{firends.red.age}}</div>
    </div>
    <!-- 开发环境版本，包含了有帮助的命令行警告 -->
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        // 复杂数据的绑定
        var multis = new Vue({
            el: "#multi",
            data: {
                name: "liujiawei",
                age: "18",
                hobby: ['play', 'dnf', 'game'],
                firends: {
                    red: {
                        age: 18,
                        name: "red"
                    }
                }
            }
        });
    </script>
</body>

</html>
```
### v-text插值表达式
ps:div换成h标签的话 内容无法显示！！
```html
<body>
    v-text 设置textContent<br />
    <div id="vtext">
        <h4 v-text="message"></h4>
        <h4 v-text="info"></h4>
        <h5>{{message+' split'}}</h5>
    </div>


    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        new Vue({
            el: "#vtext",
            data: {
                message: "hello v-text",
                info: "This is information"
            }
        });
    </script>
</body>
```
### v-html
**v-html设置元素的innerHTML，其内容会被解析为html内容噢**
```html
<body>
    v-text 设置textContent<br />
    <div id="vtext">
        <h4 v-html="message"></h4>
        <h4 v-html="info"></h4>
        <h5>{{message+' split'}}</h5>
    </div>


    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        new Vue({
            el: "#vtext",
            data: {
                message: "<div style='color:red'>hello v-text</div>",
                info: "<h1>haha</h1>"
            }
        });
    </script>
</body>
```
### v-on事件绑定
基础语法v-on:事件名="方法"
```html
<body>
    <div id="on">
        <h4 v-html="message"></h4>
        <button v-on:click="fn1">alert</button>
        <button @click="fn2">console</button>
        <button @dblclick="fn3">dblclick</button>
    </div>
    <!-- 生产环境版本，优化了尺寸和速度 -->
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        new Vue({
            el: "#on",
            data: {
                message: "<div style='color:red'>hello v-text</div>",
                info: "<h1>haha</h1>",
                obj: {
                    name: "hello"
                }
            },
            methods: {
                fn1: function() {
                    alert(123);
                },
                fn2: function() {
                    console.log("hello world java!");
                },
                fn3: function() {
                    console.log("db click");
                    // 触发该事件后 修改message的值
                    this.message = this.obj['name']
                    // 或this.message = this.obj.name
                }
            }
        });
        new Vue({

        });
    </script>
</body>
```

### 综合例子：计数器
巩固on事件
```html
<body>
    <div id="count">
        <button @click="add">+</button>
        <span>{{init}}</span>
        <button @click="sub">-</button>
    </div>
    <!-- 生产环境版本，优化了尺寸和速度 -->
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        let app = new Vue({
            el: "#count",
            data: {
                init: 0,
                max: 20
            },
            methods: {
                add: function() {
                    if (this.init < this.max) {
                        this.init++;
                    } else {
                        alert("最多为20噢！");
                    }
                },
                sub: function() {
                    if (this.init > 0) {
                        this.init--
                    } else {
                        alert("最小为0噢！");
                    }
                }
            }
        });
    </script>
</body>
```

### 判断
#### v-show
设置元素是否显示，原理为：display:none
```html
<body>
    <div id="show">
        <button @click="disapper">点击消失</button>
        <span v-show="isShow">我是谁！</span>
        <button @click="show">点击显示</button>
    </div>
    <!-- 生产环境版本，优化了尺寸和速度 -->
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        new Vue({
            el: "#show",
            data: {
                isShow: true
            },
            methods: {
                disapper: function() {
                    this.isShow = false;
                },
                show: function() {
                    this.isShow = true;
                }
            }
        });
    </script>
</body>
```
#### v-if
设置元素是否保留，原理为：操作dom，不保留就删除dom元素

- 不同用户显示不同标签
```html
<body>
    <div id="if">
        <button @click="change">点击更改权限</button>
        <button @click="changeName">修改姓名为：lalala</button>
        <button @click="recoverName">恢复初试姓名</button>

        <h4 v-if="power">欢迎进入</h4>
        <h4 v-if="!power">你未成年，不能进去噢！</h4>
        <h4 v-if="name =='lalalala'">Who are you!</h4>
    </div>
    <!-- 生产环境版本，优化了尺寸和速度 -->
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        new Vue({
            el: "#if",
            data: {
                name: "hello",
                oldName: "hello",
                power: true,
                address: "未知！"
            },
            methods: {
                change: function() {
                    this.power = !this.power;
                },
                changeName: function() {
                    this.name = "lalalala";
                },
                recoverName: function() {
                    this.name = this.oldName;
                }
            }
        });
    </script>
</body>
```
### v-bind
绑定元素属性。如title，src，class等

- 激活class，让class失效
- 切换图片
- 更改标签颜色等等
```html
<body>
    <h2>设置元素属性</h2>
    <div id="id">
        <img v-bind:src="img_src" v-bind:title="img_title" v-bind:class="img_class" alt="">
        <div></div>
        <img v-bind:src="img_src" v-bind:class="img_class" alt="">
        <div></div>
        <button @click="larger">图片放大</button>
        <button @click="small">图片缩小</button>
        <img v-bind:src="img_src" v-bind:class="{active:isActive}" alt="">
    </div>
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        new Vue({
            el: "#id",
            data: {
                img_src: "logo.png",
                img_title: "hello world png",
                img_class: "active",
                isActive: true
            },
            methods: {
                larger: function() {
                    this.isActive = false;
                },
                small: function() {
                    this.isActive = true;
                }
            }
        });
    </script>
</body>
```

### v-for
循环遍历,如拿到数据，对数据进行遍历显示
```html
<body>
    <div id="for">
        <h3 v-for="ele in array">{{ele}}</h3>
        <hr> 加索引噢
        <h3 v-for="(ele,index) in array">{{ele}},{{index+1}}</h3>
        <hr> 复杂数据遍历
        <h3 v-for="(element,key) in friends">
            <a href="#">{{key}}</a>
            <span>{{element.name}}</span>
            <span>{{element.age}}</span>
            <span>{{element.address}}</span>
            <hr>
        </h3>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#for",
            data: {
                array: ["xiaohong", "xiaolang", "xiaolu", "xiaoa"],
                friends: {
                    red1: { name: "read!",age: 18,address: "jxpx"},
                    red2: {name: "read22!",age: 18,address: "jxpx22"},
                    red3: {name: "read!333",age: 18,address: "jxpx333"}
                }
            }
        });
    </script>
</body>
```
### v-model
双向数据绑定【表单元素】
#### 注意事项
v-model 会忽略所有表单元素的 value、checked、selected attribute 的初始值而总是将 Vue 实例的数据作为数据来源。你应该通过 JavaScript 在组件的 data 选项中声明初始值。
#### 普通文本的绑定
用一个v-model就可以了。文本内容改变也会体现到data里面去！
```html
<body>
    <!--綁定普通文本-->
    <form id="text">
        <input type="text" v-model="text"><br> 实时显示:{{text}}
        <hr>
        <textarea v-model="bigText"></textarea>
    </form>


    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#text",
            data: {
                text: "12313",
                bigText: "!@#!@#!@#!@#!@#@!@"
            }
        });
    </script>
</body>
```
#### 单选按钮
- 常规单选写法
radioData对应的value如果与当选框中的值一致，则该单选框被默认选中
```html
    <!--綁定普通文本-->
    <form id="radio">
        吃饭<input type="radio" name="hobby" value="吃" v-model="radioData"> 睡觉
        <input type="radio" name="hobby" value="睡" v-model="radioData"> <br> {{radioData}}
    </form>

    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#radio",
            data: {
                text: "12313",
                bigText: "!@#!@#!@#!@#!@#@!@",
                radioData: '吃'
            }
        });
    </script>
```
- v-for遍历写法
假设数据是从服务器请求过来的，需要动态遍历生成多个单选框！

默认选择的写法也是可以通过v-model来双向绑定的，同样是value一致的被选中。也可以采用特殊写法，

例子采用特殊写法，需要注意！【**同时 需要为标签加上v-model才可以双向绑定！**】
```html
<body>
    <!--綁定普通文本-->
    <form id="vforRadio">
        for循环下radio的默认选中<br>
        <label v-for="it in radioData">
            {{it.name}}<input type="radio" name="hobby" v-bind:checked="it.name==radio"  v-bind:value="it.value" >
        </label>
        <br>
        <label>
            普通的默认选中：<input type="radio" v-model="radio"  value="睡">
        </label>
    </form>

    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#vforRadio",
            data: {
                radioData: [{
                    name: "吃",
                    value: 1
                }, {
                    name: "睡",
                    value: 2
                }],
                radio: '睡'
            }
        });
    </script>
</body>
```
#### 多选按钮
- 普通写法和v-for生成写法

普通写法判断是否默认选择 也是绑定 v-model="xx"就可以了

for循环生成的写法也是一样，也可以采用特殊写法，

例子采用特殊写法需要借用函数判断indexOf(it.name)!=-1
```html
<body>
    <form id="vforChecked">
        普通多选框默认选中： 睡
        <input type="checkbox" name="hb" value="睡" v-model='checkeds'> 吃
        <input type="checkbox" name="hb" value="吃" v-model='checkeds'>
        <br> for循环默认选中
        <label v-for="it in CheckedData">
            {{it.name}}
            <input type="checkbox" name="hobby" v-bind:value="it.value" v-bind:checked="checkeds.indexOf(it.name)!=-1">
        </label>
    </form>

    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#vforChecked",
            data: {
                CheckedData: [{
                    name: "吃",
                    value: 1
                }, {
                    name: "睡",
                    value: 2
                }],
                checkeds: ['睡', '吃']
            }
        });
    </script>
</body>
```
#### 下拉框
下拉框的普通方式和v-for生成都可以用v-model，特殊写法不考虑了！
```html
<body>
    <!--綁定普通文本-->
    <form id="selectEle">
        普通下拉框默认选中：
        <select v-model="selectEle">
            <option value="1">111</option>
            <option value="2">222</option>
            <option value="3">333</option>
            <option value="4">444</option>
            <option value="5">555</option>
        </select> {{selectEle}}

        <br> v-for生成的下拉框
        <br>
        <select v-model="selectEle">
            <option>please select</option>
            <option v-for="it in selectData" v-bind:value="it.value">{{it.name}}</option>
        </select> {{selectEle}}
    </form>


    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#selectEle",
            data: {
                selectEle: 2,
                selectData: [{
                    name: "吃",
                    value: 1
                }, {
                    name: "睡",
                    value: 2
                }, {
                    name: "玩",
                    value: 3
                }, {
                    name: "乐",
                    value: 4
                }],
            }
        });
    </script>
</body>
```
#### 方法v-on的补充
形式参数
```html
<body>
    <div id="on">
        <button @click="fn3(message)">click</button>
    </div>
    <!-- 生产环境版本，优化了尺寸和速度 -->
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script>
        new Vue({
            el: "#on",
            data: {
                message: "<div style='color:red'>hello v-text</div>",
            },
            methods: {
                fn3: function(text) {
                    console.log("db click" + text);
                    // this.message = this.obj['name']
                }
            }
        });
    </script>
</body>
```
键盘事件
#### 总结
v-xx的里面可以用js的函数噢！

## Vue组件化

### 组件化基本案例

```html
<body>
    自定义组件<br>
    <div id="div">
        <btn-com></btn-com>
        <btn-com></btn-com>
        <btn-com></btn-com>
        <btn-com></btn-com>
        <btn-com></btn-com>
        <btn-com></btn-com>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        // 创建组件，组件需要在Vue对象的实例中使用，充当示例的子元素
        Vue.component("btn-com", {
            data: function() {
                return {
                    count: 0
                };
            },
            // 时间中也可直接对元素进行操作 组件化，可以让实现代码复用，非常强大 而且不同组件之间不影响
            template: '<button @click="add">你点击了该按钮{{count}}次</button>',
            methods: {
                add: function() {
                    this.count++;
                    console.log(123);
                }
            }
        });
        new Vue({ el: "#div" });
    </script>
</body>
```

----

```html
<body>
    <div id="tt">
        <two v-for="i in message" v-bind:wtf="i">
        </two>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/vue/dist/vue.js"></script>
    <script>
        // 定义组件
        Vue.component("two", {
            props: ['wtf'], // props是自定义属性
            template: '<li>{{wtf.name}}</li>'
        });
        new Vue({
            el: "#tt",
            data: { message: [{ name: "kk" }, { name: "kkk" }] }
        });
    </script>
</body>

out put
黑点 kk
黑点 kkk
```

组件化总结：定义一个标签模板，需要显示那些数据，用自定义的props属性来获取值，然后使用组件，为这个组件赋值date属性。属性的绑定用的v-band:属性名 如`v-ban:wtf`