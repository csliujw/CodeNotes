# 笔记介绍

看黑马vue视频做的笔记。

## 课程介绍

前5天： 都在学习Vue基本的语法和概念；打包工具 Webpack , Gulp
后5天： 以项目驱动教学；


### 什么是Vue.js

+ Vue.js 是目前最火的一个前端框架，React是最流行的一个前端框架（React除了开发网站，还可以开发手机App， Vue语法也是可以用于进行手机App开发的，需要借助于Weex）

+ Vue.js 是前端的**主流框架之一**，和Angular.js、React.js 一起，并成为前端三大主流框架！

+ Vue.js 是一套构建用户界面的框架，**只关注视图层**，它不仅易于上手，还便于与第三方库或既有项目整合。（Vue有配套的第三方类库，可以整合起来做大型项目的开发）

+ 前端的主要工作？主要负责MVC中的V这一层；主要工作就是和界面打交道，来制作前端页面效果；

## 框架和库的区别

 + 框架：是一套完整的解决方案；对项目的侵入性较大，项目如果需要更换框架，则需要重新架构整个项目。

  - node 中的 express；

 + 库（插件）：提供某一个小功能，对项目的侵入性较小，如果某个库无法完成某些需求，可以很容易切换到其它库实现需求。

  - 1. 从Jquery 切换到 Zepto
  - 2. 从 EJS 切换到 art-template

## Vue的优点和MVVC

 + MVC 是后端的分层开发概念；
 + MVVM是前端视图层的概念，主要关注于 视图层分离，也就是说：MVVM把前端的视图层，分为了 三部分 Model, View , VM ViewModel

 + 在Vue中，一个核心的概念，就是让用户不再操作DOM元素，解放了用户的双手，让程序员可以更多的时间去关注业务逻辑；

<img src="..\pics\vue\heima\01.MVC和MVVM的关系图解.png" style="margin:left">



# Vue.js - Day1



## Vue基本语法

### vue入门

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="./js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<body>
		<div id="app">
			<!-- v-cloak可以解决插值表达式得闪烁问题 -->
			<p v-cloak>{{username}}</p>
			<!-- v-text默认没有闪烁问题 -->
			<p v-text="username"></p>
			<!-- 拼接字符串 -->
			<p v-text="'123123'+username"></p>
			<!-- v-html可以解析html代码 -->
			<p v-html="htmlCode"></p>
			
		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app', // 当前的vue实例要控制页面上的那个区域
			data: { // data属性中，存放的是el中要用到的数据
				username: 'ljw', // 通过vue中的指令 可以很方便的把数据渲染到页面上，不用我们手动操作DOM
				htmlCode: '<h1>I am H1</h1>'
			}
		});
	</script>

</html>

```

- 总结
  - {{xx}} 存在闪烁问题。可以用cloak解决闪烁问题。
  - v-text/v-html不存在闪烁问题。
  - v-text不能解析html代码片段，v-html可以解析html代码片段

### v-bind

v-bind是vue提供的绑定属性的指令。可以绑定标签有的属性和我们自己定义的属性。

```html
<body>
    <div id="app">
        <p v-cloak>{{username}}</p>
        <p v-text="username"></p>
        <!-- v-bind绑定属性的指令 会把""里面的当作js代码/表达式来执行 -->
        <input type="button" value="按钮" v-bind:title="btn_title+'String'" />
        <!-- 简写方式 -->
        <input type="button" value="按钮" :title="btn_title+'String'" />
    </div>
</body>
<script>
    // 创建vue实例
    var vm = new Vue({
        el: '#app', 
        data: { 
            username: 'ljw',
            htmlCode: '<h1>I am H1</h1>',
            btn_title: 'I am title'
        }
    });
</script>
```

### v-on

#### v-on 事件绑定机制

```html
<body>
    <div id="app">
        <p v-cloak>{{username}}</p>
        <p v-text="username"></p>

        <input type="button" value="按钮" :title="btn_title+'String'" />
        <!-- v-on绑定事件 -->
        <input type="button" value="按钮" :title="btn_title+'String'" v-on:click="show" />
        <!-- 简写 我觉得方法加个括号比较好，方便识别-->
        <input type="button" value="按钮" :title="btn_title+'String'"  @click="show()"/>

    </div>
</body>
<script>
    // 创建vue实例
    var vm = new Vue({
        el: '#app', 
        data: { 
            username: 'ljw',
            htmlCode: '<h1>I am H1</h1>',
            btn_title: 'I am title'
        },
        methods: {
            show: function(){
                alert("Hello");
            }
        }
    });
</script>
```

#### 事件绑定的修饰符

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<style type="text/css">
		.inner {
			height: 200px;
			width: 800px;
			background-color: antiquewhite;
		}
	</style>
	<body>
		<div id="app">
			<div class="inner" @click="divHander()">
				<!-- 默认冒泡机制。点了内层的，会在自动调用外层。可以用@click.stop阻止向上冒泡
					 这样就不会触发外层的div事件了。
				 -->
				<input type="button" name="" id="" value="点他" @click="btnHander()" />
				<!-- stop 阻止冒泡，不会触发父tag的事件噢 -->
				<input type="button" name="" id="" value="点我" @click.stop="btnHander()"/>
				<!-- prevent 阻止默认事件，如a的跳转 再加一个stop阻止事件冒泡。 prevent和stop的顺序没关系噢 -->
				<a href="https://www.baidu.com" @click.prevent.stop="aHander()">你好啊</a>
			</div>
		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			methods: {
				divHander: function() {
					console.log("~~~~~~~~~~~~~~~");
					console.log("这是触发了div的点击事件");
				},
				btnHander: function() {
					console.log("~~~~~~~~~~~~~~~");
					console.log("这是触发了按钮的点击事件");
				},
				aHander: function(){
					console.log("a的跳转被阻止了");
				}
			}

		})
	</script>
</html>
```

- 总结

  - v-on:click='方法名' 绑定我们自己定义的方法

  - methods是一个对象？

  - @click 简写

    ----

  - .stop 阻止事件冒泡。

  - .prevennt 阻止默认事件。

  - .captuer 添加事件侦听器时使用事件捕获模式。

  - .self 只当事件再该元素本身触发时才能触发。

  - .once 事件只触发一次。

  - stop和self的区别

    - stop是阻止自身向外的冒泡机制，而self只负责自己，不管别人。比如孙子，儿子，父亲。儿子用self，孙子被点了向外冒泡，儿子不会响应，父亲会响应。因为儿子用了self只会响应自身，但是不管别人。
    - self只会阻止自己身上冒泡行为的触发，并不会真正阻止 冒泡的行为。

- 小案例，走马灯

  ```html
  <!DOCTYPE html>
  <html>
  	<head>
  		<meta charset="utf-8" />
  		<title></title>
  		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
  	</head>
  	<body>
  		<div id="app">
  			<button type="button" @click="go()">滚动起来</button>
  			<button type="button" @click="go()">别滚了</button>
  			<marquee direction="left">{{'H5自带的走马灯'+msg}}</marquee>
  			<h4>{{msg}}</h4>
  		</div>
  	</body>
  	<script>
  		// 创建vue实例
  		var vm = new Vue({
  			el: '#app', 
  			data: { 
  				msg: '猥琐发育，别浪~~~~~'
  			},
  			methods: {
  				go: function(){
  					var _this = this;
  					// 如果用()=>那么内部的this和外部的保持一致。【箭头函数的this永远指向其父作用域】
  					// 如果用的function(){}写法则不一致，需要用变量保存this。让function内部访问
  					setInterval(()=>{
  						_this.msg;
  						let start = _this.msg.substring(0,1);
  						let end = _this.msg.substring(1,_this.msg.length);
  						_this.msg = end+start;
  					},200);
  
  				}
  			}
  			
  		})
  	</script>
  </html>
  ```

### v-model

v-model 双向数据绑定

**用于表单元素噢！在表单tag上加上v-model即可**

```html
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<body>
		<div id="app">
			<h2>{{msg}}</h2>
			<input type="text" v-model="msg" />
			<hr >
			绑定单选按钮====默认绑定一个值
			<input type="radio" name="radio" value="1" v-model="radiolist"  />1
			<input type="radio" name="radio" value="2" v-model="radiolist"  />2
			<hr >
			绑定单选按钮====默认为空
			<input type="radio" name="radio" value="1" v-model="radio2list"  />11
			<input type="radio" name="radio" value="1" v-model="radio2list"  />22
			<hr >
			绑定复选框
			<input type="checkbox" name="ck" value="game" v-model="ck" />game
			<input type="checkbox" name="ck" value="play" v-model="ck" />play
			<hr >
			绑定select
			<select name="select" v-model="select">
				<option value="1">1</option>
				<option value="2">2</option>
				<option value="3">3</option>
			</select>
		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				msg: 'Hello World Vue.js',
				radiolist: '1',
				radio2list: '',
				ck: ['game','play'],
				select: '3'
			}
		});
	</script>
</html>
```

**总结**

- 单一值的绑定 对应的是一个字符串或数组之类的 如：radio，input
- 多值的绑定对应的是 数组 如：复选框
- 常见的使用场景，表单的数据在vue实例中，用户只需操作vue实例中的数据即可与表单中的数据保持一致。

### vue中使用样式

#### 使用class

1. **数组**

```vue
<h1 :class="['red', 'thin']">这是一个邪恶的H1</h1>
```

2. **数组中使用三元表达式**

```vue
<h1 :class="['red', 'thin', isactive?'active':'']">这是一个邪恶的H1</h1>
```

3. **数组中嵌套对象** ==推荐==

```vue
<h1 :class="['red', 'thin', {'active': isactive}]">这是一个邪恶的H1</h1>
```

4. **直接使用对象 key为class，value为激不激活！**==推荐==

```vue
<h1 :class="{red:true, italic:true, active:true, thin:true}">这是一个邪恶的H1</h1>
```

#### 使用style

1. 直接在元素上通过 `:style` 的形式，书写样式对象

```vue
<h1 :style="{color: 'red', 'font-size': '40px'}">这是一个善良的H1</h1>
```

2. 将样式对象，定义到 `data` 中，并直接引用到 `:style` 中

 + 在data上定义样式：

```vue
data: {
        h1StyleObj: { color: 'red', 'font-size': '40px', 'font-weight': '200' }
}
```

 + 在元素中，通过属性绑定的形式，将样式对象应用到元素中：

```vue
<h1 :style="h1StyleObj">这是一个善良的H1</h1>
```

3. 在 `:style` 中通过数组，引用多个 `data` 上的样式对象

 + 在data上定义样式：

```vue
data: {
        h1StyleObj: { color: 'red', 'font-size': '40px', 'font-weight': '200' },
        h1StyleObj2: { fontStyle: 'italic' }
}
```

 + 在元素中，通过属性绑定的形式，将样式对象应用到元素中：

```vue
<h1 :style="[h1StyleObj, h1StyleObj2]">这是一个善良的H1</h1>
```

- **总结**
  - 不会查文档就行
- **使用场景**
  - 当前页面的tag颜色高亮，可用class/style

### v-for&key属性

1. 迭代数组

```html
<ul>
  <li v-for="(item, i) in list">索引：{{i}} --- 姓名：{{item.name}} --- 年龄：{{item.age}}</li>
</ul>
```

2. 迭代对象中的属性

```html
<!-- 循环遍历对象身上的属性 -->
<div v-for="(val, key, i) in userInfo">{{val}} --- {{key}} --- {{i}}</div>
```

3. 迭代数字

```html
<p v-for="i in 10">这是第 {{i}} 个P标签</p>
```

> 2.2.0+ 的版本里，**当在组件中使用** v-for 时，key 现在是必须的。

**总案例**

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<body>
		<div id="app">
			<!-- 遍历普通数组 -->
			<div>
				&nbsp;<span v-for="(it,index) in arr" v-text="it+' : '+index +' ' "></span>&nbsp;
			</div>

			<!-- 遍历复杂数组 -->
			<div>
				<span v-for="(item,index) in arr_dict">{{item.username}} : {{item.age}}</span>
			</div>

			<!-- 遍历字典 -->
			<div id="">
				<span v-for="(key,value,index) in dict">key是===={{key}},value是====={{value}},索引是===={{index}}<br> </span>
			</div>
			
			<!-- 遍历表单数据并绑定 radio checked需要设置好value噢-->
			<input v-for="item in checkeds" type="checkbox" v-bind:value="item" v-model="checkeds"/>
		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				arr: ['A', 'B', 'C'],
				arr_dict: [
					{ username: 'tom',age: 17 },
					{ username: 'jack', age: 16 },
					{ username: 'lucy', age: 22 },
				],
				dict: { username:'tomcat',age:'1997',belong:'apache' },
				checkeds: [ 'man','woman','other' ]
			}

		})
	</script>
</html>
```



当 Vue.js 用 v-for 正在更新已渲染过的元素列表时，它默认用 “**就地复用**” 策略。如果数据项的顺序被改变，Vue将**不是移动 DOM 元素来匹配数据项的顺序**， 而是**简单复用此处每个元素**，并且确保它在特定索引下显示已被渲染过的每个元素。



为了给 Vue 一个提示，**以便它能跟踪每个节点的身份，从而重用和重新排序现有元素**，你需要为每项提供一个唯一 key 属性。

### v-if&v-show

- 总结
  - v-if 有更高的切换消耗而 v-show 有更高的初始渲染消耗
  - 如果需要频繁切换 v-show 较好
  - 如果在运行时条件不大可能改变 v-if 较好。
- 使用场景
  - 用户登录和未登录显示不同的tag，可用if/ show 判断



## 总结

```html
<!-- 1. MVC 和 MVVM 的区别 -->

<!-- 2. 学习了Vue中最基本代码的结构 -->

<!-- 3. 插值表达式   v-cloak   v-text   v-html   v-bind（缩写是:）   v-on（缩写是@）   v-model   v-for   v-if     v-show -->

<!-- 4. 事件修饰符  ：  .stop   .prevent   .capture   .self     .once -->

<!-- 5. el  指定要控制的区域    data 是个对象，指定了控制的区域内要用到的数据    methods 虽然带个s后缀，但是是个对象，这里可以自定义了方法 -->

<!-- 6. 在 VM 实例中，如果要访问 data 上的数据，或者要访问 methods 中的方法， 必须带 this -->

<!-- 7. 在 v-for 要会使用 key 属性 （只接受 string / number） -->

<!-- 8. v-model 只能应用于表单元素 -->

<!-- 9. 在vue中绑定样式两种方式  v-bind:class   v-bind:style -->
```



# Vue.js - Day2

## Vue调试工具

[Vue.js devtools - 翻墙安装方式 - 推荐](https://chrome.google.com/webstore/detail/vuejs-devtools/nhdogjmejiglipccpnnnanhbledajbpd?hl=zh-CN)

## 品牌管理案例一

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<link rel="stylesheet" type="text/css" href="../css/bootstrap-3.3.7.css" />
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<style type="text/css">
		a{
			text-decoration: none;
			color: #000000;
			cursor: copy;
		}
	</style>
	<body>
		<div id="app">
			<div class="panel panel-primary">
				<div class="panel-heading">
					<h3 class="panel-title">添加品牌</h3>
				</div>
				<div class="panel-body form-inline">
					<label>
						id:
						<input type="text" v-model="id"/>
					</label>
					<label>
						Name:
						<input type="text" v-model="name" />
					</label>
					
					<input type="button" value="添加" class="btn btn-primary" @click="add()"/>
					<label>
						搜索关键字:&nbsp;&nbsp;&nbsp;<input type="text"/>&nbsp;&nbsp;
						<button type="button" class="btn btn-primary" v-model="keyword">搜索</button>
					</label>
				</div>
			</div>
			
			<table class="table table-bordered table-hover table-striped">
				<thead>
					<th>ID</th>
					<th>Name</th>
					<th>CreateTime</th>
					<th>Operation</th>
				</thead>
				<tbody>
					<tr v-for="item in list" :key="item.id">
						<td>{{item.id}}</td>
						<td>{{item.name}}</td>
						<td>{{item.ctime}}</td>
						<td><a @click="deletes(item.id)">删除</a></td>
					</tr>
				</tbody>
			</table>

		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				list: [
					{id:1,name:'宝马',ctime:new Date()},
					{id:2,name:'奔驰',ctime:new Date()},
					{id:3,name:'法拉第',ctime:new Date()},
				],
				id: '',
				name: '',
				keyword: ''
			},
			methods: {
				add: function(){
					this.list.push({id:this.id,name:this.name,ctime:new Date()});
					this.id="";
					this.name="";
				},
				deletes: function(id){
					this.list.some((item,i)=>{
						if(item.id === id){
							// 从索引为i的位置 删除一个
							this.list.splice(i,1);
							return true;
						}
					})
				}
			}
		});
	</script>

</html>
```

## 品牌管理案例二

带搜索 时间格式化过滤器

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<link rel="stylesheet" type="text/css" href="../css/bootstrap-3.3.7.css" />
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<style type="text/css">
		a{
			text-decoration: none;
			color: #000000;
			cursor: copy;
		}
	</style>
	<body>
		<div id="app">
			<div class="panel panel-primary">
				<div class="panel-heading">
					<h3 class="panel-title">添加品牌</h3>
				</div>
				<div class="panel-body form-inline">
					<label>
						id:
						<input type="text" v-model="id"/>
					</label>
					
					<label>
						Name: <input type="text" v-model="name" />
					</label>
					
					<input type="button" value="添加" class="btn btn-primary" @click="add()"/>
					
					<label>
						搜索关键字:&nbsp;<input type="text" v-model="keyword"/>&nbsp;&nbsp;
					</label>
					
				</div>
			</div>
			
			<table class="table table-bordered table-hover table-striped">
				<thead>
					<th>ID</th>
					<th>Name</th>
					<th>CreateTime</th>
					<th>Operation</th>
				</thead>
				<tbody>
					<!-- for循环绑定一个方法,方法返回的是一个符合条件的数组 -->
					<tr v-for="item in oklist(keyword)" :key="item.id">
						<td>{{item.id}}</td>
						<td>{{item.name}}</td>
						<td>{{item.ctime}}</td>
						<td><a @click="deletes(item.id)">删除</a></td>
					</tr>
				</tbody>
			</table>

		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				id: '',
				name: '',
				keyword: '',
				list: [
					{id:1,name:'宝马',ctime:new Date()},
					{id:2,name:'奔驰',ctime:new Date()},
					{id:3,name:'法拉第',ctime:new Date()}
				]
			},
			methods: {
				add: function(){
					let obj = {id:this.id,name:this.name,ctime:new Date()}
					this.list.push(obj);
					this.id="";
					this.name="";
				},
				deletes: function(id){
					this.list.some((item,i)=>{
						if(item.id === id){
							// 从索引为i的位置 删除一个
							this.list.splice(i,1);
							return true;
						}
					})
				},
				oklist: function(keyword){
					return this.list.filter(item=>{
						if(item.name.indexOf(keyword)!=-1){
							return true;
						}
					})
				}
			}
		});
	</script>

</html>
```





## 过滤器

概念：Vue.js 允许你自定义过滤器，**可被用作一些常见的文本格式化**。过滤器可以用在两个地方：**mustache 插值和 v-bind 表达式**。过滤器应该被添加在 JavaScript 表达式的尾部，由“管道”符指示；

### 私有过滤器

```html
<body>
    <div id="app">
        {{time | dateFormat}}

    </div>
</body>
<script>
    // 创建vue实例
    var vm = new Vue({
        el: '#app', // 当前的vue实例要控制页面上的那个区域
        data: {
            time: new Date()
        },
        filters: {
            dateFormat: function(time, pattern = '') {
                let dt = new Date(time);
                let year = dt.getFullYear();
                let month = dt.getMonth() + 1;
                let day = dt.getDate();
                if (pattern.toLowerCase() === 'yyyy-mm-dd') {
                    return `${year}-${month}-${day}`;
                } else {
                    return `${year}-${month}-${day} ` + 'lalala';
                }
            }
        }
    });
</script>
```

> 使用ES6中的字符串新方法 String.prototype.padStart(maxLength, fillString='') 或 String.prototype.padEnd(maxLength, fillString='')来填充字符串；

### 全局过滤器

```JavaScript

// 定义一个全局过滤器
Vue.filter('dataFormat', function (input, pattern = '') {

  var dt = new Date(input);

  // 获取年月日

  var y = dt.getFullYear();

  var m = (dt.getMonth() + 1).toString().padStart(2, '0');

  var d = dt.getDate().toString().padStart(2, '0');



  // 如果 传递进来的字符串类型，转为小写之后，等于 yyyy-mm-dd，那么就返回 年-月-日

  // 否则，就返回  年-月-日 时：分：秒

  if (pattern.toLowerCase() === 'yyyy-mm-dd') {

    return `${y}-${m}-${d}`;

  } else {

    // 获取时分秒

    var hh = dt.getHours().toString().padStart(2, '0');

    var mm = dt.getMinutes().toString().padStart(2, '0');

    var ss = dt.getSeconds().toString().padStart(2, '0');

    return `${y}-${m}-${d} ${hh}:${mm}:${ss}`;

  }

});

```

> 注意：当有局部和全局两个名称相同的过滤器时候，会以就近原则进行调用，即：局部过滤器优先于全局过滤器被调用！



## 键盘

**键盘修饰符&自定义键盘修饰符**

### [2.x中自定义键盘修饰符](https://cn.vuejs.org/v2/guide/events.html#键值修饰符)

1. 通过`Vue.config.keyCodes.名称 = 按键值`来自定义案件修饰符的别名：

```js
Vue.config.keyCodes.f2 = 113; // 键盘码要对上
```

2. 使用自定义的按键修饰符：

```html
<input type="text" v-model="name" @keyup.f2="add">
```

## [自定义指令](https://cn.vuejs.org/v2/guide/custom-directive.html)

1. 自定义全局和局部的 自定义指令：

```html
<script>
    /**
	 * 参数一是指令名称，参数二是指令对象。
     */
    Vue.directive('focus',{
        bind: function(element){ // 指令绑定到元素上的时候会立即执行这个绑定函数，只执行一次。
            // 在每个函数中第一个参数永远是el 表示被绑定了指令的那个元素，这个el参数 是一个原生的JS对象
            // 在元素 刚绑定了指令的时候，还没有插入到dom中去，这时候调用focus方法没有作用
            // 因为，一个元素只有插入dom后才能聚焦元素。【注意：是聚焦 得元素插入到dom后才可ju'j】
            element.focus();
            console.log('bind执行了')
        },
        inserted: function(element){ // 元素插入到dom中的时候，会执行inserted函数。
        },
        updated: function(){ // window更新的时候会执行update，可能会触发多次。

        }
    })
    // 创建vue实例
    var vm = new Vue({
        el: '#app',
    });
</script>
```

2. 自定义指令的使用方式：

```html
<input type="text" v-model="searchName" v-focus v-color="'red'" v-font-weight="900">
```

	3. 获取自定义指令的数据 看官方文档吧

```html
<script>
    /**
	 * 参数一是指令名称，参数二是指令对象。
     */
    Vue.directive('focus',{
        bind: function(element,binding){ // 指令绑定到元素上的时候会立即执行这个绑定函数，只执行一次。
            // 在每个函数中第一个参数永远是el 表示被绑定了指令的那个元素，这个el参数 是一个原生的JS对象
            // 在元素 刚绑定了指令的时候，还没有插入到dom中去，这时候调用focus方法没有作用
            // 因为，一个元素只有插入dom后才能聚焦元素。【注意：是聚焦 得元素插入到dom后才可ju'j】
            element.focus();
            console.log('bind执行了')
        },
        inserted: function(element){ // 元素插入到dom中的时候，会执行inserted函数。
        },
        updated: function(){ // window更新的时候会执行update，可能会触发多次。

        }
    })
    // 创建vue实例
    var vm = new Vue({
        el: '#app',
    });
</script>
```



## 相关文章

1. [vue.js 1.x 文档](https://v1-cn.vuejs.org/)
2. [vue.js 2.x 文档](https://cn.vuejs.org/)
3. [String.prototype.padStart(maxLength, fillString)](http://www.css88.com/archives/7715)
4. [js 里面的键盘事件对应的键码](http://www.cnblogs.com/wuhua1/p/6686237.html)
5. [Vue.js双向绑定的实现原理](http://www.cnblogs.com/kidney/p/6052935.html)

## vue生命周期

### 本人笔记

beforeCreate：vue得data和methods都还没被初始化无法使用。

created：数据和方法被初始化了，可用使用了。【发起请求，获得数据】

beforeMount：模板编译好了，但是模板中得数据还未插入。数据还是原来html中得数据。

mounted：模板中的数据被插入到了dom当中

<img src="..\pics\vue\heima\lifecycle.png" style="margin:left">

```html
<body>
  <div id="app">
    <input type="button" value="修改msg" @click="msg='No'">
    <h3 id="h3">{{ msg }}</h3>
  </div>

  <script>
    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {
        msg: 'ok'
      },
      methods: {
        show() {
          console.log('执行了show方法')
        }
      },
      beforeCreate() { // 这是我们遇到的第一个生命周期函数，表示实例完全被创建出来之前，会执行它
        // console.log(this.msg)
        // this.show()
        // 注意： 在 beforeCreate 生命周期函数执行的时候，data 和 methods 中的 数据都还没有没初始化
      },
      created() { // 这是遇到的第二个生命周期函数
        // console.log(this.msg)
        // this.show()
        //  在 created 中，data 和 methods 都已经被初始化好了！
        // 如果要调用 methods 中的方法，或者操作 data 中的数据，最早，只能在 created 中操作
      },
      beforeMount() { // 这是遇到的第3个生命周期函数，表示 模板已经在内存中编辑完成了，但是尚未把 模板渲染到 页面中
        // console.log(document.getElementById('h3').innerText)
        // 在 beforeMount 执行的时候，页面中的元素，还没有被真正替换过来，只是之前写的一些模板字符串
      },
      mounted() { // 这是遇到的第4个生命周期函数，表示，内存中的模板，已经真实的挂载到了页面中，用户已经可以看到渲染好的页面了
        // console.log(document.getElementById('h3').innerText)
        // 注意： mounted 是 实例创建期间的最后一个生命周期函数，当执行完 mounted 就表示，实例已经被完全创建好了，此时，如果没有其它操作的话，这个实例，就静静的 躺在我们的内存中，一动不动
      },


      // 接下来的是运行中的两个事件
      beforeUpdate() { // 这时候，表示 我们的界面还没有被更新【数据被更新了吗？  数据肯定被更新了】
        /* console.log('界面上元素的内容：' + document.getElementById('h3').innerText)
        console.log('data 中的 msg 数据是：' + this.msg) */
        // 得出结论： 当执行 beforeUpdate 的时候，页面中的显示的数据，还是旧的，此时 data 数据是最新的，页面尚未和 最新的数据保持同步
      },
      updated() {
        console.log('界面上元素的内容：' + document.getElementById('h3').innerText)
        console.log('data 中的 msg 数据是：' + this.msg)
        // updated 事件执行的时候，页面和 data 数据已经保持同步了，都是最新的
      }
    });
  </script>
</body>
```

### [黑马笔记](https://cn.vuejs.org/v2/guide/instance.html#实例生命周期)

+ 什么是生命周期：从Vue实例创建、运行、到销毁期间，总是伴随着各种各样的事件，这些事件，统称为生命周期！
+ [生命周期钩子](https://cn.vuejs.org/v2/api/#选项-生命周期钩子)：就是生命周期事件的别名而已；
+ 生命周期钩子 = 生命周期函数 = 生命周期事件
+ 主要的生命周期函数分类：
 - 创建期间的生命周期函数：

  	+ beforeCreate：实例刚在内存中被创建出来，此时，还没有初始化好 data 和 methods 属性
  	+ created：实例已经在内存中创建OK，此时 data 和 methods 已经创建OK，此时还没有开始 编译模板
  	+ beforeMount：此时已经完成了模板的编译，但是还没有挂载到页面中
  	+ mounted：此时，已经将编译好的模板，挂载到了页面指定的容器中显示
 - 运行期间的生命周期函数：

 	+ beforeUpdate：状态更新之前执行此函数， 此时 data 中的状态值是最新的，但是界面上显示的 数据还是旧的，因为此时还没有开始重新渲染DOM节点
 	+ updated：实例更新完毕之后调用此函数，此时 data 中的状态值 和 界面上显示的数据，都已经完成了更新，界面已经被重新渲染好了！
 - 销毁期间的生命周期函数：

 	+ beforeDestroy：实例销毁之前调用。在这一步，实例仍然完全可用。
 	+ destroyed：Vue 实例销毁后调用。调用后，Vue 实例指示的所有东西都会解绑定，所有的事件监听器会被移除，所有的子实例也会被销毁。

**总结**

Vue的生命周期主要的作用之一是处理数据。

## vue的动画

### 自带的tag

需要使用vue的动画的话，请用transition标签包裹需要动画的内容。

```html
// 使用transition tag
<transition></transition>
```

具体例子

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<style type="text/css">
		/* v- 是一个前缀用来区分不同组直接的动画。transition标签默认的前缀为v- */
		/* 进入之前的状态，还没进入 */
		.v-enter,
		/* 动画离开终止状态，此时元素 动画已经结束 */
		.v-leave-to {
			opacity: 0;
			transform: translateX(80px);
		}

		/* 入场动画的时间段 */
		.v-enter-active,
		/* 离场动画的时间段 */
		.v-leave-active {
			transition: all 0.5s ease;
		}

		/* 通过name 设置前缀 如：name="m" 以m-为前缀 */
		.m-enter,
		.m-leave-to {
			opacity: 0;
			transform: translateY(80px);
		}

		.m-enter-active,
		.m-leave-active {
			transition: all 0.5s ease;
		}
	</style>
	<body>
		<div id="app">
			<button type="button" @click="flag=!flag">切换</button>
			<transition>
				<h3 v-if="flag">这是一个H3</h3>
			</transition>

			<button type="button" @click="flag2=!flag2">切换</button>
			
			<transition name="m">
				<h3 v-if="flag2">这是一个113</h3>
			</transition>
		</div>
	</body>
	<script>
		Vue.config.keyCodes.f2 = 113;
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				flag: false,
				flag2: false
			},
			methods: {

			}
		});
	</script>

</html>
```

### animate与vue

使用注意。

class = “animated xx”

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
		<link rel="stylesheet" type="text/css" href="../css/animate.css"/>
	</head>
	<body>
		<div id="app">
			<button type="button" @click="flag=!flag">切换</button>
			<transition enter-active-class="animated bounceIn" leave-active-class="animated bounceOut" 
			:duration="{enter:1000,leave:2000}">
				<h3 v-if="flag" class="animated">这是一个H3</h3>
			</transition>

			<button type="button" @click="flag2=!flag2">切换</button>
			
			<transition name="m">
				<h3 v-if="flag2">这是一个113</h3>
			</transition>
		</div>
	</body>
	<script>
		Vue.config.keyCodes.f2 = 113;
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				flag: false,
				flag2: false
			},
			methods: {

			}
		});
	</script>

</html>
```

### 钩子函数

1. 定义 transition 组件以及三个钩子函数：
```html
<div id="app">
    <input type="button" value="切换动画" @click="isshow = !isshow">
    <transition
    @before-enter="beforeEnter"
    @enter="enter"
    @after-enter="afterEnter">
      <div v-if="isshow" class="show">OK</div>
    </transition>
</div>
```
2. 定义三个 methods 钩子方法：
```js
methods: {
        beforeEnter(el) { // 动画进入之前的回调
          el.style.transform = 'translateX(500px)';
        },
        enter(el, done) { // 动画进入完成时候的回调
          el.offsetWidth;
          el.style.transform = 'translateX(0px)';
          done();
        },
        afterEnter(el) { // 动画进入完成之后的回调
          this.isshow = !this.isshow;
        }
}
```
3. 定义动画过渡时长和样式：
```css
.show{
      transition: all 0.4s ease;
}css
```

### [v-for 的列表过渡](https://cn.vuejs.org/v2/guide/transitions.html#列表的进入和离开过渡)

**v-for生成的列表都需要绑定动画事件可以用transition-group**

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
		<link rel="stylesheet" type="text/css" href="../css/animate.css" />
	</head>
	<body>
		<div id="app">
			<button type="button" @click="flag=!flag">点击</button>
			有很多tag需要动画效果
			<transition-group enter-active-class="animated bounceIn" leave-active-class="animated bounceOut">
				<li v-for="item in list" :key="item.id" v-show="flag">{{item.name}}</li>
			</transition-group>
		</div>
	</body>
	<script>
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				list: [
					{ id: 1, name: 'baoma1' },
					{ id: 2, name: 'baoma2' },
					{ id: 3, name: 'baoma3' },
					{ id: 4, name: 'baoma4' },
				],
				flag: true
			},
			methods: {
				move: function(id, show) {
					this.list[id].show = false;
				}
			}
		});
	</script>

</html>
```


### 列表的排序过渡
`<transition-group>` 组件还有一个特殊之处。不仅可以进入和离开动画，**还可以改变定位**。要使用这个新功能只需了解新增的 `v-move` 特性，**它会在元素的改变定位的过程中应用**。
+ `v-move` 和 `v-leave-active` 结合使用，能够让列表的过渡更加平缓柔和



## 表格动画

有个小坑。table内可以放什么元素是有严格规定的，这会导致vue的一些组件写在table内部时无效。需要用is代替

添加数据 删除数据带动画效果

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<link rel="stylesheet" type="text/css" href="../css/bootstrap-3.3.7.css" />
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
		<style>
			.v-enter,
	    .v-leave-to {
	      opacity: 0;
	      transform: translateY(80px);
	    }
	
	    .v-enter-active,
	    .v-leave-active {
	      transition: all 0.6s ease;
	    }
	
	    /* 下面的 .v-move 和 .v-leave-active 配合使用，能够实现列表后续的元素，渐渐地漂上来的效果 */
	    .v-move {
	      transition: all 0.6s ease;
	    }
	    .v-leave-active{
	      position: absolute;
	    }
	  </style>
	</head>
	<body>
		<div id="app">
			<div class="panel panel-primary">
				<div class="panel-heading">
					<h3 class="panel-title">添加品牌</h3>
				</div>
				<div class="panel-body form-inline">
					<label>
						id:
						<input type="text" v-model="id" />
					</label>

					<label>
						Name: <input type="text" v-model="name" @keyup.enter="add()" />
					</label>

					<input type="button" value="添加" class="btn btn-primary" @click="add()" />

					<label>
						搜索关键字:&nbsp;<input type="text" v-model="keyword" />&nbsp;&nbsp;
					</label>

				</div>
			</div>

			<table class="table table-bordered table-hover table-striped">
				<thead>
					<th>ID</th>
					<th>Name</th>
					<th>CreateTime</th>
					<th>Operation</th>
				</thead>
				<!-- 这里有一个坑！table的解析不能直接用transition-group这个标签，而是要用is -->
				<tbody is="transition-group" appear>
					<!-- for循环绑定一个方法,方法返回的是一个符合条件的数组 -->
					<tr v-for="item in oklist(keyword)" :key="item.id">
						<td>{{item.id}}</td>
						<td>{{item.name}}</td>
						<td>{{item.ctime | dataFormat()}}</td>
						<td><a @click="deletes(item.id)">删除</a></td>
					</tr>
				</tbody>
			</table>
		</div>
	</body>
	<script>
		// 此处不用箭头函数 因为箭头函数的this会指向父级作用域。【此处的父级是windows】
		Vue.filter('dataFormat', function(ctime) {
			let time = new Date(ctime);
			let year = time.getFullYear();
			let month = time.getMonth() + 1;
			let day = time.getDate();
			// 模板字符串
			return `${year}-${month}-${day}`
		});
		// 创建vue实例
		var vm = new Vue({
			el: '#app',
			data: {
				id: '',
				name: '',
				keyword: '',
				list: [
					{ id: 1, name: '宝马', ctime: new Date() },
					{ id: 2, name: '奔驰', ctime: new Date() },
					{ id: 3, name: '法拉第', ctime: new Date() }
				]
			},
			methods: {
				add: function() {
					if (this.id.trim() == '' || this.name.trim() == '') {
						alert('请输入对应的数据！');
					} else {
						let obj = { id: this.id, name: this.name, ctime: new Date()	}
						this.list.push(obj);
						this.id = ""; 
						this.name = "";
					}
				},
				deletes: function(id) {
					this.list.some((item, i) => {
						if (item.id === id) {
							// 从索引为i的位置 删除一个
							this.list.splice(i, 1);
							return true;
						}
					})
				},
				oklist: function(keyword) {
					return this.list.filter(item => {
						if (item.name.indexOf(keyword) != -1) {
							return true;
						}
					})
				}
			}
		});
	</script>
</html>
```

**注意事项**

有些 HTML 元素，诸如 ``、``、`` 和 ``，对于哪些元素可以出现在其内部是有严格限制的。而有些元素，诸如 ``、 和 ``，只能出现在其它某些特定的元素内部。

这会导致我们使用这些有约束条件的元素时遇到一些问题。例如：

```
<table>
  <blog-post-row></blog-post-row>
</table>
```

这个自定义组件 `` 会被作为无效的内容提升到外部，并导致最终渲染结果出错。幸好这个特殊的 `is` attribute 给了我们一个变通的办法：

```
<table>
  <tr is="blog-post-row"></tr>
</table>
```

## tranistion总结★

tranistion是vue提供的一个动画组件。

我们只需要定义好css，tranisition会自动绑定好css，实现动画效果。

```css
/* v- 是一个前缀用来区分不同组直接的动画。transition标签默认的前缀为v- */
/* 进入之前的状态，还没进入 */
.v-enter,
/* 动画离开终止状态，此时元素 动画已经结束 */
.v-leave-to {
    opacity: 0;
    transform: translateX(80px);
}

/* 入场动画的时间段 */
.v-enter-active,
/* 离场动画的时间段 */
.v-leave-active {
    transition: all 0.5s ease;
}
```

----

除了用它自己绑定css外，我们也可以为其设置属性，手动绑定我们自己的css。如:

```html
<transition enter-active-class="animated bounceIn" leave-active-class="animated bounceOut" 
            :duration="{enter:1000,leave:2000}">
    <h3 v-if="flag" class="animated">这是一个H3</h3>
</transition>
```

----

trainstiony有一个坑。对于table这类元素，对于哪些元素可以出现在其内部是有严格限制的。我们在table内部使用`transition`，`transition-group`这些，是无法生效的，需要使用`is`解决

- appear 是 啥来着，我记得和那个v-move有关
- tag是内部默认包裹的一个标签【span】，可以自己手动设置，如：tag=“div”，选择div包裹transition内的东西

```html
<tbody is="transition-group" appear tag>
    <!-- for循环绑定一个方法,方法返回的是一个符合条件的数组 -->
    <tr v-for="item in oklist(keyword)" :key="item.id">
        <td>{{item.id}}</td>
        <td>{{item.name}}</td>
        <td>{{item.ctime | dataFormat()}}</td>
        <td><a @click="deletes(item.id)">删除</a></td>
    </tr>
</tbody>
```



## 相关文章

1. [vue.js 1.x 文档](https://v1-cn.vuejs.org/)
2. [vue.js 2.x 文档](https://cn.vuejs.org/)
3. [String.prototype.padStart(maxLength, fillString)](http://www.css88.com/archives/7715)
4. [js 里面的键盘事件对应的键码](http://www.cnblogs.com/wuhua1/p/6686237.html)
5. [pagekit/vue-resource](https://github.com/pagekit/vue-resource)
6. [navicat如何导入sql文件和导出sql文件](https://jingyan.baidu.com/article/a65957f4976aad24e67f9b9b.html)
7. [贝塞尔在线生成器](http://cubic-bezier.com/#.4,-0.3,1,.33)



# Vue.js - Day3

## 定义Vue组件
什么是组件： 组件的出现，就是为了拆分Vue实例的代码量的，能够让我们以不同的组件，来划分不同的功能模块，将来我们需要什么样的功能，就可以去调用对应的组件即可；
组件化和模块化的不同：

 + 模块化： 是从代码逻辑的角度进行划分的；方便代码分层开发，保证每个功能模块的职能单一；
 + **组件化**： 是从UI界面的角度进行划分的；前端的组件化，方便UI组件的重用；
## 全局组件定义的

## 代码

**一共三种方式**

1. 使用 Vue.extend 配合 Vue.component 方法：
2. 直接使用 Vue.component 方法：
3. 将模板字符串，定义到script标签种：

   ​	同时，需要使用 Vue.component 来定义组件：

> 注意： 组件中的DOM结构，有且只能有唯一的根元素（Root Element）来进行包裹！

> **具体代码**

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<body>
		<div id="app">
			<!-- 标签不能用驼峰，如果是驼峰需要用-分离 -->
			<my-com></my-com>
			<my-com2></my-com2>
			<my-com3></my-com3>
		</div>
		
		<template id="ids">
			<div>
				<h3>组件注册方式三，有代码提示</h3>
			</div>
		</template>
	</body>
	<script>
		// 组件创建方式一
		let com = Vue.extend({
			// 组件的模板里只能有唯一的根元素。
			template: '<h3>这是vue全局组件注册方式1</h3>'
		});
		
		// 组件名字 === 组件对象
		Vue.component('myCom',com);
		
		Vue.component("my-com2",{
			template: '<h3>这是vue全局组件注册方式2</h3>'
		})
		
		
		Vue.component('my-com3',{
			// 在被控制的app外面，使用template 元素 定义组件的模板结构
			template:'#ids'
		})


		new Vue({
			el:'#app'
		})
		
	</script>

</html>
```

#### vue的属性回顾

```javascript
<script>
    let v = new Vue({
        el: '#app', // 挂在的标签
        data: {}, // 数据
        methods: {}, // 方法
        filters: {}, // 过滤器。全局的不带s
        directives: {}, // 自定义指令
        components: {}, // s 可注册多个组件
        beforeCreate: {},
        created: {},
        beforeMount: {},
        mounted: {},
        beforeUpdate: {},
        updated: {},
        beforeDestroy: {},
        destroyed: {}
    });
</script>
```

## 私有组件定义

- 可以直接书写html代码
- 可以用template

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8" />
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<body>
		<div id="app">
			<login></login>
			<logout></logout>
		</div>
		
		<template id="logout">
			<h3>这是logout</h3>
		</template>

	</body>
	<script>
		let v = new Vue({
			el: '#app', // 挂在的标签
			data: {}, // 数据
			methods: {}, // 方法
			components: {
				login: {
					template:'<h1>私有组件login</h1>'
				},
				logout: {
					template: '#logout'
				}
			}
		});
	</script>
</html>
```

## 组件中数据和响应事件

- **组件中的data为function，是为了保证同一种类型的组件，在复用组件的时候不同的tag，数据要不一样【避免数据共享，造成问题】。**
- 组件中的数据必须为一个function！返回一个对象。【避免共享数据！】
- 在组件中，`data`需要被定义为一个方法，例如：

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<body>
		<div id="app">
			<my-com></my-com>
		</div>

		<script type="text/javascript">
			let c1 = Vue.extend({
				template: '<div @click="ck()">Hello{{msg}} </div>',

				// 组件中的data要为方法.组件可以引用自己的数据
				data: function() {
					return {
						msg: '组件自己的数据!',
						list: [1, 2, 3, 4, 5, 6, 5, 3, 2]
					}
				},
				// 定义组件自己的方法 注意事件绑定要绑定在组件的html标签里，而非自定义的组件上
				methods: {
					ck: function() {
						confirm(this.msg);
						alert(this)
					}
				}
			})
			Vue.component('my-com', c1);

			new Vue({
				el: '#app'
			})
		</script>
	</body>
</html>
```
## `component标签使用组件`

**将对于名称的组件渲染到xx上，用is属性绑定**

```html
<body>
    <div id="app">
        <component is="com2"></component>
    </div>

    <script type="text/javascript">
        Vue.component('com1',{
            template:'<div>com1</div>'
        });
        Vue.component('com2',{
            template:'<div>com2</div>'
        });
        Vue.component('com3',{
            template:'<div>com3</div>'
        });
        new Vue({
            el: '#app'
        });

    </script>
</body>
```
**component实现组件切换**

用`:is`绑定动态的值

```html
</head>
<body>
    <div id="app">
        <button @click="cg()">点击切换</button>
        <!-- 加 ： 动态绑定 -->
        <component :is="comName+number"></component>
    </div>

    <script type="text/javascript">
        Vue.component('com1',{
            template:'<div>com1</div>'
        });
        Vue.component('com2',{
            template:'<div>com2</div>'
        });
        Vue.component('com3',{
            template:'<div>com3</div>'
        });

        new Vue({
            el: '#app',
            data: {
                comName: 'com',
                number: 1
            },
            methods: {
                cg: function(){
                    if(this.number<3){
                        this.number = this.number+1;
                    }else{
                        this.number = 1;
                    }
                }
            }
        });

    </script>
</body>
```
**带动画的组件切换**

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
	</head>
	<style type="text/css">
		.v-enter,
		.v-leave-to{
			opacity: 0;
			/* transform：转换 属性向元素应用 2D 或 3D 转换。该属性允许我们对元素进行旋转、缩放、移动或倾斜。 */
			transform: translateX(1.25rem);
		}
		
		.v-enter-active,
		.v-leave-active{
			/* transition：过渡 设置过渡 0.5秒的过渡时间  ease：缓解*/
			transition: all 0.5s ease;
		}
	</style>
	<body>
		<div id="app">
			<button @click="cg()">点击切换</button>
			<!-- 组件的过渡动画，用着标签包裹起来就行。 mode是设置模式
				 先出去再进来
			 -->
			<transition mode="out-in">
				<component :is="comName+number"></component>
			</transition>
			
		</div>

		<script type="text/javascript">
			Vue.component('com1',{
				template:'<div>com1123213123123</div>'
			});
			Vue.component('com2',{
				template:'<div>com21123123123123</div>'
			});
			Vue.component('com3',{
				template:'<div>com3asdfasfasfsdfa</div>'
			});
			
			new Vue({
				el: '#app',
				data: {
					comName: 'com',
					number: 1
				},
				methods: {
					cg: function(){
						if(this.number<3){
							this.number = this.number+1;
						}else{
							this.number = 1;
						}
					}
				}
			});
			
		</script>
	</body>
</html>
```

# Vue.js - Day4

## 父组件向子组件传值

子组件无法直接访问到父组件中的data

子组件想要访问父组件的data，需要用属性绑定的形式，把需要传递给子组件的数据，以属性绑定的形式传递给子组件。

组件实例定义方式，注意：一定要使用`props`属性来定义父组件传递过来的数据

- 子组件要想使用父组件的数据，需要使用属性绑定。

- 子组件只能使用父组件的数据，不能修改。
- 子组件 props 中的数据，都是只读的，无法重新赋值。【即便可以成功，控制台也会报错warning】
- 子组件可以使用并修改自己的data

```html
<div id="app">
    <!-- 父组件向子组件传值 需要属性使用绑定。把父组件的数据，绑定给子组件的某个属性 -->
    <!-- 父组件传给子组件的值，子组件只能使用，不能修改。子组件自己的data，子组件可以修改 -->
    <login v-bind:msg='fmsg'></login>
</div>

<script type="text/javascript">
    let login = {
        template:'<div>{{msg}}</div>',
        props: ['msg'],
        data(){
            return {}
        }
    }

    new Vue({
        el: '#app',
        data: {
            fmsg:'123'
        },
        components:{
            login
        }
    });
</script>
```
## 父组件向子组件传方法

- 子组件中的tag绑定自己的方法，然后在自己的方法中用 `this.$emit('事件的名称')`调用父方法的函数。
- 然后使用组件时用@事件的名称=“方法名”
- 总结：this.$emit('事件')，组件绑定事件
- `$emit`有多个参数，一个是事件名，其他的是参数。 ==$emit('xx',1123,445,'jack')

```html
<div id="app">
    <!-- 在子组件上的xx事件上绑定父组件的方法 -->
    <login @father-send="fatherShow"></login>
</div>

<script type="text/javascript">
    let login = {
        // 子组件绑定自己的method
        template: '<div @click="son()">please click me!</div>',
        methods: {
            // 子组件的method调用组件上的事件
            // 事件名   参数
            son:function(){	this.$emit('father-send','123123') }
        }

    }

    // 父组件
    new Vue({
        el: '#app',
        data: { fmsg: '123' },
        methods: {
            fatherShow: function(data) { console.log(data) }
        },
        components: { login	}
    });
</script>
```



## 子组件向父组件传值

子组件调用父组件的方法，在方法中传入参数，父组件的方法获得参数，就拿到了子组件的数据了！**机智！**

1. 原理：父组件将方法的引用，传递到子组件内部，子组件在内部调用父组件传递过来的方法，同时把要发送给父组件的数据，在调用方法的时候当作参数传递进去；
2. 父组件将方法的引用传递给子组件，其中，`getMsg`是父组件中`methods`中定义的方法名称，`func`是子组件调用传递过来方法时候的方法名称
```
<son @func="getMsg"></son>
```
3. 子组件内部通过`this.$emit('方法名', 要传递的数据)`方式，来调用父组件中的方法，同时把数据传递给父组件使用
```html
<body>
    <div id="app">
        <!-- 在子组件上的xx事件上绑定父组件的方法 -->
        <login @father-send="fatherShow"></login>
    </div>

    <script type="text/javascript">
        let login = {
            // 子组件绑定自己的method
            template: '<div @click="son()">please click me!</div>',
            data(){ return { message:'hello world'} },
            methods: {
                // 子组件的method调用组件上的事件
                // 事件名   参数
                son:function(){ this.$emit('father-send',this.$data); }
            }
        }

        // 父组件
        new Vue({
            el: '#app',
            data: { fmsg: '123' },
            methods: {
                fatherShow: function(data) {
                    console.log(data)
                    console.log(data.message)
                }
            },
            components: { login }
        });
    </script>
</body>
```

## 评论列表案例
目标：主要练习父子组件之间传值

##  `this.$refs` 获取元素和组件★★★★

**使用组件的时候绑定的ref才有效！**

**通过`<router-view>`标签渲染的无效噢！**

```html
<body>
    <div id="app">
        <button type="button" @click="get">12312</button>
        <!-- 给子组件绑定了ref -->
        <h3 ref="h3">13123</h3>
        <login v-bind:msg='fmsg' ref="son"></login>
    </div>

    <script type="text/javascript">
        let login = {
            template: '<div>{{msg}}</div>',
            props: ['msg'],
            data() {
                return {}
            }
        }

        new Vue({
            el: '#app',
            data: {
                fmsg: '123'
            },
            components: {
                login
            },
            methods: {
                // 我们可以在父组件中，用$refs访问子组件和子组件中的数据。也可以用$refs获得dom对象
                get: function() {
                    console.log(this.$refs.son.msg);
                    console.log(this.$refs.h3);
                    console.log(this.$refs.h3.innerText);
                }
            }
        });
    </script>
</body>
```

## 什么是路由

**Vue中用路由进行组件切换**

1. 对于普通的网站，所有的超链接都是URL地址，所有的URL地址都对应服务器上对应的资源；

2. 对于单页面应用程序来说，主要通过URL中的hash(#号)来实现不同页面之间的切换，同时，hash有一个特点：HTTP请求中不会包含hash相关的内容；所以，单页面程序中的页面跳转主要用hash实现；

3. 在单页面应用程序中，这种通过hash改变来切换页面的方式，称作前端路由（区别于后端路由）；

## vue-router
1. 导入 vue-router 组件类库：
```html
<!-- 1. 导入 vue-router 组件类库 -->
<script src="./lib/vue-router-2.7.0.js"></script>
```
2. 使用 router-link 组件来导航
```html
<!-- 2. 使用 router-link 组件来导航 -->
<router-link to="/login">登录</router-link>
<router-link to="/register">注册</router-link>
```
3. 使用 router-view 组件来显示匹配到的组件
```html
<!-- 3. 使用 router-view 组件来显示匹配到的组件 -->
<router-view></router-view>
```
4. 创建使用`Vue.extend`创建组件
```js
// 4.1 使用 Vue.extend 来创建登录组件
var login = Vue.extend({
    template: '<h1>登录组件</h1>'
});

// 4.2 使用 Vue.extend 来创建注册组件
var register = Vue.extend({
    template: '<h1>注册组件</h1>'
});
```
5. 创建一个路由 router 实例，通过 routers 属性来定义路由匹配规则
```js
// 5. 创建一个路由 router 实例，通过 routers 属性来定义路由匹配规则
var router = new VueRouter({
    routes: [
        { path: '/login', component: login },
        { path: '/register', component: register }
    ]
});
```
6. 使用 router 属性来使用路由规则
```js
// 6. 创建 Vue 实例，得到 ViewModel
var vm = new Vue({
    el: '#app',
    router: router // 使用 router 属性来使用路由规则
});
```

**完整的实例**

```html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title></title>
		<script src="../js/vue.js" type="text/javascript" charset="utf-8"></script>
		<script src="../js/vue-router-3.0.1.js" type="text/javascript" charset="utf-8"></script>
	</head>

	<body>
		<div id="app">
			<router-link to="/login">登录</router-link>
			<router-link to="/register">注册</router-link>
			<!-- 
			 vue提供的标签，用来当作占位符的
			 会把匹配到的组件，渲染到这个占位符的位置
			 -->
			<router-view></router-view>
		</div>

		<script type="text/javascript">
			
			// 创建两个组件 
			let login = {
				template: '<div>登录组件</div>',
			}
			let register = {
				template:'<div>注册组件</div>'
			}

			// 把组件和路由规则映射起来
			let rou = new VueRouter({
				routes:[
					{ path:'/login',component:login },
					{ path:'/register',component:register },
					{ path:'/',component:register }
				]
			});
			
			// 将路由规则对象注册到vm实例上，用来监听url地址的变化。展示相应的组件。
			let vm = new Vue({
				el: '#app',
				router: rou
			});
		</script>
	</body>
</html>
```

## 设置路由高亮

> **想要点击的那个路由的tag高亮用 `.router-link-active`即可，或者在Vuerouter中设置`linkActiveClass:class的名字**`

## 设置路由切换动效

把<router-view>用transition包裹就可以了。然后写css

## 在路由规则中定义参数
1. **在规则中定义参数：**[推荐，rest风格]
```json
{ path: '/register/:id', component: register },
{ path: '/register/:id/:name', component: register }
```
2. 通过 `this.$route.params`来==获取路由中的参数==：
```js
var register = Vue.extend({
      template: '<h1>注册组件 --- {{this.$route.params.id}}</h1>'
);
```

## 使用 `children` 属性实现路由嵌套

```
  <div id="app">
    <router-link to="/account">Account</router-link>

    <router-view></router-view>
  </div>

  <script>
    // 父路由中的组件
    const account = Vue.extend({
      template: `<div>
        这是account组件
        <router-link to="/account/login">login</router-link> | 
        <router-link to="/account/register">register</router-link>
        <router-view></router-view>
      </div>`
    });

    // 子路由中的 login 组件
    const login = Vue.extend({
      template: '<div>登录组件</div>'
    });

    // 子路由中的 register 组件
    const register = Vue.extend({
      template: '<div>注册组件</div>'
    });

    // 路由实例
    var router = new VueRouter({
      routes: [
        { path: '/', redirect: '/account/login' }, // 使用 redirect 实现路由重定向
        {
          path: '/account',
          component: account,
          children: [ // 通过 children 数组属性，来实现路由的嵌套
            { path: 'login', component: login }, // 注意，子路由的开头位置，不要加 / 路径符
            { path: 'register', component: register }
          ]
        }
      ]
    });

    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {},
      methods: {},
      components: {
        account
      },
      router: router
    });
  </script>
```

## 命名视图实现经典布局
1. 标签代码结构：
```
<div id="app">
    <router-view></router-view>
    <div class="content">
      <router-view name="a"></router-view>
      <router-view name="b"></router-view>
    </div>
  </div>
```
2. JS代码：
```
<script>
    var header = Vue.component('header', {
      template: '<div class="header">header</div>'
    });

    var sidebar = Vue.component('sidebar', {
      template: '<div class="sidebar">sidebar</div>'
    });

    var mainbox = Vue.component('mainbox', {
      template: '<div class="mainbox">mainbox</div>'
    });

    // 创建路由对象
    var router = new VueRouter({
      routes: [
        {
          path: '/', components: {
            default: header,
            a: sidebar,
            b: mainbox
          }
        }
      ]
    });

    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {},
      methods: {},
      router
    });
  </script>
```
3. CSS 样式：
```
  <style>
    .header {
      border: 1px solid red;
    }

    .content{
      display: flex;
    }
    .sidebar {
      flex: 2;
      border: 1px solid green;
      height: 500px;
    }
    .mainbox{
      flex: 8;
      border: 1px solid blue;
      height: 500px;
    }
  </style>
```

## `watch`属性的使用
考虑一个问题：想要实现 `名` 和 `姓` 两个文本框的内容改变，则全名的文本框中的值也跟着改变；（用以前的知识如何实现？？？）

1. 监听`data`中属性的改变：
```
<div id="app">
    <input type="text" v-model="firstName"> +
    <input type="text" v-model="lastName"> =
    <span>{{fullName}}</span>
  </div>

  <script>
    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {
        firstName: 'jack',
        lastName: 'chen',
        fullName: 'jack - chen'
      },
      methods: {},
      watch: {
        'firstName': function (newVal, oldVal) { // 第一个参数是新数据，第二个参数是旧数据
          this.fullName = newVal + ' - ' + this.lastName;
        },
        'lastName': function (newVal, oldVal) {
          this.fullName = this.firstName + ' - ' + newVal;
        }
      }
    });
  </script>
```
2. 监听路由对象的改变：
```
<div id="app">
    <router-link to="/login">登录</router-link>
    <router-link to="/register">注册</router-link>

    <router-view></router-view>
  </div>

  <script>
    var login = Vue.extend({
      template: '<h1>登录组件</h1>'
    });

    var register = Vue.extend({
      template: '<h1>注册组件</h1>'
    });

    var router = new VueRouter({
      routes: [
        { path: "/login", component: login },
        { path: "/register", component: register }
      ]
    });

    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {},
      methods: {},
      router: router,
      watch: {
        '$route': function (newVal, oldVal) {
          if (newVal.path === '/login') {
            console.log('这是登录组件');
          }
        }
      }
    });
  </script>
```

## `computed`计算属性的使用
1. 默认只有`getter`的计算属性：
```
<div id="app">
    <input type="text" v-model="firstName"> +
    <input type="text" v-model="lastName"> =
    <span>{{fullName}}</span>
  </div>

  <script>
    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {
        firstName: 'jack',
        lastName: 'chen'
      },
      methods: {},
      computed: { // 计算属性； 特点：当计算属性中所以来的任何一个 data 属性改变之后，都会重新触发 本计算属性 的重新计算，从而更新 fullName 的值
        fullName() {
          return this.firstName + ' - ' + this.lastName;
        }
      }
    });
  </script>
```
2. 定义有`getter`和`setter`的计算属性：
```
<div id="app">
    <input type="text" v-model="firstName">
    <input type="text" v-model="lastName">
    <!-- 点击按钮重新为 计算属性 fullName 赋值 -->
    <input type="button" value="修改fullName" @click="changeName">

    <span>{{fullName}}</span>
  </div>

  <script>
    // 创建 Vue 实例，得到 ViewModel
    var vm = new Vue({
      el: '#app',
      data: {
        firstName: 'jack',
        lastName: 'chen'
      },
      methods: {
        changeName() {
          this.fullName = 'TOM - chen2';
        }
      },
      computed: {
        fullName: {
          get: function () {
            return this.firstName + ' - ' + this.lastName;
          },
          set: function (newVal) {
            var parts = newVal.split(' - ');
            this.firstName = parts[0];
            this.lastName = parts[1];
          }
        }
      }
    });
  </script>
```

## `watch`、`computed`和`methods`之间的对比
1. `computed`属性的结果会被缓存，除非依赖的响应式属性变化才会重新计算。主要当作属性来使用；
2. `methods`方法表示一个具体的操作，主要书写业务逻辑；
3. `watch`一个对象，键是需要观察的表达式，值是对应回调函数。主要用来监听某些特定数据的变化，从而进行某些具体的业务逻辑操作；可以看作是`computed`和`methods`的结合体；【观察虚拟对象，如vue自定义的组件，路由】

## `nrm`的安装使用
作用：提供了一些最常用的NPM包镜像地址，能够让我们快速的切换安装包时候的服务器地址；
什么是镜像：原来包刚一开始是只存在于国外的NPM服务器，但是由于网络原因，经常访问不到，这时候，我们可以在国内，创建一个和官网完全一样的NPM服务器，只不过，数据都是从人家那里拿过来的，除此之外，使用方式完全一样；
1. 运行`npm i nrm -g`全局安装`nrm`包；
2. 使用`nrm ls`查看当前所有可用的镜像源地址以及当前所使用的镜像源地址；
3. 使用`nrm use npm`或`nrm use taobao`切换不同的镜像源地址；

## 相关文件
1. [URL中的hash（井号）](http://www.cnblogs.com/joyho/articles/4430148.html)





# Vue.js - Day5 - Webpack

## 在网页中会引用哪些常见的静态资源？
+ JS
 - .js  .jsx  .coffee  .ts（TypeScript  类 C# 语言）
+ CSS
 - .css  .less   .sass  .scss
+ Images
 - .jpg   .png   .gif   .bmp   .svg
+ 字体文件（Fonts）
 - .svg   .ttf   .eot   .woff   .woff2
+ 模板文件
 - .ejs   .jade  .vue【这是在webpack中定义组件的方式，推荐这么用】


## 网页中引入的静态资源多了以后有什么问题？？？
1. 网页加载速度慢， 因为 我们要发起很多的二次请求；
2. 要处理错综复杂的依赖关系


## 如何解决上述两个问题
1. 合并、压缩、精灵图、图片的Base64编码
2. 可以使用之前学过的requireJS、也可以使用webpack可以解决各个包之间的复杂依赖关系；

## 什么是webpack?
webpack 是前端的一个**项目构建工具**，它是基于 Node.js 开发出来的一个前端工具；


## 如何完美实现上述的2种解决方案
1. 使用Gulp， 是基于 task 任务的；
2. 使用Webpack， 是基于整个项目进行构建的；
+ **借助于webpack这个前端自动化构建工具，可以完美实现资源的合并、打包、压缩、混淆等诸多功能。**
+ 根据官网的图片介绍webpack打包的过程
+ [webpack官网](http://webpack.github.io/)

## webpack安装的两种方式
1. 运行`npm i webpack -g`全局安装webpack，这样就能在全局使用webpack的命令
2. 在项目根目录中运行`npm i webpack --save-dev`安装到项目依赖中

## 初步使用webpack打包构建列表隔行变色案例
1. 运行`npm init`初始化项目，使用npm管理项目中的依赖包
2. 创建项目基本的目录结构
3. 使用`cnpm i jquery --save`安装jquery类库
4. 创建`main.js`并书写各行变色的代码逻辑：
```
	// 导入jquery类库
    import $ from 'jquery'

    // 设置偶数行背景色，索引从0开始，0是偶数
    $('#list li:even').css('backgroundColor','lightblue');
    // 设置奇数行背景色
    $('#list li:odd').css('backgroundColor','pink');
```
5. 直接在页面上引用`main.js`会报错，因为浏览器不认识`import`这种高级的JS语法，需要使用webpack进行处理，webpack默认会把这种高级的语法转换为低级的浏览器能识别的语法；
6. 运行`webpack 入口文件路径 输出文件路径`对`main.js`进行处理：
```
webpack src/js/main.js dist/bundle.js
```

## 使用webpack的配置文件简化打包时候的命令
1. 在项目根目录中创建`webpack.config.js`
2. 由于运行webpack命令的时候，webpack需要指定入口文件和输出文件的路径，所以，我们需要在`webpack.config.js`中配置这两个路径：
```
    // 导入处理路径的模块
    var path = require('path');

    // 导出一个配置对象，将来webpack在启动的时候，会默认来查找webpack.config.js，并读取这个文件中导出的配置对象，来进行打包处理
    module.exports = {
        entry: path.resolve(__dirname, 'src/js/main.js'), // 项目入口文件
        output: { // 配置输出选项
            path: path.resolve(__dirname, 'dist'), // 配置输出的路径
            filename: 'bundle.js' // 配置输出的文件名
        }
    }
```

## 实现webpack的实时打包构建
1. 由于每次重新修改代码之后，都需要手动运行webpack打包的命令，比较麻烦，所以使用`webpack-dev-server`来实现代码实时打包编译，当修改代码之后，会自动进行打包构建。
2. 运行`cnpm i webpack-dev-server --save-dev`安装到开发依赖
3. 安装完成之后，在命令行直接运行`webpack-dev-server`来进行打包，发现报错，此时需要借助于`package.json`文件中的指令，来进行运行`webpack-dev-server`命令，在`scripts`节点下新增`"dev": "webpack-dev-server"`指令，发现可以进行实时打包，但是dist目录下并没有生成`bundle.js`文件，这是因为`webpack-dev-server`将打包好的文件放在了内存中
 + 把`bundle.js`放在内存中的好处是：由于需要实时打包编译，所以放在内存中速度会非常快
 + 这个时候访问webpack-dev-server启动的`http://localhost:8080/`网站，发现是一个文件夹的面板，需要点击到src目录下，才能打开我们的index首页，此时引用不到bundle.js文件，需要修改index.html中script的src属性为:`<script src="../bundle.js"></script>`
 + 为了能在访问`http://localhost:8080/`的时候直接访问到index首页，可以使用`--contentBase src`指令来修改dev指令，指定启动的根目录：
 ```
 "dev": "webpack-dev-server --contentBase src"
 ```
 同时修改index页面中script的src属性为`<script src="bundle.js"></script>`

## 使用`html-webpack-plugin`插件配置启动页面
由于使用`--contentBase`指令的过程比较繁琐，需要指定启动的目录，同时还需要修改index.html中script标签的src属性，所以推荐大家使用`html-webpack-plugin`插件配置启动页面.
1. 运行`cnpm i html-webpack-plugin --save-dev`安装到开发依赖
2. 修改`webpack.config.js`配置文件如下：
```
    // 导入处理路径的模块
    var path = require('path');
    // 导入自动生成HTMl文件的插件
    var htmlWebpackPlugin = require('html-webpack-plugin');

    module.exports = {
        entry: path.resolve(__dirname, 'src/js/main.js'), // 项目入口文件
        output: { // 配置输出选项
            path: path.resolve(__dirname, 'dist'), // 配置输出的路径
            filename: 'bundle.js' // 配置输出的文件名
        },
        plugins:[ // 添加plugins节点配置插件
            new htmlWebpackPlugin({
                template:path.resolve(__dirname, 'src/index.html'),//模板路径
                filename:'index.html'//自动生成的HTML文件的名称
            })
        ]
    }
```
3. 修改`package.json`中`script`节点中的dev指令如下：
```
"dev": "webpack-dev-server"
```
4. 将index.html中script标签注释掉，因为`html-webpack-plugin`插件会自动把bundle.js注入到index.html页面中！

## 实现自动打开浏览器、热更新和配置浏览器的默认端口号
**注意：热更新在JS中表现的不明显，可以从一会儿要讲到的CSS身上进行介绍说明！**
### 方式1：
+ 修改`package.json`的script节点如下，其中`--open`表示自动打开浏览器，`--port 4321`表示打开的端口号为4321，`--hot`表示启用浏览器热更新：
```
"dev": "webpack-dev-server --hot --port 4321 --open"
```

### 方式2：
1. 修改`webpack.config.js`文件，新增`devServer`节点如下：
```
devServer:{
        hot:true,
        open:true,
        port:4321
    }
```
2. 在头部引入`webpack`模块：
```
var webpack = require('webpack');
```
3. 在`plugins`节点下新增：
```
new webpack.HotModuleReplacementPlugin()
```

## 使用webpack打包css文件
1. 运行`cnpm i style-loader css-loader --save-dev`
2. 修改`webpack.config.js`这个配置文件：
```
module: { // 用来配置第三方loader模块的
        rules: [ // 文件的匹配规则
            { test: /\.css$/, use: ['style-loader', 'css-loader'] }//处理css文件的规则
        ]
    }
```
3. 注意：`use`表示使用哪些模块来处理`test`所匹配到的文件；`use`中相关loader模块的调用顺序是从后向前调用的；

## 使用webpack打包less文件
1. 运行`cnpm i less-loader less -D`
2. 修改`webpack.config.js`这个配置文件：
```
{ test: /\.less$/, use: ['style-loader', 'css-loader', 'less-loader'] },
```

## 使用webpack打包sass文件
1. 运行`cnpm i sass-loader node-sass --save-dev`
2. 在`webpack.config.js`中添加处理sass文件的loader模块：
```
{ test: /\.scss$/, use: ['style-loader', 'css-loader', 'sass-loader'] }
```

## 使用webpack处理css中的路径
1. 运行`cnpm i url-loader file-loader --save-dev`
2. 在`webpack.config.js`中添加处理url路径的loader模块：
```
{ test: /\.(png|jpg|gif)$/, use: 'url-loader' }
```
3. 可以通过`limit`指定进行base64编码的图片大小；只有小于指定字节（byte）的图片才会进行base64编码：
```
{ test: /\.(png|jpg|gif)$/, use: 'url-loader?limit=43960' },
```

## 使用babel处理高级JS语法
1. 运行`cnpm i babel-core babel-loader babel-plugin-transform-runtime --save-dev`安装babel的相关loader包
2. 运行`cnpm i babel-preset-es2015 babel-preset-stage-0 --save-dev`安装babel转换的语法
3. 在`webpack.config.js`中添加相关loader模块，其中需要注意的是，一定要把`node_modules`文件夹添加到排除项：
```
{ test: /\.js$/, use: 'babel-loader', exclude: /node_modules/ }
```
4. 在项目根目录中添加`.babelrc`文件，并修改这个配置文件如下：
```
{
    "presets":["es2015", "stage-0"],
    "plugins":["transform-runtime"]
}
```
5. **注意：语法插件`babel-preset-es2015`可以更新为`babel-preset-env`，它包含了所有的ES相关的语法；**

## 相关文章
[babel-preset-env：你需要的唯一Babel插件](https://segmentfault.com/p/1210000008466178)
[Runtime transform 运行时编译es6](https://segmentfault.com/a/1190000009065987)

