# 前言

笔记来源 -- 千锋教育 JS 视频 + JS 高级程序设计（第四版）+ MDN

可以用 VSCode 对 JS 代码 debug；可以在 JS 代码中加入 `debugger` 进行调试；更多细节请看 VSCode 官网。

# JS 介绍

<b>JS 发展历史</b>

- 1994 年，网景公司（Netscape）发布了 Navigator 浏览器 0.9 版，这是世界上第一款比较成熟的网络浏览器，轰动一时。但是这是一款名副其实的浏览器只能浏览页面，浏览器无法与用户互动，当时解决这个问题有两个办法，一个是采用现有的语言，允许它们直接嵌入网页。另一个是发明一种全新的语言。
  liveScript ==> JavaScript ==> ECMAscript
- 1995 年 Sun 公司将  Oak语言改名为 Java，正式向市场推出。Sun 公司大肆宣传，许诺这种语言可以"一次编写，到处运行"（Write Once, Run Anywhere），它看上去很可能成为未来的主宰。
- 网景公司动了心，决定与 Sun 公司结成联盟
- 34 岁的系统程序员 Brendan Eich 登场了。1995 年 4 月，网景公司录用了他，他只用 10 天时间就把 JavaScript 设计出来了。（多态语言）
- JS 的设计理念
  - 借鉴 C 语言的基本语法; 
  - 借鉴 Java 语言的数据类型和内存管理; 
  - 借鉴 Scheme 语言，将函数提升到"第一等公民"（first class）的地位;
  - 借鉴 Self 语言，使用基于原型（prototype）的继承机制。

<b>JS 的作用</b>

- 常见的网页效果【表单验证，轮播图。。。】
- 与 H5 配合实现游戏【水果忍者： http://www.jq22.com/demo/html5-fruit-ninja/】
- 实现应用级别的程序【http://naotu.baidu.com】
- 实现统计效果【http://echarts.baidu.com/examples/】
- 地理定位等功能【http://lbsyun.baidu.com/jsdemo.htm#i4_5】
- 在线学编程【https://codecombat.163.com/play/】
- 深度学习模型部署

<b>JS 的组成</b>

- ECMASCRIPT：定义了 JavaScript 的语法规范，描述了语言的基本语法和数据类型。
- BOM (Browser Object Model)：浏览器对象模型

  - 有一套成熟的可以操作浏览器的 API，通过 BOM 可以操作浏览器。比如：弹出框、浏览器跳转、获取分辨率等
  - DOM (Document Object Model)：文档对象模型。有一套成熟的可以操作页面元素的 API，通过 DOM 可以操作页面中的元素。比如：增加 div，减少 div，给 div 交互位置等

<b>JS 就是通过固定的语法去操作`浏览器`和`标签结构`来实现网页上的各种效果</b>

# 基础语法

## 书写位置

- 和 `css` 一样，我们的 `js` 也可以有多种方式书写在页面上让其生效
- `js` 也有多种方式书写，分为<b>行内式，内嵌式，外链式</b>

<b>行内式 JS 代码（不推荐）</b>

写在标签上的 js 代码需要依靠事件（行为）来触发

```html
<!-- 写在 a 标签的 href 属性上，
此处的 JavaScript 单词大小写不敏感 -->
<a href="JavaScript:alert('我是一个弹出层');">点击一下试试</a>

<!-- 写在其他元素上 -->
<div onclick="alert('我是一个弹出层')">点一下试试看</div>
```

<b>内嵌式 JS 代码</b>

内嵌式就是指直接把代码写在 html 页面中，内嵌式的 JS 代码会在页面打开的时候直接触发

```html
<!-- 在 html 页面书写一个 script 标签，标签内部书写 js 代码 -->
<script type="text/JavaScript">
	alert('我是一个弹出层')
</script>
<!-- 
	注：script 标签可以放在 head 里面也可以放在 body 里面
-->
```

<b>`外链式 JS 代码（推荐）`</b>

外链式 JS 代码只要引入了 html 页面，就会在页面打开的时候直接触发

新建一个 `.js` 后缀的文件，在文件内书写 js 代码，把写好的 js 文件引入 html 页面

```javascript
// 我是 index.js 文件
alert('我是一个弹出层')
```

```html
<!-- 我是一个 html 文件 -->

<!-- 通过 script 标签的 src 属性，把写好的 js 文件引入页面 -->
<script src="index.js"></script>

<!-- 一个页面可以引入多个 js 文件 -->
<script src="index1.js"></script>
<script src="index2.js"></script>
<script src="index3.js"></script>
```

## 注释

- 学习一个语言，先学习一个语言的注释，因为注释是给我们自己看的，也是给开发人员看的，写好一个注释，也有利于我们以后阅读代码
- JS 的注释和 C 语言类似

<b>单行注释</b>

- 一般就是用来描述下面一行代码的作用
- 可以直接写两个 `/` ，也可以按 `ctrl + /`

```javascript
// 我是一个单行注释

// 下面代码表示在浏览器里面出现一个弹出层
alert('我是一个弹出层')
```

<b>多行注释</b>

- 一般用来写一大段话，或者注释一段代码
- 可以直接写 `/**/` 然后在两个星号中间写注释，也可以按 `shift + alt + a`

```javascript
/*
	我是一个多行注释
*/

/*
	注释的代码不会执行
	alert('我是一个弹出层')
	alert('我是一个弹出层')
*/
alert('我是一个弹出层')
```

## 变量

- 变量指的是在程序中保存数据的一个容器
- 变量是计算机内存中存储数据的标识符，根据变量名称可以获取到内存中存储的数据
- 也就是说，我们向内存中存储了一个数据，然后要给这个数据起一个名字，为了是我们以后再次找到他
- 语法： `let 变量名 = 值`

### 定义变量及赋值

```javascript
// 定义一个变量
var num;

// 给一个变量赋值
num = 100;

// 定义一个变量的同时给其赋值
var num2 = 200;
```

<b>注意</b>

- 一个变量名只能存储一个值
- 当再次给一个变量赋值的时候，前面一次的值就没有了
- 变量名称区分大小写（JS 区分大小写）

### 变量的命名规则和命名规范

规则：必须遵守的，不遵守就是错

1. 一个变量名称可以由<b>数字、字母、英文下划线（_）、美元符号（$）</b>组成
2. 严格区分大小写
3. 不能由数字开头，不要使用中文汉字命名
4. 不能是<b>保留字 / 关键字</b>
5. 不要出现空格

规范： 建议遵守的（开发者默认），不遵守不会报错

1. 变量名尽量有意义（语义化）
2. 遵循驼峰命名规则，由多个单词组成的时候，从第二个单词开始首字母大写

## 数据类型

- 是指我们存储在内存中的数据的类型
- 我们通常分为两大类<b>基本数据类型 & 复杂/引用数据类型</b>

### 基本数据类型

1. 数值类型（number）
   - 一切数字都是数值类型（包括二进制，十进制，十六进制等）
   - NaN（not a number），一个非数字
2. 字符串类型（string）
   - 被引号包裹的所有内容（可以是单引号也可以是双引号）
3. 布尔类型（boolean）
   - 只有两个（true 或者 false）
4. null 类型（null）
   - 只有一个，就是 null，表示空的意思
5. undefined 类型（undefined）
   - 只有一个，就是 undefined，表示没有值的意思

### 复杂数据类型

1. 对象类型（object）
2. 函数类型（function），JS 中函数是一等公民，也是一种对象。
3. symbol 类型（ES 6 引入）

### 判断数据类型

- 既然已经把数据分开了类型，那么我们就要知道我们存储的数据是一个什么类型的数据
- 使用 `typeof` 关键字来进行判断

```javascript
// 第一种使用方式
var n1 = 100;
console.log(typeof n1);

// 第二种使用方式
var s1 = 'abcdefg';
console.log(typeof(s1));
```

### 判断一个变量是不是数字

- 可以使用 `isNaN` 这个方法来判断一个变量是不是数字
- `isNaN` ：is not a number

```javascript
// 如果变量是一个数字
var n1 = 100;
console.log(isNaN(n1)); //=> false

// 如果变量不是一个数字
var s1 = 'Jack'
console.log(isNaN(s1)); //=> true
```

### 数据类型转换

- 数据类型之间的转换，比如数字转成字符串，字符串转成布尔，布尔转成数字等

<b>其他数据类型转成数值</b>

1. `Number(变量)`

   - 可以把一个变量强制转换成数值类型
   - 可以转换小数，会保留小数
   - 可以转换布尔值
   - 遇到不可转换的都会返回 NaN

2. `parseInt(变量)`

   - 从第一位开始检查，是数字就转换，知道一个不是数字的内容
   - 开头就不是数字，那么直接返回 NaN
   - 不认识小数点，只能保留整数

3. `parseFloat(变量)`

   - 从第一位开始检查，是数字就转换，知道一个不是数字的内容

   - 开头就不是数字，那么直接返回 NaN

   - 认识一次小数点

4. 除了加法以外的数学运算

   - 运算符两边都是可运算数字才行
   - 如果运算符任何一遍不是一个可运算数字，那么就会返回 NaN
   - 加法不可以用

<b>其他数据类型转成字符串</b>

1. `变量.toString()`
   - 有一些数据类型不能使用 `toString()` 方法，比如 undefined 和 null
2. `String(变量)`
   - 所有数据类型都可以
3. 使用加法运算
   - 在 JS 里面，`+` 由两个含义
   - 字符串拼接： 只要 `+` 任意一边是字符串，就会进行字符串拼接
   - 加法运算：只有 `+` 两边都是数字的时候，才会进行数学运算

### 其他数据类型转成布尔

在 js 中，只有 `''`、`0`、`null`、`undefined`、`NaN`，这些是 false，其余都是 true

## 运算符

就是在代码里面进行运算的时候使用的符号，不光只是数学运算，我们在 js 里面还有很多的运算方式

### 数学运算符

`+`

- 只有符号两边都是数字的时候才会进行加法运算
- 只要符号任意一边是字符串类型，就会进行字符串拼接

`-`

- 会执行减法运算

- 会自动把两边都转换成数字进行运算

`*`

- 会执行乘法运算
- 会自动把两边都转换成数字进行运算

`/`

- 会执行除法运算
- 会自动把两边都转换成数字进行运算

`%`

- 会执行取余运算
- 会自动把两边都转换成数字进行运算

### 赋值运算符

1. `=`

   - 就是把 `=` 右边的赋值给等号左边的变量名
   - `var num = 100`
   - 就是把 100 赋值给 num 变量
   - 那么 num 变量的值就是 100

2. `+=`

   ```javascript
   var a = 10;
   a += 10;
   console.log(a); //=> 20
   ```

   - `a += 10` 等价于 `a = a + 10`

3. `-=`

   ```javascript
   var a = 10;
   a -= 10;
   console.log(a); //=> 0
   ```

   - `a -= 10` 等价于 `a = a - 10`

4. `*=`

   ```javascript
   var a = 10;
   a *= 10;
   console.log(a); //=> 100
   ```

   - `a *= 10` 等价于 `a = a * 10`

5. `/+`

   ```javascript
   var a = 10;
   a /= 10;
   console.log(a); //=> 1
   ```

   - `a /= 10` 等价于 `a = a / 10`

6. `%=`

   ```javascript
   var a = 10;
   a %= 10;
   console.log(a); //=> 0
   ```

   - `a %= 10` 等价于 `a = a % 10`

###  比较运算符

1. `==`
   - 比较符号两边的值是否相等，不管数据类型
   - `1 == '1'`
   - 两个的值是一样的，所以得到 true
2. `===`
   - 比较符号两边的值和数据类型是否都相等
   - `1 === '1'`
   - 两个值虽然一样，但是因为数据类型不一样，所以得到 false
3. `!=`
   - 比较符号两边的值是否不等
   - `1 != '1'`
   - 因为两边的值是相等的，所以比较他们不等的时候得到 false
4. `!==`
   - 比较符号两边的数据类型和值是否不等
   - `1 !== '1'`
   - 因为两边的数据类型确实不一样，所以得到 true
5. `>=`
   - 比较左边的值是否 大于或等于 右边的值
   - `1 >= 1`  true
   - `1 >= 0`  true
   - `1 >= 2`  false
6. `<=`
   - 比较左边的值是否 小于或等于 右边的值
   - `1 <= 2`  true
   - `1 <= 1`  true
   - `1 <= 0`  false 
7. `>`
   - 比较左边的值是否 大于 右边的值
   - `1 > 0`  true
   - `1 > 1`  false
   - `1 > 2`  false
8. `<`
   - 比较左边的值是否 小于 右边的值
   - `1 < 2`  true
   - `1 < 1` false
   - `1 < 0` false

### 逻辑运算符 

1. `&&`
   - 进行 且 的运算
   - 符号左边必须为 true 并且右边也是 true，才会返回 true
   - 只要有一边不是 true，那么就会返回 false
   - `true && true`  true
   - `true && false`  false
   - `false && true`  false
   - `false && false`  false
2. `||`
   - 进行 或 的运算
   - 符号的左边为 true 或者右边为 true，都会返回 true
   - 只有两边都是 false 的时候才会返回 false
   - `true || true`  true
   - `true || false`  true
   - `false || true`  true
   - `false || false`  false
3. `!`
   - 进行 取反 运算
   - 本身是 true 的，会变成 false
   - 本身是 false 的，会变成 true
   - `!true`  false
   - `!false`  true

### 自增自减运算符（一元运算符）

1. `++`

   - 进行自增运算

   - 分成两种，<b>前置++ / 后置++</b>

   - 前置++，会先把值自动 +1，在返回

     ```javascript
     var a = 10;
     console.log(++a);
     // 会返回 11，并且把 a 的值变成 11
     ```

   - 后置++，会先把值返回，在自动+1

     ```javascript
     var a = 10;
     console.log(a++);
     // 会返回 10，然后把 a 的值变成 11
     ```

2. `--`

   - 进行自减运算
   - 分成两种，<b>前置-- / 后置--</b>
   - 和 `++` 运算符道理一样

## 分支结构

- 我们的 js 代码都是顺序执行的（从上到下）
- 逻辑分支就是根据我们设定好的条件来决定要不要执行某些代码

### if 条件分支

<b>if 语句</b>

- 通过一个 if 语句来决定代码执行与否

- 语法： `if (条件) { 要执行的代码 }`

- 通过 `()` 里面的条件是否成立来决定 `{}` 里面的代码是否执行

  ```javascript
  // 条件为 true 的时候执行 {} 里面的代码
  if (true) {
    alert('因为条件是 true，我会执行')
  }
  
  // 条件为 false 的时候不执行 {} 里面的代码
  if (false) {
  	alert('因为条件是 false，我不会执行')    
  }
  ```

<b>if-else 语句</b>

- 通过 if 条件来决定，执行哪一个 `{}` 里面的代码

- 语法： `if (条件) { 条件为 true 的时候执行 } else { 条件为 false 的时候执行 }`

- 两个 `{}` 内的代码一定有一个会执行

  ```javascript
  // 条件为 true 的时候，会执行 if 后面的 {} 
  if (true) {
    alert('因为条件是 true，我会执行')
  } else {
    alert('因为条件是 true，我不会执行')
  }
  
  // 条件为 false 的时候，会执行 else 后面的 {}
  if (false) {
    alert('因为条件为 false，我不会执行')
  } else {
    alert('因为条件为 false，我会执行')
  }
  ```

<b>if-else-if 语句</b>

- 可以通过 if 和 else if 来设置多个条件进行判断

- 语法：`if (条件1) { 条件1为 true 的时候执行 } else if (条件2) { 条件2为 true 的时候执行 }`

- 会从头开始依次判断条件

  - 如果第一个条件为 true 了，那么就会执行后面的 `{}` 里面的内容
  - 如果第一个条件为 false，那么就会判断第二个条件，依次类推

- 多个 `{}` ，只会有一个被执行，一旦有一个条件为 true 了，后面的就不在判断了

  ```javascript
  // 第一个条件为 true，第二个条件为 false，最终会打印 “我是代码段1”
  if (true) {
    alert('我是代码段1')
  } else if (false) {
  	alert('我是代码段2')           
  }
  
  // 第一个条件为 true，第二个条件为 true，最终会打印 “我是代码段1”
  // 因为只要前面有一个条件满足了，就不会继续判断了
  if (true) {
    alert('我是代码段1')
  } else if (true) {
    alert('我是代码段2')
  }
  
  // 第一个条件为 false，第二个条件为 true，最终会打印 “我是代码段2”
  // 只有前一个条件为 false 的时候才会继续向后判断
  if (false) {
    alert('我是代码段1')
  } else if (true) {
    alert('我是代码段2')
  }
  
  // 第一个条件为 false，第二个条件为 false，最终什么也不会发生
  // 因为当所有条件都为 false 的时候，两个 {} 里面的代码都不会执行
  if (false) {
    alert('我是代码段1')
  } else if (false) {
    alert('我是代码段2')
  }
  ```

<b>if-else-if...else 语句</b>

- 和之前的 `if else if ...` 基本一致，只不过是在所有条件都不满足的时候，执行最后 else 后面的 `{}`

  ```javascript
  // 第一个条件为 false，第二个条件为 false，最终会打印 “我是代码段3”
  // 只有前面所有的条件都不满足的时候会执行 else 后面的 {} 里面的代码
  // 只要前面有一个条件满足了，那么后面的就都不会执行了
  if (false) {
    alert('我是代码段1')
  } else if (false) {
    alert('我是代码段2')
  } else {
    alert('我是代码段3')
  }
  ```

### switch 条件分支

- 也是条件判断语句的一种

- 是对于某一个变量的判断

- 语法：`switch`语句最初只支持 `int` 数据类型。然而，随着语言的发展，`switch` 语句现在可以接受多种数据类型作为其条件表达式，包括 `string`、`boolean`、`symbol` 以及 `enum` 等。

  ```javascript
  switch (要判断的变量) {
    case 情况1:
      情况1要执行的代码
      break
    case 情况2:
      情况2要执行的代码
      break
    case 情况3:
      情况3要执行的代码
      break
    default:
      上述情况都不满足的时候执行的代码
  }
  ```

  - 要判断某一个变量 等于 某一个值得时候使用

- 例子🌰： 根据变量给出的数字显示是星期几

  ```javascript
  var week = 1
  switch (week) {
    case 1:
      alert('星期一')
      break
    case 2:
      alert('星期二')
      break
    case 3:
      alert('星期三')
      break
    case 4:
      alert('星期四')
      break
    case 5:
      alert('星期五')
      break
    case 6:
      alert('星期六')
      break
    case 7:
      alert('星期日')
      break
    default:
      alert('请输入一个 1 ～ 7 之间的数字')
  }
  ```


### 三元运算

- 三元运算，就是用<b>两个符号</b>组成一个语句

- 三元运算只是对 <b>if else</b> 语句的一个简写形式

- 语法： `条件 ? 条件为 true 的时候执行 : 条件为 false 的时候执行`

  ```javascript
  var age = 18;
  age >= 18 ? alert('已经成年') : alert('没有成年')
  ```

## 循环结构

- 循环结构，就是根据某些给出的条件，重复的执行同一段代码
- 循环必须要有某些固定的内容组成
  1. 初始化
  2. 条件判断
  3. 要执行的代码
  4. 自身改变

### while

- `while`，中文叫 当…时，其实就是当条件满足时就执行代码，一旦不满足了就不执行了

- 语法 `while (条件) { 满足条件就执行 }`

- 因为满足条件就执行，所以我们写的时候一定要注意，就是设定一个边界值，不然就一直循环下去了

  ```javascript
  // 1. 初始化条件
  var num = 0;
  // 2. 条件判断
  while (num < 10) {
    // 3. 要执行的代码
    console.log('当前的 num 的值是 ' + num)
    // 4. 自身改变
    num = num + 1
  }
  ```

  - 如果没有自身改变，那么就会一直循环不停了

### do while

- 是一个和 `while` 循环类似的循环

- `while` 会先进行条件判断，满足就执行，不满足直接就不执行了

- 但是 `do while` 循环是，先不管条件，先执行一回，然后在开始进行条件判断

- 语法： `do { 要执行的代码 } while (条件)`

  ```javascript
  // 下面这个代码，条件一开始就不满足，但是依旧会执行一次 do 后面 {} 内部的代码
  var num = 10
  do {
    console.log('我执行了一次')
    num = num + 1
  } while (num < 10)
  ```


### for

- 和 `while` 和 `do while` 循环都不太一样的一种循环结构

- 道理是和其他两种一样的，都是循环执行代码的

- 语法： `for (var i = 0; i < 10; i++) { 要执行的代码 }`

  ```javascript
  // 把初始化，条件判断，自身改变，写在了一起
  for (var i = 1; i <= 10; i++) {
    // 这里写的是要执行的代码
    console.log(i)
  }
  
  // 控制台会依次输出 1 ~ 10 
  ```

- 这个只是看起来不太舒服，但是用起来比较好用

### break-中止循环

- 在循环没有进行完毕的时候，因为我设置的条件满足，提前终止循环

- 比如：我要吃五个包子，吃到三个的时候，不能在吃了，我就停止吃包子这个事情

- 要终止循环，就可以直接使用 `break` 关键字

  ```javascript
  for (var i = 1; i <= 5; i++) {
    // 没循环一次，吃一个包子
    console.log('我吃了一个包子')
    // 当 i 的值为 3 的时候，条件为 true，执行 {} 里面的代码终止循环
    // 循环就不会继续向下执行了，也就没有 4 和 5 了
    if (i === 3) {
      break
    }
  }
  ```

### continue 结束本次循环

- 在循环中，把循环的本次跳过去，继续执行后续的循环

- 比如：吃五个包子，到第三个的时候，第三个掉地下了，不吃了，跳过第三个，继续吃第四个和第五个

- 跳过本次循环，就可以使用 `continue` 关键字

  ```javascript
  for (var i = 1; i <= 5; i++) {
    // 当 i 的值为 3 的时候，执行 {} 里面的代码
    // {} 里面有 continue，那么本次循环后面的代码就都不执行了
    // 自动算作 i 为 3 的这一次结束了，去继续执行 i = 4 的那次循环了
    if (i === 3) {
      console.log('这个是第三个包子，掉地下了，我不吃了')
      continue
    }
    console.log('我吃了一个包子')
  }
  ```


## 函数

- 对于 js 来说，函数就是把任意一段代码放在一个盒子里面

- 在我想要让这段代码执行的时候，直接执行这个盒子里面的代码就行

- 先看一段代码

  ```javascript
  // 这个是我们以前写的一段代码
  for (var i = 0; i < 10; i++) {
    console.log(i)
  }
  
  // 函数，这个 {} 就是那个 “盒子”
  function fn() {
    // 这个函数我们以前写的代码
    for (var i = 0; i < 10; i++) {
      console.log(i)
    }
  }
  
  fn(); // 执行盒子中的代码
  ```

### 函数的两个阶段

按照我们刚才的说法，两个阶段就是<b>放在盒子里面</b>和<b>让盒子里面的代码执行</b>

- 把代码放在盒子里面就是函数的定义阶段
- 函数的定义方式有两种声明式和赋值式

### 定义函数

函数的常见定义方式（非对象内定义）有三种

<b>声明式</b>

- 使用 `function` 这个关键字来声明一个函数

- 语法：

  ```javascript
  function fn() {
    // 一段代码
  }
  // function: 声明函数的关键字，表示接下来是一个函数了
  // fn: 函数的名字，我们自己定义的（遵循变量名的命名规则和命名规范）
  // (): 必须写，是用来放参数的位置（一会我们再聊）
  // {}: 就是我们用来放一段代码的位置（也就是我们刚才说的 “盒子”）
  ```

<b>赋值式</b>

- 其实就是和我们使用 `var` 关键字是一个道理了

- 首先使用 `var` 定义一个变量，把一个函数当作值直接赋值给这个变量就可以了

- 语法： 

  ```javascript
  var fn = function () {
    // 一段代码
  }
  // 不需要在 function 后面书写函数的名字了，因为在前面已经有了
  ```


<b>箭头函数（暂时了解，后面细讲）</b>

新语法，类似于其他语言的 lambda 表达式

```js
var fn = (a,b)=> a+b;
```

### 调用上的区别

- 虽然两种定义方式的调用都是一样的，但是还是有一些区别的

- 声明式函数：调用可以在<b>定义之前或者定义之后</b>

  ```javascript
  // 可以调用
  fn()
  
  // 声明式函数
  function fn() {
    console.log('我是 fn 函数')
  }
  
  // 可以调用
  fn()
  ```

- 赋值式/箭头函数：调用只能在<b>定义之前</b>

  ```javascript
  // 会报错
  fn()
  
  // 赋值式函数
  var fn = function () {
    console.log('我是 fn 函数')
  }
  
  // 可以调用
  fn()
  ```


### 函数参数

和其他语言一样，函数参数分为形参和实参。

- 形参就是一个占位符，告诉你这个地方要传入一个参数
- 实参就是在函数调用的时候给行参赋值，给一个实际的内容的

<b>形参与实参对应的关系</b>

- 形参比实参少，多传入的参数无法通过形参获取（可以通过函数内部的 arguments 参数获取到传给函数的所有变量）

```js
function fn(num1, num2) {
  // 函数内部可以使用 num1 和 num2
    console.log(num1);
    console.log(num2);
    console.log(arguments);
}

// 本次调用的时候，传递了两个实参，100 200 和 300
// 100 对应了 num1，200 对应了 num2，300 没有对应的变量
// 所以在函数内部就没有办法依靠变量来使用 300 这个值
fn(100, 200, 300)
```

- 形参比实参多，多余的形参获取不导致，默认为 undefined。

```js
function fn(num1, num2) {
  // 函数内部可以使用 num1 和 num2
    console.log(num1);
    console.log(num2);
    console.log(arguments);
}


// 本次调用的时候，传递了两个实参，100 对应 num1
// 而 num2 没有实参和其对应，那么 num2 的值就是 undefined
fn(100)
```

### 返回值

- 当我开始执行函数以后，函数内部的代码就会从上到下的依次执行

- 必须要等到函数内的代码执行完毕

- 而 `return` 关键字就是可以在函数中间的位置停掉，让后面的代码不在继续执行

  ```javascript
  function fn() {
    console.log(1)
    console.log(2)
    console.log(3)
    
    // 写了 return 以后，后面的 4 和 5 就不会继续执行了
    return
    console.log(4)
    console.log(5)
  }
  
  // 函数调用
  fn()
  ```

- 函数调用本身也是一个表达式，表达式就应该有一个值出现

- 现在的函数执行完毕之后，是不会有结果出现的

  ```javascript
  // 比如 1 + 2 是一个表达式，那么 这个表达式的结果就是 3
  console.log(1 + 2) // 3
  
  function fn() {
    // 执行代码
  }
  
  // fn() 也是一个表达式，这个表达式就没有结果出现
  console.log(fn()) // undefined
  ```

- `return` 关键字就是可以给函数执行完毕一个结果

  ```javascript
  function fn() {
    // 执行代码
    return 100
  }
  
  // 此时，fn() 这个表达式执行完毕之后就有结果出现了
  console.log(fn()) // 100
  ```

  - 我们可以在函数内部使用 `return` 关键把任何内容当作这个函数运行后的结果

### 函数的优点

- 函数就是对一段代码的封装，在我们想调用的时候调用
- 函数的几个优点
  1. 封装代码，使代码更加简洁
  2. 复用，在重复功能的时候直接调用就好
  3. 代码执行时机，随时可以在我们想要执行的时候执行

### 预解析/变量提升

```js
fn()
console.log(num)

function fn() {
  console.log('我是 fn 函数')
}

var num = 100
```

经过预解析之后可以变形为，函数的声明被提升了。

```javascript
function fn() {
  console.log('我是 fn 函数')
}
var num

fn()
console.log(num)
num = 100
```

赋值是函数会按照 `var` 关键字的规则进行预解析。

看 MDN 补充。

### 作用域

作用域，就是一个变量可以生效的范围；变量不是在所有地方都可以使用的，而这个变量的使用范围就是作用域。

#### 全局作用域

- 全局作用域是最大的作用域

- 在全局作用域中定义的变量可以在任何地方使用

- 页面打开的时候，浏览器会自动给我们生成一个全局作用域 window

- 这个作用域会一直存在，直到页面关闭就销毁了

  ```javascript
  // 下面两个变量都是存在在全局作用域下面的，都是可以在任意地方使用的
  var num = 100
  var num2 = 200
  ```

#### 局部作用域

- 局部作用域就是在全局作用域下面有开辟出来的一个相对小一些的作用域

- 在局部作用域中定义的变量只能在这个局部作用域内部使用

- <b>在 JS 中只有函数能生成一个局部作用域，别的都不行</b>

- 每一个函数，都是一个局部作用域


```javascript
// 这个 num 是一个全局作用域下的变量 在任何地方都可以使用
var num = 100

function fn() {
  // 下面这个变量就是一个 fn 局部作用域内部的变量
  // 只能在 fn 函数内部使用
  var num2 = 200
}

fn()
```

在 JS 中，var 声明的变量作用域是一个函数，而 let 声明的变量作用域是 `{}`；不要去使用 var，而是使用 const 和 let。

#### 变量访问规则

- 首先，在自己的作用域内部查找，如果有，就直接拿来使用
- 如果没有，就去上一级作用域查找，如果有，就拿来使用
- 如果没有，就继续去上一级作用域查找，依次类推
- 如果一直到全局作用域都没有这个变量，那么就会直接报错（该变量 is not defined）

```js
var num = 100

function fn() {
  var num2 = 200
  
  function fun() {
    var num3 = 300
    
    console.log(num3) // 自己作用域内有，拿过来用
    console.log(num2) // 自己作用域内没有，就去上一级，就是 fn 的作用域里面找，发现有，拿过来用
    console.log(num) // 自己这没有，去上一级 fn 那里也没有，再上一级到全局作用域，发现有，直接用
    console.log(a) // 自己没有，一级一级找上去到全局都没有，就会报错
  }
  
  fun()
}

fn()
```

- 当你想给一个变量赋值的时候，那么就先要找到这个变量，在给他赋值；变量赋值规则如下
- 先在自己作用域内部查找，有就直接赋值
- 没有就去上一级作用域内部查找，有就直接赋值
- 在没有再去上一级作用域查找，有就直接赋值
- 如果一直找到全局作用域都没有，那么就把这个变量定义为全局变量，在给他赋值

```javascript
function fn() {
  num = 100
}
fn()

// fn 调用以后，要给 num 赋值
// 查看自己的作用域内部没有 num 变量
// 就会向上一级查找
// 上一级就是全局作用域，发现依旧没有
// 那么就会把 num 定义为全局的变量，并为其赋值
// 所以 fn() 以后，全局就有了一个变量叫做 num 并且值是 100
console.log(num) // 100
```

## 对象

### 创建对象

JS 创建对象的常见三种方式

- 字面量创建对象

```js
// 创建一个空对象
var obj = {}

// 像对象中添加成员
obj.name = 'Jack'
obj.age = 18
```

- 内置构造函数创建对象

```js
// 创建一个空对象
var obj = new Object()

// 向对象中添加成员
obj.name = 'Rose'
obj.age = 20
```

- 自定义函数创建对象

```js
function fn(){
    this.say = function(){
        console.log('hello')
    }
}

let c = new fn();
c.say();
```

- ES6 -- class

```js
class MyClass{
    constructor(){
        console.log('create object');
    }
    say(){
        console.log('hello');
    }
}
let mc = new MyClass();
mc.say()
```

### 继承

JS 中对象的继承方式多种多样。

## 数组

可以存储不同类型的数据。

<b>创建数组</b>

```js
let colors = new Array();
let color = new Array(20);
// 带初始化元素的数组,
let colors = new Array("red","blue") 
// 字面量数组
let colors2 = ['red', 'blue']
```

<b>ES6 新增：创建数组的静态方法</b>

- Array.from(类数组结构, [function]) -- 将类数组结构转换为数组实例（可迭代的也可以转成数组）
- Array.of(arg1,arg2,..etc) -- 将一组参数转换为数组实例

```js
console.log(Array.from('hello world java'))
// 第二个参数是可选的，用于对array做处理
console.log(Array.from('hello world java', x=>x+"?"))

// 创建对象 person，person 中包含一个生成器函数
const iter = {
    // *[Symbol] 表示这是一个生成器函数
    *[Symbol.iterator](){
        yield 1;
        yield 2;
    }
}
console.log(Array.from(iter))
```

还可以使用 from 把集合和 map 转成新数组，对于 map，key-value 被视作数组的一个元素（数组套数组）

```js
const m = new Map().set('k1',1).set('k2',2)
const s = new Set().add(1).add(2)

console.log(Array.from(m)) // [ ['k1',1], ['k2',1] ]
console.log(Array.from(s))

// 将 map 中的 key 作为数组中的元素，抛弃 value
console.log(Array.from(m, e=>e[0])) // ["k1","k2"]
```

还能把对象的属性值转为数组，不过属性要是 0~ 数字，要用 length 属性。

```js
const arrayLikeObject = { 
 0: 1, 
 1: 2, 
 2: 3, 
 3: 4, 
 length: 4 
}; 
console.log(Array.from(arrayLikeObject)); // [1, 2, 3, 4]
```

<b>数组空位</b>

可以用 ,,, 来创建空位 `const options = [, , , , ,]` 五个逗号，创建包含五个元素的数组。但是不要用数组空位，因为对于数组空位，会存在行为不一致和性能隐患，实在要用，就给空位赋值 undefined。

<b>数组索引，可以通过索引给数组`添加元素`</b>

```js
let colors = ['red', 'blue'];
colors[2] = 'green'
colors[100] = 'other'
// 但是 3~99 是没有的，试图访问会出返回 undefined
```

<b>检测数组 -- 使用 Array.isArray 而非 instanceof</b>

instanceof 可以用于判断变量属于什么类型。但是如果有两个页面（iframe）会出现问题。

先了解下 instanceof 的工作原理

instanceof 运算符是 JavaScript 中用于检测一个对象是否由某个构造函数创建的一种方法。它的工作原理是通过检查对象的[[Prototype]] 链来确定该对象是否是特定类或构造函数的实例。

而每个页面可以有自己的全局执行上下文，这意味着它们可以有自己独立的全局对象和构造函数。

- iframe one 创建了一个 Array，把它传递给了 iframe two；
- iframe one 和 two 都有自己的 Array 构造函数的原型链，虽然功能都一样，但是 one 传递过去的 Array 并不在 two 的原型链上，因此会是 false

为解决这个问题，ECMAScript 提供了 Array.isArray()方法。这个方法的目的就是确定一个值是否为数组，而不用管它是在哪个全局执行上下文中创建的。

<b>迭代方法</b>

ES6 中，Array 的原型上暴露了 3 个用于检索数组内容的方法：keys()、values() 和 entries()。keys() 返回数组索引的迭代器，values() 返回数组元素的迭代器，而 entries() 返回索引/值对的迭代器

```js
const a = ["foo", "bar", "baz", "qux"];
// 因为这些方法都返回迭代器，所以可以将它们的内容
// 通过 Array.from()直接转换为数组实例
const aKeys = Array.from(a.keys());
const aValues = Array.from(a.values());
const aEntries = Array.from(a.entries());
console.log(aKeys); // [0, 1, 2, 3] 
console.log(aValues); // ["foo", "bar", "baz", "qux"] 
console.log(aEntries); // [[0, "foo"], [1, "bar"], [2, "baz"], [3, "qux"]] 

// 使用 ES6 的解构可以非常容易地在循环中拆分键 / 值对：
for (const [idx, element] of a.entries()) {
    console.log(idx, element);
}
```

<b>复制和填充</b>

- fill，填充数组
- copyWithin，按指定范围浅复制数组中的部分内容，然后插入到指定索引位置处；仔细阅读源码中的注释
  - target 表示要复制元素的起始索引
  - start 和 end 表示要复制那个范围的数据

```js
const num = [0,0,0,0,0]
num.fill(5) // [5,5,5,5,5]
num.fill(3, 0, 2) // [0,2) 处填充 3

let num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// target, start, end
// 指定 target 为 4, index=4 位置开始的数据会被覆盖
// 多少的数据被覆盖呢？由 start 和 end 决定，没写那就是全覆盖
// start, end 未指定，那么默认 4 后面的全部被覆盖
// 0 1 2 3 0 1 2 3 4 5
// num.copyWithin(4)
// target start 指定了 end 未指定 end 默认为len
// num.copyWithin(0, 8)
num.copyWithin(0, 8, 10) // 和上面的等价

console.log(num)
```

<b>转换</b>

- toString() \ toLocaleString() 转为字符串
- valueOf() 的结果仍然是数组
- 数组.join('分隔符') 使用指定的分隔符拼接元素

<b>stack \ queue \ sort</b>

- Array 可以当 stack 用 -- push / pop
- Array 可以当 queue 用 -- push / shift
- 排序方法，reverse / sort，
  - sort 中可以传入一个比较器，定义排序规则
  - `num.sort((a,b)=>{ return b-a; })`
  - 原地排序

<b>常见操作</b>

列出的 API 都会创建新数组对象

- concat，默认会展平 concat 的数组，

  - 展平：将数组的元素一个一个添加进去，如果是数组套数组，内部的数组不会继续展平。
  - 取消展平行为：将数组对象的 Symbol.isConcatSpreadable 设置为 false

  ```js
  let n1 = [1,2,3]
  let n2 = [4,5,[6]]
  
  n1.concat(n2) // [1, 2, 3, 4, 5, Array(1)]
  n2[Symbol.isConcatSpreadable]=false
  n1.concat(n2)// [1, 2, 3, Array(3)]
  ```

- slice，切片，从数组中切片出部分元素创建一个新数组

  ```js
  let n = [0,1,2,3,4,5]
  n.slice(2) // [2, 3, 4, 5]
  n.slice(2,5)// [2, 3, 4]
  ```

- splice，在数组中插入 / 删除 / 替换 元素，用的非常多~

  - start -- 起始索引
  - deleteCount -- 删除多少元素
  - items -- 将 item 插入到删除元素的位置

  ```js
  let num = [0,1,2,3,4,5]
  // 在索引0处删除0个元素，并在删除元素的索引插入元素 100 和 200
  // 最后 num 为 [100, 200, 0, 1, 2, 3, 4, 5]
  num.splice(0,0,100,200)
  ```

<b>搜索</b>

搜索有三个 function，分别是 indexOf / lastIndexOf / includes 都是采用的严格相等。

介绍下 includes，判断数组中是否包含该元素，可以指定检索的范围

- searchElement 要检索的元素
- fromIndex 从那个 index 开始找元素，可选（js 中 `?fromIndex` 表示参数可选）

<span style="color:blue">此外，数组也提供了 find 和 findIndex 查找元素，查找符合要求的第一个元素，两个函数需要传入一个回调函数（断言函数）判断是否符合查找要求。</span>

```js
// 查找第一个大于 5 的元素
let person = [ {age:18},{age:20} ]
person.find(e=>e.age===20)
// 可以传入 element,index,原数组
person.find((e,index,array)=>array[0].age==20)
```

<b>迭代方法</b>

和 Java 的函数式编程差不多，有 every / filter / forEach / map / some

| 方法    | 作用                                       |
| ------- | ------------------------------------------ |
| every   | 每个元素都返回 true，函数返回 true         |
| filter  | 返回符合条件的元素                         |
| forEach | 拿到每个元素做操作，无返回值，属于消费函数 |
| map     | 可做元素转换，返回转换后的元素             |
| some    | 有一个元素返回 true，函数返回 true         |

```js
let num = [1,2,3,4,5,6,7];
// 每个方法可传入的参数都是这三个
num.map((ele,index,array)=> ele ** index);
```

<b>归并（约）方法</b>

- 归并方法，迭代所有的元素，返回一个最终值，如，用来求和。
- reduce / reduceRight，reduceRight 是从右向左规约
- 方法的参数都是下面四个
  - pre 前一次操作的执行结果
  - next 要和 pre 进行运算的数据
  - index 索引
  - array 原数组

```js
let num = [1,2,3,4,5,6,7];
num.reduce((pre,next,index,array)=>pre+next);
```

## 字符串

## Math & Date

## BOM

## DOM

## Event



# 其他

- this 的问题，this 出现在函数中。
  - 谁调用的函数，this 就是谁
  - `this` 都会被绑定到正在构造的新对象上
- var 声明会被拿到函数或全局作用域的顶部，位于作用域中所有代码之前。

- v8 引擎中，两个对象结构一样，则会共享隐藏的类，省内存，如果其中一个对象修改了属于它的属性（如添加或删除一个函数，绑定或删除一个属性）就不会再共享类了
  - 所以，定义完整的对象属性，而非中途添加
  - 不使用某些值时，置为 null，而非删除，不改变对象的结构，可以继续共享隐藏对象，同时不用的变量也可以被 gc 探查到



# **JavaScript高级程序设计**

## **Date**

​                ● Date 日期

​                ○ Date.parse() 将一个日期字符串转成毫秒

​                ○ 注意，月数是从 0 开始计数的，所以一月对应数字 0

​                ● Date 继承的方法

​                ○ toLocaleString - 返回与浏览器运行的本地环境一致的日期和时间

​                ○ toString - 带时区信息

​                ○ valueOf -- 毫秒数

​                ● 日期格式化方法也有很多，基本都是见名知意

​                ● 也可以单独获取时分秒

## **RegExp**

典型的，用到再查



## **原始值包装类型**

- Boolean 的 toString 返回的是字符串

- Number 可以判断是否是整数和安全整数

  - Number.isInteger(1.00) # True
  - Number.isInteger(1.01) False
  - Number.isSafeInteger(-1 * (2**5200))
  - (10.256).toFixed(2) # 返回固定两个小数的字符串

- String

  - 也没什么特别好记的

  - 字符串迭代和解构

    ```js
    let msg = 'abcdefg';
    // message[要访问的属性]
    // xxx() 调用这个属性
    let sIter = message[Symbol.iterator]();
    for(const item of sIter){
        console.log(item);
    }
    // 前面的 for 循环迭代完了，下面的 next 就拿不到东西了，直接 undefined
    // 也可以调用 next 方法获取下一个元素
    console.log(sIter.next())
    
    // 解构字符串什么意思呢?
    console.log(...message) // 拆成一个个字符输出
    console.log([...message]) // 拆成一个个字符，存到数组里
    ```

  - 字符串大小写转换

    - toUpperCase / toLocaleUpperCase，Locale 是针对特定区域的大小写转换

  - 字符串大小比较

    - localeCompare，所在的地区（国家和语言）决定了这个方法如何比较字符串
    - `>` 粗暴的根据 unicode 码元的值进行比较

  - 字符串编码

    - 以编码方法为例：

      encodeURI()、encodeURIComponent()

      encodeURI 不会编码属于 URL 组件的特殊字符，比如冒号、斜杠、问号、井号

      encodeURIComponent()会编码它发现的所有非标准字符

      ```js
      let uri = 'https://www.baidu.com#index hello'
      
      console.log(encodeURI(uri)) // https://www.baidu.com#index%20hello
      
      console.log(encodeURIComponent(uri)) //https%3A%2F%2Fwww.baidu.com%23index%20hello
      ```

- window 对象

  ```javascript
  let global = function(){
      return this;
  }
  console.log(global()) // 这里是 window 调用的 func 所以 this 是 window
  ```

# 集合引用类型

## Object

创建对象的方式

- 使用 new 和 object 的构造函数
- 使用对象字面量

```javascript
let person = new Object()
person.age = 18

// 对象字面量
let person = {
    age: 17
}
```

在对象字面量表示法中，属性名可以是字符串或数值

```javascript
let person = {
    "name": "jerry"
}
```

对象字面量表示法通常只在为了让属性一目了然时才使用

<b>注意：</b>在使用对象字面量表示法定义对象时，并不会实际调用 Object 构造函数。

<b>对象的字段名可由计算得出</b>

```js
const PREFIX = 'prefix';

let obj = {
    [`${PREFIX}Field`]: 'prefixed field',
    ['suffix'+2]:'hello',
    ['say'](){
        console.log("hello")
    }
};

console.log(obj.prefixField); // 输出：'prefixed field'
obj.say() // hello
```

<b>对象属性的访问方式</b>

- 通过点语法访问
- 使用 `[]` 访问

```js
// 这是必须使用 [] 访问的情况
let person = {
    1:"number1"
}
console.log(person[1])
```

<b>使用对象字面量传递可选参数</b>

```js
function display(args){
    let output = '';
    if (typeof args.name == 'string'){
        output += "name: "+ args.name + "\n";
    }
    if(typeof args.age == 'number'){
        output += 'age:'+ args.age + '\n';
    }
    console.log(output);
}
display({ name:'jerry',age:18 })
display({ name:'tomm'})
```

## Array

可以存储不同类型的数据。

<b>创建数组</b>

```js
let colors = new Array();
let color = new Array(20);
// 带初始化元素的数组,
let colors = new Array("red","blue") 
// 字面量数组
let colors2 = ['red', 'blue']
```

<b>ES6 新增：创建数组的静态方法</b>

- Array.from(类数组结构, [function]) -- 将类数组结构转换为数组实例（可迭代的也可以转成数组）
- Array.of(arg1,arg2,..etc) -- 将一组参数转换为数组实例

```js
console.log(Array.from('hello world java'))
// 第二个参数是可选的，用于对array做处理
console.log(Array.from('hello world java', x=>x+"?"))

// 创建对象 person，person 中包含一个生成器函数
const iter = {
    // *[Symbol] 表示这是一个生成器函数
    *[Symbol.iterator](){
        yield 1;
        yield 2;
    }
}
console.log(Array.from(iter))
```

还可以使用 from 把集合和 map 转成新数组，对于 map，key-value 被视作数组的一个元素（数组套数组）

```js
const m = new Map().set('k1',1).set('k2',2)
const s = new Set().add(1).add(2)

console.log(Array.from(m)) // [ ['k1',1], ['k2',1] ]
console.log(Array.from(s))

// 将 map 中的 key 作为数组中的元素，抛弃 value
console.log(Array.from(m, e=>e[0])) // ["k1","k2"]
```

还能把对象的属性值转为数组，不过属性要是 0~ 数字，要用 length 属性。

```js
const arrayLikeObject = { 
 0: 1, 
 1: 2, 
 2: 3, 
 3: 4, 
 length: 4 
}; 
console.log(Array.from(arrayLikeObject)); // [1, 2, 3, 4]
```

<b>数组空位</b>

可以用 `,,,` 来创建空位 `const options = [, , , , ,]` 五个逗号，创建包含五个元素的数组。但是不要用数组空位，因为对于数组空位，会存在行为不一致和性能隐患，实在要用，就给空位赋值 undefined。

<b>数组索引，可以通过索引给数组`添加元素`</b>

```js
let colors = ['red', 'blue'];
colors[2] = 'green'
colors[100] = 'other'
// 但是 3~99 是没有的，试图访问会出返回 undefined
```

<b>检测数组 -- 使用 Array.isArray 而非 instanceof</b>

instanceof 可以用于判断变量属于什么类型。但是如果有两个页面（iframe）会出现问题。

先了解下 instanceof 的工作原理

instanceof 运算符是 JavaScript 中用于检测一个对象是否由某个构造函数创建的一种方法。它的工作原理是通过检查对象的 [[Prototype]] 链来确定该对象是否是特定类或构造函数的实例。

而每个页面可以有自己的全局执行上下文，这意味着它们可以有自己独立的全局对象和构造函数。

- iframe one 创建了一个 Array，把它传递给了 iframe two；
- iframe one 和 two 都有自己的 Array 构造函数的原型链，虽然功能都一样，但是 one 传递过去的 Array 并不在 two 的原型链上，因此会是 false

为解决这个问题，ECMAScript 提供了 Array.isArray()方法。这个方法的目的就是确定一个值是否为数组，而不用管它是在哪个全局执行上下文中创建的。

<b>迭代方法</b>

ES6 中，Array 的原型上暴露了 3 个用于检索数组内容的方法：keys()、values() 和 entries()。keys() 返回数组索引的迭代器，values() 返回数组元素的迭代器，而 entries() 返回索引/值对的迭代器

```js
const a = ["foo", "bar", "baz", "qux"];
// 因为这些方法都返回迭代器，所以可以将它们的内容
// 通过 Array.from()直接转换为数组实例
const aKeys = Array.from(a.keys());
const aValues = Array.from(a.values());
const aEntries = Array.from(a.entries());
console.log(aKeys); // [0, 1, 2, 3] 
console.log(aValues); // ["foo", "bar", "baz", "qux"] 
console.log(aEntries); // [[0, "foo"], [1, "bar"], [2, "baz"], [3, "qux"]] 

// 使用 ES6 的解构可以非常容易地在循环中拆分键 / 值对：
for (const [idx, element] of a.entries()) {
    console.log(idx, element);
}
```

<b>复制和填充</b>

- fill，填充数组
- copyWithin，按指定范围浅复制数组中的部分内容，然后插入到指定索引位置处；仔细阅读源码中的注释
  - target 表示要复制元素的起始索引
  - start 和 end 表示要复制那个范围的数据

```js
const num = [0,0,0,0,0]
num.fill(5) // [5,5,5,5,5]
num.fill(3, 0, 2) // [0,2) 处填充 3

let num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
// target, start, end
// 指定 target 为 4, index=4 位置开始的数据会被覆盖
// 多少的数据被覆盖呢？由 start 和 end 决定，没写那就是全覆盖
// start, end 未指定，那么默认 4 后面的全部被覆盖
// 0 1 2 3 0 1 2 3 4 5
// num.copyWithin(4)
// target start 指定了 end 未指定 end 默认为len
// num.copyWithin(0, 8)
num.copyWithin(0, 8, 10) // 和上面的等价

console.log(num)
```

<b>转换</b>

- toString() \ toLocaleString() 转为字符串
- valueOf() 的结果仍然是数组
- 数组.join('分隔符') 使用指定的分隔符拼接元素

<b>stack \ queue \ sort</b>

- Array 可以当 stack 用 -- push / pop
- Array 可以当 queue 用 -- push / shift
- 排序方法，reverse / sort，
  - sort 中可以传入一个比较器，定义排序规则
  - `num.sort((a,b)=>{ return b-a; })`
  - 原地排序

<b>常见操作</b>

列出的 API 都会创建新数组对象

- concat，默认会展平 concat 的数组，

  - 展平：将数组的元素一个一个添加进去，如果是数组套数组，内部的数组不会继续展平。
  - 取消展平行为：将数组对象的 Symbol.isConcatSpreadable 设置为 false

  ```js
  let n1 = [1,2,3]
  let n2 = [4,5,[6]]
  
  n1.concat(n2) // [1, 2, 3, 4, 5, Array(1)]
  n2[Symbol.isConcatSpreadable]=false
  n1.concat(n2)// [1, 2, 3, Array(3)]
  ```

- slice，切片，从数组中切片出部分元素创建一个新数组

  ```js
  let n = [0,1,2,3,4,5]
  n.slice(2) // [2, 3, 4, 5]
  n.slice(2,5)// [2, 3, 4]
  ```

- splice，在数组中插入 / 删除 / 替换 元素，用的非常多~

  - start -- 起始索引
  - deleteCount -- 删除多少元素
  - items -- 将 item 插入到删除元素的位置

  ```js
  let num = [0,1,2,3,4,5]
  // 在索引0处删除0个元素，并在删除元素的索引插入元素 100 和 200
  // 最后 num 为 [100, 200, 0, 1, 2, 3, 4, 5]
  num.splice(0,0,100,200)
  ```

<b>搜索</b>

搜索有三个 function，分别是 indexOf / lastIndexOf / includes 都是采用的严格相等。

介绍下 includes，判断数组中是否包含该元素，可以指定检索的范围

- searchElement 要检索的元素
- fromIndex 从那个 index 开始找元素，可选（js 中 `?fromIndex` 表示参数可选）

<span style="color:blue">此外，数组也提供了 find 和 findIndex 查找元素，查找符合要求的第一个元素，两个函数需要传入一个回调函数（断言函数）判断是否符合查找要求。</span>

```js
// 查找第一个大于 5 的元素
let person = [ {age:18},{age:20} ]
person.find(e=>e.age===20)
// 可以传入 element,index,原数组
person.find((e,index,array)=>array[0].age==20)
```

<b>迭代方法</b>

和 Java 的函数式编程差不多，有 every / filter / forEach / map / some

| 方法    | 作用                                       |
| ------- | ------------------------------------------ |
| every   | 每个元素都返回 true，函数返回 true         |
| filter  | 返回符合条件的元素                         |
| forEach | 拿到每个元素做操作，无返回值，属于消费函数 |
| map     | 可做元素转换，返回转换后的元素             |
| some    | 有一个元素返回 true，函数返回 true         |

```js
let num = [1,2,3,4,5,6,7];
// 每个方法可传入的参数都是这三个
num.map((ele,index,array)=> ele ** index);
```

<b>归并（约）方法</b>

- 归并方法，迭代所有的元素，返回一个最终值，如，用来求和。
- reduce / reduceRight，reduceRight 是从右向左规约
- 方法的参数都是下面四个
  - pre 前一次操作的执行结果
  - next 要和 pre 进行运算的数据
  - index 索引
  - array 原数组

```js
let num = [1,2,3,4,5,6,7];
num.reduce((pre,next,index,array)=>pre+next);
```

## 定型数组

提升原生库传输数据的效率。在早期的 WebGL 中，由于 JS 数组和原生数组不匹配，需要转换，有很大的性能开销。

### ArrayBuffer

ArrayBuffer 是所有定型数组及视图引用的基本单位，类似于 C 的 malloc，ArrayBuffer 有如下特点

- 内存分配失败会抛出错误
- 可分配的内存不能超过 Number.MAX_SAFE_INTEGER
- 声明 ArrayBuffer 会将所有的二进制位初始化
- ArrayBuffer 分配的内存可以被 GC

<b>对 ArrayBuffer 进行读写的话需要使用`视图`。</b>

WebGL 用的，不看了。我不用写 WebGL。

## Map

Map 的大多数特性都可以通过 Object 类型实现，但是还是有一些细微差异。

- Map 可用 JS 中任意类型数据作为 key，Object 则只能使用数值、字符串/符号作为 key；
- Map 内部使用的 SameValueZero 比较操作，基本上等同于严格比价。

#### 创建 Map

<b>使用 new 创建空 map</b>

```js
let m = new Map();
```

<b>创建 map 时初始化映射</b>

```js
let m = new Map([
    ['k1','v1'],
    ['k2','v2']
]);

```

使用自定义迭代器创建初始化映射

```js
let m2 = new Map({
    [Symbol.iterator]: function*(){
        yield ['k1','v1'];
        yield ['k2','v2']
    }
});
```

<b>`解释下这个语法`</b>

- `[Symbol.iterator]` 是一个计算属性语法，是给字面量对象设置了一个属性，属性值为 function；此处是自定义了一个迭代器。
- function* 表示这是一个生成器函数，它会返回一个迭代器对象，可以通过调用这个迭代器对象的 `next()` 方法来获取生成的值。
- 每次调用 `next()` 方法，生成器函数就会执行到下一个 `yield` 语句，并将 `yield` 后面的值作为 `next()` 方法的返回值。
- 生成器函数生成的迭代器对象实现了 `next` 方法。这是由 JavaScript 引擎自动完成的

```js
// 验证生成器函数生成的迭代器实现了 next 方法
let generator = function*() {
    yield ['k1', 'v1'];
    yield ['k2', 'v2'];
}();
console.log(generator.next());
```

<b>映射 undefined</b>

```js
let m = new Map([[]]);
m.has(undefined) // true
```

#### CRUD

| 方法            | 描述                              |
| --------------- | --------------------------------- |
| has(key)        | 判断是否存在该 key                |
| get(key)        | 获取 key 对应的 value             |
| set(key, value) | 添加/修改 key-value，支持链式调用 |
| clear()         | 清空 map                          |
| size            | 不是方法，是 number               |

#### 顺序与迭代

Map 会维护 key-value 的插入顺序，因此可以根据插入顺序执行迭代操作。迭代的方式有很多种，假定定义了这样一个 map。

```js
const m = new Map([
    ['k1','v1'],
    ['k2','v2'],
])
```

| 迭代方式                     | 说明                              |
| ---------------------------- | --------------------------------- |
| `m[Symbol.iterator]`         | 获取 map 提供的迭代器进行迭代     |
| `m.entries()`                | 获取 map key-value 组成的 entries |
| `m.keys()`                   | 获取所有的 key                    |
| `m.value()`                  | 获取所有的 value                  |
| `m.forEach((k,v)=>log(k,v))` | forEach 遍历                      |

```js
let m = new Map([
    ['k1','v1'],
    ['k2','v2'],
])
// 此处用了解构语法
for(const [k,v] of m.entries()){
    console.log(k,v);
}

// k1 v1
// k2 v2
```

#### Object or Map？

<b>内存占用</b>

Object 和 Map 的工程级实现在不同浏览器间存在明显差异，但存储单个键/值对所占用的内存数量都会随键的数量线性增加。批量添加或删除键/值对则取决于各浏览器对该类型内存分配的工程实现。不同浏览器的情况不同，但给定固定大小的内存，Map 大约可以比 Object 多存储 50%的键/值对。

<b>插入性能</b>

向 Object 和 Map 中插入新键/值对的消耗大致相当，不过插入 Map 在所有浏览器中一般会稍微快一点儿。对这两个类型来说，插入速度并不会随着键/值对数量而线性增加。如果代码涉及大量插入操作，那么显然 Map 的性能更佳。

<b>查找速度</b>

与插入不同，从大型 Object 和 Map 中查找键/值对的性能差异极小，但如果只包含少量键/值对，则 Object 有时候速度更快。在把 Object 当成数组使用的情况下（比如使用连续整数作为属性），浏览器引擎可以进行优化，在内存中使用更高效的布局。这对 Map 来说是不可能的。对这两个类型而言，查找速度不会随着键/值对数量增加而线性增加。如果代码涉及大量查找操作，那么某些情况下可能选择 Object 更好一些。

<b>删除性能</b>

使用 delete 删除 Object 属性的性能一直以来饱受诟病，目前在很多浏览器中仍然如此。为此，出现了一些伪删除对象属性的操作，包括把属性值设置为 undefined 或 null。但很多时候，这都是一种讨厌的或不适宜的折中。而对大多数浏览器引擎来说，Map 的 delete()操作都比插入和查找更快。如果代码涉及大量删除操作，那么毫无疑问应该选择 Map。

## WeakMap

WeakMap 的 key 只能是 Object，基本用法可 Map 类似，不再赘述。

<b>注意：WeakMap 不可迭代，因为 WeakMap 中的 key-value 随时都可能被销毁</b>

因为 WeakMap 实例不会妨碍垃圾回收，所以非常适合保存关联元数据。来看下面这个例子，其中使用了常规的 Map。

```js
const m = new Map(); 
const loginButton = document.querySelector('#login'); 
// 给这个节点关联一些元数据
m.set(loginButton, {disabled: true});
```

假设在上面的代码执行后，页面被 JavaScript 改变了，原来的登录按钮从 DOM 树中被删掉了。但由于映射中还保存着按钮的引用，所以对应的 DOM 节点仍然会逗留在内存中，除非明确将其从映射中删除或者等到映射本身被销毁。

如果这里使用的是弱映射，如以下代码所示，那么当节点从 DOM 树中被删除后，垃圾回收程序就可以立即释放其内存（假设没有其他地方引用这个对象）

```js
const wm = new WeakMap(); 

const loginButton = document.querySelector('#login'); 

// 给这个节点关联一些元数据
wm.set(loginButton, {disabled: true});
```

## Set

创建 set 的方式和 map 类似，不过 set 没有 key 罢了

```js
let s = new Set([1,2,3])
let s2 = new Set({
    [symbol.iterator]: function*(){
        yield 1;
        yield 2;
    }
})
```

有 has / add / push / delete 等方法。

<b>Set 会维护值插入时的顺序，因此支持按顺序迭代。</b>

迭代方式有：s.values() \ s[Symbol.iterator]

## WeakSet

也是类似于 WeakMap 较于 Map，也是弱集合中的值只能是 Object 或者继承自 Object 的类型，不支持迭代。

## 扩展运算符

`...`

扩展操作符在对可迭代对象执行浅复制时特别有用，只需简单的语法就可以复制整个对象

```js
let arr1 = [1, 2, 3]; 
let arr2 = [...arr1]; 
console.log(arr1); // [1, 2, 3] 
console.log(arr2); // [1, 2, 3] 
console.log(arr1 === arr2); // false
```

对于期待可迭代对象的构造函数，只要传入一个可迭代对象就可以实现复制：

```js
let map1 = new Map([[1, 2], [3, 4]]); 
let map2 = new Map(map1);
```

<b>注意：浅复制意味着只会复制对象引用</b>

# 迭代器和生成器

- 迭代器
- 生成器 -- 要结合 yield 使用， yield* 可以将一个可迭代对象序列化为一连串单独产出的值。

# 面向对象

## 对象

### 回顾创建对象

前面简单学习过 JS 如何创建对象

- new Object()，创建对象，然后给对象绑定属性和方法
- 创建字面量对象
- 使用 class 语法糖
- new 构造方法()

下面是创建对象的一些示例代码

```js
const obj1 = new Object();
obj1.say = function(){ console.log("hello obj"); }

const obj2 = {
    say(){
        console.log('hello obj2')
    }
    say2: function(){
        console.log('hello obj2')
    }
		// 计算属性
    ['name'](){
        console.log('hello obj2')
    }
};
// obj2.name() 调用计算属性方法 name 也可以 obj2['name']()调用
class Obj3{
    constructor(){
        console.log('创建了 Obj3')
    }
}
obj3 = new Obj3()
```

### 属性的类型

ECMA-262 使用一些内部特性来描述属性的特征。这些特性是由为 JavaScript 实现引擎的规范定义的。因此，开发者不能在 JavaScript 中直接访问这些特性。为了将某个特性标识为内部特性，规范会用两个中括号把特性的名称括起来，比如`[[Enumerable]]`

<b>JS 的内部特性会用两个中括号括起来 [[Enumerable]]</b>

<b>属性分类：数据属性 / 访问器属性</b>

| 属性               | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| `[[Configurable]]` | 属性是否可被定义，如 delete、修改特性、及是否可以把它改为访问器属性（可以 setter / getter 的属性）。直接定义在对象上的属性默认为 true |
| `[[Enumerable]]`   | 表示属性是否可以通过 for-in 循环返回。直接定义在对象上的属性默认为 true |
| `[[Writable]]`     | 表示属性的值是否可以被修改。直接定义在对象上的属性默认为 true |
| `[[Value]]`        | 包含属性实际的值。默认为 undefined                           |

可使用 `Object.defineProperty` 为一个空对象设置属性

```js
let person = {};
Object.defineProperty(person,'name',{
    configurable: false,
    value: 'Jerry'
});

// 因此，虽然可以对同一个属性多次调用 Object.defineProperty()，但在把 configurable 设置为 false 之后就会受限制了

// 抛出错误
Object.defineProperty(person,'name',{
    configurable: true,
    value: 'Jerry'
});
```

<b>访问器属性</b>

访问器属性不包含数据值。它们包含一个获取（getter）函数和一个设置（setter）函数，不过这两个函数不是必需的。在读取访问器属性时，会调用获取函数，这个函数的责任就是返回一个有效的值。

访问器属性有 4 个特性描述它们的行为

| 属性               | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| `[[Configurable]]` | 属性是否可被定义，如 delete、修改特性、及是否可以把它改为访问器属性（可以 setter / getter 的属性）。直接定义在对象上的属性默认为 true |
| `[[Enumerable]]`   | 表示属性是否可以通过 for-in 循环返回。直接定义在对象上的属性默认为 true |
| `[[Get]]`          | 获取函数，在读取属性时调用。默认值为 undefined。             |
| `[[Set]]`          | 设置函数，在写入属性时调用。默认值为 undefined。             |

访问器属性不能直接定义，必须使用 `Object.defineProperty()`，对于敏感的有范围限制的属性，可以考虑使用访问器属性。

```js
let book = {
    year_:  2018,
    edition: 1
};

Object.defineProperty(book,'year',{
    get(){
        return this.year_;
    },
    set(newValue){
        if(newValue<=2025){
            this.year_ = newValue;
            this.edition +=newValue-2018;
        }
    }
});

book.year = 2026 // 并不能改变 year_ 的值
```

- 只定义获取函数意味着属性是只读
- 只定义设置函数意味着不可读

同时设置多个属性

```js
let book = {};
Object.defineProperties(book, {
    other1_: { value: 1 },
    other2_: { value: 2 },
    other1: {
        get() { return this.other1_; },
        set(newValue) { this.other1_ = newValue; }
    },
    other2: {
        get() { return this.other2_; },
        set(newVal) { self.other2_ = newVal }
    }
})

console.log(book.other1);
```

函数的若干种写法

```js
let book = {
    year_: 2018,
    edition: 1
};
Object.defineProperties(book, {
    other1_: {
        value: 1
    },
    other2_: {
        value: 2
    },
    other1: {
        get: function () {
            return this.other1_;
        },
        set: function (newVal) {
            self.other1_ = newVal
        }
    },
    other2: {
        get: function () {
            return this.other2_;
        },
        set: function (newVal) {
            self.other2_ = newVal
        }
    }
})

console.log(book.other1);

```

## 可计算属性



# 代理

## 捕获器

```js
const target = {
    foo: 'bar'
};

const handler  = {
    get(trapTarget, property, receiver){
        return 'handler override';
    }
}

const proxy = new Proxy(target, handler);
console.log(target.foo);
console.log(proxy.foo);
```



