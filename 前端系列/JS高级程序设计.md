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



