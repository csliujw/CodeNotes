# 第一部分 入门

## 第二章 开发环境

### 2.1 Java虚拟机

- JVM：Java虚拟机。Java代码是运行在虚拟机上的。
- 跨平台：代码运行在虚拟机上，不同版的OS（linux，windows，mac）对应不同的虚拟机。虚拟机本身不具备跨平台功能，每个OS下都有不同版本的虚拟机。【可理解为，各个OS下的虚拟机都是采用一套编码指令集】

### 2.2 JRE和JDK

- JRE (Java Runtime Environment):Java程序的运行时环境，包含JVM和运行时所需要的核心类库
- JDK(Java Development Kit):是Java程序开发的工具包，包含JRE和开发人员使用的工具。
- 运行Java程序有jre就行，开发Java程序需要JDK。
- Windows会把%CC% CC当作变量进行翻译

## 第三章 入门程序

### 3.1 程序开发步骤说明

- 编写、编译、运行
- Java源程序-->Java字节码文件-->JVM运行
- Javac.exe 编译器 处理后 字节码
- Java.exe 解释器 解释字节码的内容

## 第四章 常量

- 常量；在程序运行期间固定不变的量。
- 常量的分类
    - 字符串常量：凡是用双引号引起来的部分叫字符串常量。“asdfas”，可以是空串
    - 整数常量：直接写上数字的，没有小数点。
    - 浮点数常量：直接写上数字的，有小数点。
    - 字符常量：用单引号引起来的 ‘A’，不能是空字符‘’。
    - 布尔常量：只有量中取值。true，false
    - 空常量：null，代表没有任何数据。不能直接用来打印。syso(null)是错的。

## 第五章 变量&数据类型

### 5.1 数据类型

#### 5.1.1 基本数据类型

- 整数
    - byte  1个字节
    - short 2个字节
    - int   4个字节
    - long  8个字节
- 浮点数
    - float     4个字节
    - double    8个字节
- 字符型
    - char      2个字节
- 布尔型
    - boolean   1个字节

>**Java中默认类型：整型是int，浮点类型是double**
>想要精确的数字不推荐用double，用BigDemical吧。

#### 5.1.2 引用数据类型

> **字符串，数组，类，接口，Lambda**

注意事项：

- 字符串不是基本数据类型
- 浮点数可能是一个近似值
- 数据范围与字节数不一定相关。如float数据范围比long更大，但float是4字节，long是8字节
- 浮点数默认是double，如果一定要用float，需要加上一个后缀F(推荐大写)
- 如果是整数，默认为int，如果一定要用long，需要加上一个后缀L(推荐大写)

### 5.2 变量

- 变量：程序运行期间，内容可以发生改变的量
- 编译原理，左值与右值

### 5.3 强制数据类型转换 ▲

- 强制类型转换一般不推荐使用，因为可能发生精度损失
- byte，short，char这三种数据类型可发生数学运算。
- byte，short，char在运算时都会被首先提升为int类型，然后再计算。

```java
byte num1 = 40;
byte num2 = 50;
//byte + byte --> int + int
int result = num1 + num2;
//如果用 byte接收 需要强转
byte result = (byte)(num1 + num2);
short同理
```

### 5.4 ASCII码表

```
0 -- 48
A -- 65
a -- 97
```

### 5.5 数字和字符的对照关系表（编码表）

```java
ASCII码表：American Standard Code for Information Interchange
Unicode码表：万国码。也是数字和符号对照关系，开头0-127部分和ASCII完全一样，但是从128开始包含更多字符。
```

## 第六章 常用运算

- 一元运算符：只需要一个数据就可以进行操作的运算符。

    - 取反
    - 自增
    - etc

- 二元运算符：需要两个数据才可以进行操作的运算符。

    - 加法
    - 减法
    - 赋值

- 三元运算符：需要三个数据才可以进行的运算符。

    - 变量名称 = 条件判断?表达式A : 表达式B;
    - int max = a > b ? a : b;

- 拓展

    - 对于byte/short/char三种类型来说，如果右侧赋值的数值没有超过范围，那么java编译器会自动隐含地为我们补上一个(byte)(short)(char).

    ```
    short = 5 + 8;(都要是常量才行)
    等同于
    short = 13;
    先计算出的结果在进行赋值的
    称为编译器的常量优化。
    ```

## 第七章 基本语法

### 7.1 switch语句使用的注意事项

- 多个case后面的数值不可以重复
- switch后面小括号中只能是下列数据类型
    - 基本数据类型 byte/short/char/int
    - 引用数据类型 String字符串、enum枚举

### 7.2 循环概述

- for循环

```java
for( 一 ; 二 ; 四 ){
    三
}

for(初始化表达1 ; 布尔表达式2 ; 步进表达式4){
    循环体3
}
流程 1 2 3 4 --> 2 3 4 --> 2 3 4 -->直到2不满足为止。
初始化语句只会执行一次。
```

- 增强for循环 foreach

```java
for(float x : f){
    System.out.println(x);
}
将每一个f的元素赋值给x
```

### 7.3 方法重载与重写

- 方法调用的三种格式

```java
1.单独调用：方法名称(参数)
2.打印调用：System.out.println(方法名称(参数))
3.赋值调用：数据类型 变量名称 = 方法名称(参数)
```

- 方法重载 Overload ▲

    - ==方法重载== ：指在同一个类中，允许存在一个以上的同名方法，只要它们的参数列表不同即可，与修饰符和返回值类型无关。
    - 参数列表：个数不同，数据类型不同，顺序不同。
    - 重载方法调用：JVM通过方法的参数列表，调用不同的方法。

    ```java
    以下参数顺序不一样也是重载！
    public static void test(int a, short b){
    
    }
    public static void test(short b,int a){
    
    }
    ```

    - 实际上，println就是一个被重载的函数

- 方法重写 Overrider 

    - 子类中出现和父类中一模一样的方法(包括返回值类型,方法名,参数列表）
    - 1.重写的方法必须要和父类一模一样(包括返回值类型,方法名,参数列表)
    - 2.重写的方法可以使用@Override注解来标识

#### 7.3.1 重载的注意事项▲

```java
public static void f1(short i){
    System.out.println("f1(short)");
}
public static void f1(byte i){
    System.out.println("f1(byte)");
}
public static void f1(int i){
    System.out.println("f1(int)");
}
public static void main(String[] args) {
    short i = 5;
    byte ii = 6;
    int iii = 7;
    f1(1);  // f1(int)
    f1(1);	// f1(int)
    f1(1);	// f1(int)
    System.out.println("==========华丽的分割线==========");
    f1(i);	// f1(short)
    f1(ii);	// f1(byte)
    f1(iii);// f1(int)
}
```

### 7.4 数组

#### 7.4.1 数组得初始化

- 动态初始化 -- 指定数组长度

```java
int [] array = new int[300];
```

- 静态初始化 -- 指定数组内容

```
int [] array = new int[]{1,2,3,4,5,6}; // 标准格式
int [] array = {1,2,3,4,5,6}; // 省略格式
静态初始化不能拆分成
int [] array;
array = {1,2,34};
这样是错误得
```

- 总结

```
动态初始化有默认值的过程，
整型 默认为 0
浮点 默认为 0.0
字符 默认为 '\u0000'
布尔 默认为 false
引用 默认为 null

静态初始化也有，不过系统自动马上将默认值替换为了大括号当中的具体数值。
```

#### 7.4.2 数组作为参数，返回值

```java
public static void cals(int[] arr){
    xxxx
}

public static int[] calculate(int a,int b){
    int [] array = {a,b};
    return array;
}

数组作为参数，作为返回值其实都是数组的地址值
```

### 7.5 Java内存划分

- 1.栈（stack）：存放的都是方法中的局部变量。方法的运行一定要在栈中运行
    - 局部变量：方法的参数，或者方法{}内部的变量
    - 作用域：一旦超出作用域，立刻从栈内存当中消失
- 2.堆（heap）：凡是new出来的东西都在堆中
    - 堆里面的数据都有默认值。默认值同上
- 3.方法区（method area）：存储.class相关信息，包含方法的信息。
- 4.本地方法栈（native method stack）：与操作系统相关
- 5.寄存器（register）：与CPU相关

### 7.6 常见异常

> ArrayIndexOfBoundsException

> NullPointException

> OutOfMemmory

# 第二部分 基础

类的初始化过程？？

## 第一章 面向对象概述

类：描述相同事物的共同特征的抽象

对象：具体存在的实例，是真实地。 实例==对象。

代码层面，必须现有类，才能创建出对象。

定义类的格式：

五大成分（自己总结的！！）如果不是五大成分，那么他就不是正确的。

修饰符 class 类名{

​	//1.成员变量（Field 描述类和对象的属性信息）

​	//2.成员方法（Method：描述类或者对象的行为信息）

​	//3.构造器（Constructor：初始化一个类的对象并返回引用）

​	//4.代码块

​	//5.内部类

}

构造器的复习：

​	作用：初始化一个类的对象并返回。

​	格式：

​		修饰符 类名（形参）{

​		}

​	构造器初始化对象的格式

​	类名 对象名称 = new 构造器()

---

this关键字的作用：

​	this代表当前对象的引用

​	this关键字可以用在实例方法和构造器中

​	this用在方法中，谁调用这个方法，this就代表谁。

​	this用在构造器，代表构造器正在初始化那个对象的引用

插件一键生成无参 有参 toString

----

封装的作用：

​	1.可以提高安全性

​	2.可以实现代码组件化

封装的规范：

​	1.建议成员变量都私有

​	2.提供成套的getter+setter方法暴露成员变量的取值和赋值

`小结：`封装的核心思想，，合理隐藏，合理暴露。

----

static关键字（重点）

Java通过成员变量是否有static修饰来区分是类的还是属于对象的。

static == 静态 == 修饰的成员（方法和成员变量）属于类本身。

有static，静态成员变量：属于类本身。

无static，实例成员变量：属于每个实例对象，必须用类的对象来访问。

成员方法也类似：

1、静态方法

2、实例方法

static修饰，属于类本身，与类加载一次，因为只有一份所以可以被类和类的对象共享。

----

成员变量的分类和访问内存图

![image-20210421212413778](..\pics\JavaStrengthen\image-20210421212413778.png)





### 类与类之间的关系

- **依赖（user-a）**：一个类使用了另一个类的方法，A调用了B的方法，B出bug了，A也可能出bug，软件工程中称之为耦合。
- **聚合（has-a）**：一个对象将一个或者多个其它对象作为自己的成员
- **继承（is-a）**：

### 1.1 面向对象

> 当需要实现一个功能时，不关心具体的步骤，而是找一个已经具有该功能的人，来替我们做事。

> 什么叫面向对象：把相关的数据和方法组织为一个整体来看待，从更高的层次来进行系统建模，更贴近事物的自然运行模式【来自百度】

> 面向对象的基本特征：继承，封装，多态

### 1.2 类和对象

- 类：是一组相关 <u>*属性和行为的集合*</u> 。可以看成是一类事物的模板，使用事物的属性特征和行为特征来描述该 类事物。现实中，描述一类事物：
  - 属性：就是该事物的状态信息。 
  - 行为：就是该事物能够做什么。
  - 举例：小猫。
  - 属性：名字、体重、年龄、颜色。  
  - 行为：走、跑、叫。
- **什么是对象** 
  - 对象：是一类事物的具体体现。对象是类的一个实例（对象并不是找个女朋友），必然具备该类事物的属性 和行为。
  - 现实中，一类事物的一个实例：一只小猫。
  - 属性：tom、5kg、2 years、yellow。   
  - 行为：溜墙根走、蹦跶的跑、喵喵叫。 

> 类与对象的关系 ：类是对一类事物的描述，是抽象的。 对象是一类事物的实例，是具体的。 类是对象的模板，对象是类的实体。

### 1.3 一个对象的内存图

方法区中存放class信息。
class中的成员方法一直在方法区中。
堆中拿到成员方法的地址，通过地址对方法进行调用【回忆组成原理】。
堆将方法区中的成员变量拿到堆中（相当于copy一份），对其进行初始化值得操作。【不同对象的成员变量是独立的（非静态成员变量）】
main方法中得变量指向堆中的对象，并对对象进行赋值操作。
stack--栈，FIFO

### 1.4 成员变量和局部变量

```java
/*
局部变量和成员变量
1. 定义的位置不一样【重点】
局部变量：在方法的内部
成员变量：在方法的外部，直接写在类当中

2. 作用范围不一样【重点】
局部变量：只有方法当中才可以使用，出了方法就不能再用
成员变量：整个类全都可以通用。

3. 默认值不一样【重点】
局部变量：没有默认值，如果要想使用，必须手动进行赋值
成员变量：如果没有赋值，会有默认值，规则和数组一样

4. 内存的位置不一样（了解）
局部变量：位于栈内存
成员变量：位于堆内存

5. 生命周期不一样（了解）[通常是这样，但是不绝对]
局部变量：随着方法进栈而诞生，随着方法出栈而消失
成员变量：随着对象创建而诞生，随着对象被垃圾回收而消失
*/
public class Demo01VariableDifference {
 
    String name; // 成员变量

    public void methodA() {
        int num = 20; // 局部变量
        System.out.println(num);
        System.out.println(name);
    }

    public void methodB(int param) { // 方法的参数就是局部变量
        // 参数在方法调用的时候，必然会被赋值的。
        System.out.println(param);

        int age; // 局部变量
        System.out.println(age); // 没赋值不能用

        System.out.println(num); // 错误写法！
        System.out.println(name);
    }

}
```

### 1.5 访问修饰符

> **private/protect/public/默认访问**

#### 1.5.1 private访问属性

> **只有本类中可以随意访问，其他类都不行。**

### 1.6 this关键字

#### 1.6.1 this关键字的一些概念

> **通过谁调用的方法谁就是this。**

> **this只能在方法内部使用，且不能在静态方法中使用。为什么？看JVM。**

> **类加载机制！静态的使用不必对类进行实例化。this指的是当前对象的引用。**

#### 1.6.2 this关键字的一些作用

- 在构造器中调用构造器

```java
public class Flower{
	private int price;
	private String name;
	public Flower(int price){
		this.price = price;
	}
	public Flower(String name){
		this(12);
        this.name = name;
	}
}

PS:
this只能调用一个构造器
this调用的构造器要放在最前面    
```

### 1.6 匿名对象



## 第二章 API

### 概述

`API(Application Programming Interface)`，应用程序编程接口。

### `API`使用步骤 

- 1.打开帮助文档。
- 2.点击显示，找到索引，看到输入框。
- 3.你要找谁？在输入框里输入，然后回车。
- 4.看包。java.lang下的类不需要导包，其他需要。
- 5.看类的解释和说明。
- 6.学习构造方法。

## 第三章 字符串

### 3.1 概述

> **字符串：程序中凡是所有的双引号字符串都是String类的对象【就算没有new，也照样是】**

#### 3.1.1 字符串的特点

- 字符串的内容永不可变。【常量池？】
- 因字符串不可变，故字符串可共享使用【不可变，不会出现线程安全问题】
- 字符串效果相当于char[]字符数组，但底层原理是byte[]字节数组
- String str = "Hello" 也是字符串对象

#### 3.1.2 字符串常量池

> **字符串常量池**：程序中直接写上双引号的字符串，就在字符串常量池中。从jdk1.7开始，字符串常量在堆中。【方便gc嘛？】

> 对于基本类型来说， == 是进行数值比较

> 对应用类型来说，==是进行【地址值】的比较

就算不new 字符串直接双引号也是一个对象。故String str1 是一个对象。

字符串常量池中的对象保持的其实是byte数组的地址值。

而直接new出来的，是不在常量池中的。【具体过程看图。用new String(char型数组)有一个中转过程】
    char[] --> byte[] --> 字符串对象
    字符串对象再指向byte数组

**总结：双引号直接写的在常量池中，new的不在池中。**

```java
public static void main(String[] args) {
    String str1 = "abc";
    String str2 = "abc";

    char[] charArray = {'a', 'b', 'c'};
    String str3 = new String(charArray);

    System.out.println(str1 == str2);// true
    System.out.println(str1 == str3);// false
    System.out.println(str2 == str3);// false

    String str4 = new String("abc");
    String str5 = new String("abc");
    System.out.println(str1 == str4); // false
    System.out.println(str4 == str5); // false
}
```

### 3.2 字符串常用API

#### 3.2.1 字符串的比较▲

> **== 是进行对象的地址值比较。如果确实需要比较字符串的内容，可以使用如下的方法**

```java
public static void testEqual(){
    String str1 = new String("11");
    String str2 = new String("11");

    String str3 = "11";
    System.out.println(str1.equals(str2)); // true
    System.out.println(str1.equals(str3)); // true
    System.out.println(str1.equals("11")); // true
}

String 对equals进行了重写！
```

> **注意事项**

- 如果比较双方一个常量一个变量，推荐把常量写在前面。【避免NullPointerException】
- **equalsIgnoreCase**忽略大小写进行比较。

#### 3.2.2 字符串获取相关方法

- ```java
  - length
  - concat(String str) 拼接会产生新的字符串
  - charAt(int index)
  - indexOf(String str) 查找首次出现的位置，没有返回-1
  
  public static void testGetStr(){
      String str1 = "abc";
      String str2 = "df";
      System.out.println(str1.length()); // 3
      System.out.println(str1.charAt(0)); // a
      System.out.println(str1.concat(str2)); // abcdf
      System.out.println(str1.indexOf("ab")); // 0
      System.out.println(str2.indexOf('d')); // 0
  }
  ```

- concat的测试▲

  ```java
  public void testConcat(){
      String str1 = "abc";
      String str2 = "df";
      String concat = str1.concat(str2);
      String concat2 = "abcdf";
      String concat3 = "abcdf";
      System.out.println(concat == concat2); // false
      System.out.println(concat2 == concat3);// true
  }
  concat内部返回的字符串是使用的new。故会有上述结果！
  ```

#### 3.2.3 字符串截取、转换、分割

> **截取指定索引的数据**

```java
@Test
public void testSubstring(){
    String str1 = "abcefghig";
	// beginIndex
    System.out.println(str1.substring(1));
    // beginIndex, endIndex 左闭右开
    System.out.println(str1.substring(1,str1.length()));
    // false
    System.out.println(str1.substring(1) == str1.substring(1,str1.length()));
}
查看源码可知 返回的是new String
```

> **字符串转换字符数组，字节数组**

```java
@Test
public void testConvert(){
    String str = "hello world";
    char[] chars = str.toCharArray(); // 转化为字符数组
    byte[] bytes = str.getBytes(); // 转化为字节数组
    String replace = str.replace("o", "liu"); // 把所有的o替换成liu
    System.out.println(replace); //hellliu wliurld
}
```

> **分割**

```java
@Test
public void testSplit() {
    String str = "aa,bb,cc";
    String[] split = str.split(","); // 里面是正则表达式
    for (String s : split ) {
        System.out.println(s);// aa bb cc
    }
}

@Test
public void testSplit2() {
    String str = "aa.b.cc";
    String[] split = str.split("\\."); //用.作为划分
    for (String s : split ) {
        System.out.println(s);// aa bb cc
    }
}
```

## 第四章 静态关键字

> **可实现数据共享。static修饰的内容不再属于对象自己，而是属于类的，所以凡是本类的对象，都共享同一份。**

### 4.1 静态概述

- static修饰的成员方法是静态方法，静态方法不属于对象，而是属于类。

- PS： 静态不能直接访问非静态。

  - 因为内存中是【先】有静态内容，【后】有非静态内容

- PS：静态中不能使用this。

  - 因为this代表当前对象，通过谁调用的方法就是当前对象。但是静态与对象无关。静态是【类名称.静态方法】

  - ```java
    new Object().staticMethod(); 最终会翻译成ClassName.staticMethod();
    ```

### 4.1 静态代码块

> 格式

```java
public class ClassName{
	static{
		静态代码块执行。
	}
}
// 特点：当第一次用到本类时，静态代码块执行唯一的一次【静态代码块只执行一次】
// 用到类就行。就是只是类名称.staticMethod()调用也是用到了类，static会被执行。
```

#### 4.1.1 静态代码块的注意事项

- 静态代码块的执行顺序与定义的顺序有关。先定义的先执行。
- 静态代码块的执行顺序优于静态方法，构造方法！【先有静态，再有堆中的对象。静态总是优于非静态。】

#### 4.1.2 静态工具类Arrays

> 常用方法如下：

```java
@Test
public void test1(){
    Integer []array = {1,23,4,5};
    String str = Arrays.toString(array); // 转成String 可以是基本类型 如int
    Arrays.sort(array); // 排序 ascending 升序 可以是基本类型 如int
    Arrays.sort(array, Collections.reverseOrder()); // 反转，变成了降序。注意这个方法要用引用类型
    System.out.println(array[0]);
}
```

- 自定义类型的排序，那么这个定义的类需要有**Comparable或者Comparator接口支持**。

- <span style="color:red">拓展看下Comparable和Comparator的区别和应用</span>

  ```java
  // 挖坑
  ```

#### 4.1.3 静态工具类Math

```java
@Test
public void test1(){
    int abs = Math.abs(-5);
    double ceil = Math.ceil(12.3); // 向上取整 13
    double floor = Math.floor(12.4); // 向下取整 12
    long round = Math.round(12.6); // 13 四舍五入
}
```

## 第五章 继承

> 继承是多态的前提，没有继承就没有多态！

> 继承主要解决的问题是：**共性抽取**

```
父类也称为基类、超类
子类也称为：派生类
在继承关系中：“子类就是一个父类”。也就是说，子类可以被当成父类看待
例如父类是员工，子类是讲师，那么讲师就是一个员工。 关系 is-a。
父类是抽象的，子类是具体的。
```

被继承的类：父类/超类

继承父类的类：子类

继承的作用？

- 提高代码复用
- 功能增强

继承的特点：

- 子类继承了一个父类，子类可以直机得到父类的属性和方法。（私有的好像无法得到？）

----

继承是 is a

组合是 hava a

----

![image-20210422125216087](..\pics\JavaStrengthen\oop_extend_lianxi.png)

----

`继承后子类不能继承的成员★★`

引入：

​	子类继承父类，子类就得到了父类的属性和行为

​	但是并非所有的父类的属性和行为等子类都可继承

子类不能继承父类的东西

​	【无争议的观点】子类不能继承父类的构造器

​	【有争议的观点】子类能否继承父类的私有成员？

​	【有争议的观点】子类能否继承父类的静态成员？

### 5.1 继承中成员变量的访问特点

> 目前关系数据库有六种范式：第一范式（1NF）、第二范式（2NF）、第三范式（3NF）、巴斯-科德范式（BCNF）、第四范式(4NF）和第五范式（5NF，又称完美范式）。

```java
public class DemoExtends extends Fu{
    int a = 100;
    @Test
    public void test1(){
        //运行时看左边。这里就是看Fu类。没有就一级一级向上找。
        Fu de = new DemoExtends();
        System.out.println(de.a); // 10
    }
}
class Fu{
    int a = 10;
}
```

> **区分子类方法中重名的三种变量**

```java
局部变量 直接写
本类的成员变量 this.变量名
父类的成员变量 super.成员变量
```

### 5.2 重写和重载

- 重写：在继承关系中，**方法名称一样，参数列表【也一样】**。覆盖、覆写 === 【没说返回值！】
- 重载：方法名称一样，参数列表【不一样】

> **方法覆盖重写的特点**：创建的是子类对象，则优先用子类方法

- 方法覆盖重写的注意事项

  - 1，必须保证父子类之间方法名相同，参数列表也相同

  - 2，子类方法的返回值必须【小于等于】父类方法的返回值范围。

  - 简而言之，参数必须要一样，且返回类型必须要兼容。

    ```java
    // 子类的返回类型小于等于父类
    public class Zi extends Fu{
        @Override
        public String method(){
            return "123";
        }
    }
    class Fu{
        public Object method(){
            return null;
        }
    }
    // 为什么？ 是因为向上转型安全，向下转型不安全吗？ 百度的，不确定!
    ```
  
- 不管父类使用了那种参数，覆盖此方法的子类也一定要使用相同的参数。而不论父类声明的返回类型是声明，**子类必须要声明返回一样的类型或该类型的子类**。要记得，子类对象得保证能够执行父类得一切。
  
- 3，子类方法的权限必须【大于等于】父类方法的权限修饰符
  
  ```java
    public > protected > (default) > private
    PS : (default)不是关键字default，而是什么都不写，留空！
  ```
  
- 方法重写的应用场景

  - **设计原则**：

    > 对于已经投入使用的类，尽量不要进行修改。推荐定义一个新的类，来重复利用其中共性内容，并且添加改动新内容。

### 5.3 继承中构造方法的访问特点

子类构造方法中默认隐含有一个super()调用，所以一定是先调用父类构造

只有**子类构造方法才能调用父类构造方法**且只能调用一个构造方法！

```java
// 这是错误的，因为只能调用一个父类的构造。
public Zi(){
	super();
	super(5);
}
// 调用普通方法没问题
public Zi(){
    super.method();
    super.qq();
}
```

this调用构造也是只能调用一个，不能循环调用

```
public Zi(int x){
    this();
    System.out.println("int x");
}
```

this不能循环调用【循环引用？Spring循环依赖？】

```java
// 这样是错误的！
public Zi(){
    this(2);
    System.out.println("我是无参");
}

public Zi(int x){
    this();
    System.out.println("int x");
}
```

super和this不能同时显式调用.

```java
// 报错 因为 super or this都需要放在第一行！
public Zi(){
    super();
    this(2);
    System.out.println("我是无参");
}

// 没问题， 父类的构造也是会执行的。
public Zi(){
    this(2);
    System.out.println("我是无参");
}
```

### 5.4 继承中 this和super的内存图

## 第六章 抽象

> **若父类中的方法不能确定如何进行{}方法体实现，那么这就应该是一个抽象方法。**

### 6.1 抽象概述

- 抽象方法：就是加上abstract关键字，然后去掉大括号，直接分号结束
- 抽象类：抽象方法所在的类，必须是抽象类才行！在class之前写上abstract即可
- default关键字

### 6.2 如何使用抽象类和抽象方法

- 不能直接创建new抽象类对象。

- 必须用一个子类来继承抽象父类。

- **子类必须覆盖重写抽象父类当中所有的抽象方法。**

  - 子类重写时，去掉抽象方法的abstract关键字，然后补上方法体。

- 创建子类对象进行使用。

- **PS：Please attention**

  - 抽象类可以自己写构造函数

  - 如果抽象类只有 有参构造，那么子类的构造函数一定要显示调用这个有参构造！

    ```java
    public abstract class Animal {
        public Animal(int x){
            System.out.println(x);
        }
    
        public void say(){
            System.out.println("hello");
        }
    
        public abstract void walk();
    }
    
    public class Cat extends Animal {
    
        public Cat(int x) {
            super(x);
        }
    
        public Cat() {
            super(1);
        }
    
        @Override
        public void walk() {
            System.out.println(":walk");
        }
    
        @Test
        public void test1(){
            new Cat();
        }
    }
    ```
    
  - 抽象类可以实例化，但是不能直接实例化。只能在子类被实例化的过程中，间接实例化。因为实例化子类的时候抽象类也会被实例化。【用的是extends关键字。父类的super会被隐式调用】
  
    <a href="https://zhuanlan.zhihu.com/p/95406830">建议看这篇博文</a>

## 第七章 接口

> **一种公共的规范标准。【定义规范】【多个类的公共规范】**

> **结构是一种引用数据类型，最重要的内容就是其中的：抽象方法**

> 接口中定义的方法**默认使用public abstract修饰**

```java
public interface Name{

}
```

> **注意！Notice!**

- 如果是**Java 7**那么接口中可以包含有
  - 常量
  - 抽象方法
- 如果是**Java 8**还可以额外包含有
  - 默认方法 public default 返回值类型 方法名称( 参数列表 ){  方法体 }
  - 静态方法
- 如果是**Java 9**还可以额外包含有
  - 私有方法

### 7.1 代码示例

> 在任何版本的Java中，接口都能定义抽象方法。
> 格式：public abstract 返回值类型 方法名称(参数列表);

- 注意事项：
  - 接口当中的抽象方法，修饰符必须是两个固定的关键字：public abstract
  - 这两个关键字修饰符，可以选择性地省略。（今天刚学，所以不推荐。）
  - 方法的三要素，可以随意定义。
  - **如果接口的实现类没有覆盖重写接口中所有的抽象方法，那么这个类必须是抽象类！**

```java
// 这是一个抽象方法
public abstract void methodAbs1();

// 这也是抽象方法
abstract void methodAbs2();

// 这也是抽象方法
public void methodAbs3();

// 这也是抽象方法
void methodAbs4();
```

Java 8开始，接口中允许定义默认方法【**接口当中的默认方法，可以解决接口升级问题。**】

```java
// 接口当中的默认方法，可以解决接口升级问题。
public default void say(){
	syso("hello");
}

// 具体解释：接口中的default可以不用被重写。如果我们要扩充接口，但是又不想更改其他已经实现接口的类，可采用default。
```

Java 8开始，接口中允许定义静态方法

```java
public static 返回值类型 方法名称（参数列表）{

}

public interface MyInterface {
    public default void say(){
        System.out.println("hello");
    }

    public default void walk(){
        System.out.println("hello");
    }

    public static void eat(){
        System.out.println("eat");
    }
}

可直接通过【接口名.staticMethod】调用！且只能用接口名调用！不能用实现类调用！
```

**Java 9开始**，接口中允许定义私有方法

普通私有方法，解决多个默认方法之间重复代码问题

```java
private 返回值类型 方法名称(参数列表) {
    方法体
}
```

静态私有方法，解决多个静态方法之间重复代码问题

```java
private static 返回值类型 方法名称(参数列表) {
    方法体
}
```

**接口中可定义常量,且可省略public static final，默认就是他！**【接口中的常量必须赋值！因为有final修饰！】

```java
public static final int num = 10;
接口名.num调用！
```

### 7.2 总结

> **在Java 9+版本中，接口的内容可以有：**

1. - 成员变量其实是常量，格式：
     [public] [static] [final] 数据类型 常量名称 = 数据值;
      注意：
     常量必须进行赋值，而且一旦赋值不能改变。
     常量名称完全大写，用下划线进行分隔。

2. - 接口中最重要的就是抽象方法，格式：
     [public] [abstract] 返回值类型 方法名称(参数列表);
      注意：实现类必须覆盖重写接口所有的抽象方法，除非实现类是抽象类。

3. - 从Java 8开始，接口里允许定义默认方法，格式：
     [public] default 返回值类型 方法名称(参数列表) { 方法体 }
      注意：默认方法也可以被覆盖重写

4. - 从Java 8开始，接口里允许定义静态方法，格式：
     [public] static 返回值类型 方法名称(参数列表) { 方法体 }
      注意：应该通过接口名称进行调用，不能通过实现类对象调用接口静态方法

5. - 从Java 9开始，接口里允许定义私有很乏，格式：
     普通私有方法：private 返回值类型 方法名称(参数列表) { 方法体 }
      静态私有方法：private static 返回值类型 方法名称(参数列表) { 方法体 }
      注意：private的方法只有接口自己才能调用，不能被实现类或别人使用。

### 7.3 接口的注意事项

- 接口中不能有构造方法，不能有静态代码块
- 一个类的直接父类只有一个，但是可同时实现多个接口
- **如果实现类所实现的多个接口中，存在重复的默认方法，那么实现类一定要对冲突的默认方法进行覆盖重写。**
- **一个类如果直接父类当中的方法和接口中的默认方法产生了冲突，优先用父类当中的方法！**

## 第八章 多态

> **extends继承或implements实现是多态性的前提！**

```java
多态写法，父类引用指向子类对象
Fu obj = new Zi();
obj.method();
obj.methodFu();
```

### 8.1 访问成员变量的两种方式

- 直接通过对象名称访问成员变量：看等号左边是谁，优先使用谁，没有则向上找
- 间接通过成员方法访问
- 老毕在讲到多态执行问题时，结合下面的例子，给我们总结了一套口诀：“成员变量，静态方法看左边；非静态方法：编译看左边，运行看右边。”意思是：当父类变量引用子类对象时（Fu f = new Zi();），在这个引用变量f指向的对象中，他的成员变量和静态方法与父类是一致的，他的非静态方法，在编译时是与父类一致的，运行时却与子类一致（发生了复写）。
- 简而言之
  - **成员变量：编译看左边，运行还看左边**
  - **成员方法：编译看左边，运行看右边**

### 8.2 多态的好处

```java
Employee one = new Teacher();
Employee two = new Assistant();
```

**无论右边new的时候换成那个子类对象，等号左边调用方法都不会变化！**

### 8.3 对象的向上、下转型

- 向上转型一定是安全的，没有问题的，正确的。弊端在于，对象一旦向上转型为父类，就无法调用子类原本持有的内容。

- 向下转型是不安全的，使用时一定要保证他本来是猫才能向下转型变成猫

- instanceof进行类型判断

  ```java
  if( animal instanceof Dog){
  	syso("是狗");
  }
  一般先判断是否是该类，是才进行向下转型！
  ```

## 第九章 final

### 9.1 final修饰类

final修饰的类是没有子孙的，但是有父亲（太监类）

```java
// 无法被继承
public final class FinalDemo {
    
}
```

### 9.2 final修饰方法

final修饰的方法是最终方法，不能覆盖重写（override）

```
public final void method(){

}
```

注意：对于类和方法来说，abstract关键字和final关键字不能被同时使用，因为矛盾。

- 因为子类是一定要覆盖重写抽象方法的！

### 9.3 final修饰局部变量

```java
final int num = 3; // 可以
final int num;
num = 3; //可以

public void say(){
    final int num = 3;
    final int num2;java
    num2 = 3;
}
```

- **正确写法：只要保证有唯一一次赋值即可**
- **对于基本类型，不可变局势变量中的数据不可变**
- **对于引用类型，不可以就是变量中的地址值不可变**

### 9.4 final修饰成员变量

> **对于成员变量来说，如果使用final关键字修饰，那么这个变量也照样是不可变**

- 由于成员变量具有默认值，所以**用了final之后必须手动赋值，不会再给默认值！**
- **对于final成员变量，要么直接赋值，要用使用构造方法赋值，二者择其一。**

### PS 权限修饰符

**default 只能同一个类，或同一个包下的进行访问。不同包的，即便是子类也不能访问！**

## 第十章 内部类 

> **分为成员内部类和匿名内部类**

### 10.1 如何使用成员内部类

> 内用外，随便访问；外用内，需要内部类对象。

外部类定义一个方法，通过这个方法获取内部类的实例对象

```java
public InClass getInClass(){
    return new InClass();
}
```

直接new出内部类

```java
OutClass.InClass inClass = new OutClass().new InClass();
```

内部类 外部类的成员变量

```java
public class Outer{
	int num = 10;
	
	public class Inner(){
		int num = 20;
		
		public void method(){
			int num = 30;
			syso(num);// 内部类方法局部变量
			syso(this.num);// 内部类成员变量
			syso(Outer.this.num);//外部类成员变量 区分重名
		}	
	}
}
```

### 10.2 如何使用局部内部类

```java
@Test
public void test(){
    class inner{
        public void innerSay(){
            System.out.println("inner to say hello");
        }
    }
    inner n = new inner();
    n.innerSay();
}
```

> **局部内部类访问所在方法的局部变量，那么这个局部变量必须是【有效final的】**

```java
/*这样写是可以的。因为保证了num确实是不变的，final关键字是可以省略的【java8开始】。如果class前面加了一句num = 29，那就不对了，因为num改变了。*/
/*
为什么要这样做？
	这是害怕类还在，局部变量缺消失了，导致局部内部类无法访问局部变量！
*/
@Test
public void test(){
    int num = 10;
    class inner{
        public void innerSay(){
            System.out.println("inner to say hello"+num);
        }
    }
    inner n = new inner();
    n.innerSay();
}
```

```java
/*
原因：
    1.new出来的对象在堆内存中
    2.局部变量是跟着方法走的，在栈内存中
    3.方法运行结束后，立刻出栈，局部变量就会立刻消失
    4.但是new出来的对象会在堆中持续存在，直到垃圾回收消失。
*/
public Object test() {
    int num = 10;
    class inner {
        public void innerSay() {
            System.out.println("inner to say hello" + num);
        }
    }
    return n;
}

@Test
public void demo() {
    Object test = test();
}
```

- 局部内部类的使用场景

```java
// 当隐式的构造函数不能满足我们的需求，需要显示的构造函数时，使用局部内部类而非匿名内部类！
// 如果不考虑构造函数的问题，两者的功能是一样的。
interface Counter{
    int next();
}
public class LocalInnerClass {
    private int count = 0;
    // 局部内部类实现
    Counter getCounter(final String name){
        class LocalCounter implements Counter{
            public LocalCounter(){ System.out.println("LocalCounter"); }
            @Override
            public int next() {
                System.out.println(name);
                return count++;
            }
        }
        return new LocalCounter();
    }

    // 匿名内部类实现
    Counter getCounter2(final String name){
        return new Counter() {
            //只有一个默认的构造器。不能自行定义
            @Override
            public int next() {
                System.out.println(name);
                return count++;
            }
        };
    }

    public static void main(String[] args) {
        LocalInnerClass in = new LocalInnerClass();
        Counter lucy = in.getCounter("lucy");
        Counter lucy2 = in.getCounter2("lucy2");
        for (int i = 0; i <5 ; i++) {
            System.out.println(lucy.next());
        }
        for(int i = 0; i<5; i++){
            System.out.println(lucy2.next());
        }
    }
}
```



### 10.3 小结

定义一个类的时候，权限修饰符规则

- **外部类： public or default**
- **成员内部类： public protected default private**
- **局部内部类： 什么都不能写！**

### 10.4 匿名内部类

> **类只需要使用一次，那么可省略其定义！改而使用【匿名内部类】**

```java
MyInterface some = new MyInterface(){
	@Override
	public void method(){
		syso();
	}
}
```

```java
//具体例子
interface MyInterface{
    public abstract void say();
}

public class DemoInnerPart {
    public Object testObject(){
        return new Object(){
            @Override
            public String toString(){
                System.out.println("Override toString");
            }
        };
    }
    public MyInterface testObject2(){
        return new MyInterface() {
            @Override
            public void say() {
                System.out.println("say hello!");
            }
        };
    }
}
```


- **匿名内部类的注意事项**
  - 匿名内部类在创建对象的时候，只能使用唯一一次。
  - 如果希望多次创建对象，而且类的内容一样的话，那么必须使用单独定义的实现类！
- **匿名内部类的使用场景**

  - 情况一： 接口、抽象类使用：相当于不用特意去写一个类去实现这个接口的方法，直接在实例化的时候就写好这个方法（接口、抽象类不能实例化，所以采用匿名内部类的方式来写）
  - 情况二：当接口作为参数放在方法体里的时候，用new 接口()的方式来实例独享，则匿名内部类必须要实现这两个方法。
- **为什么需要内部类**

> 一般来说，内部类继承自某个类或实现某个接口，内部类的代码操作创建它的外围类的对象。所以可以认为内部类提供了某种进入其外围类的窗口。

> 内部类最吸引人的原因是，每个内部类都能独立地继承自一个（接口的）实现，所以无论外围类是否已经继承了某个（接口的）实现，对于内部类都没有影响。内部类允许继承多个非接口类型（类或抽象类）

> 个人认为允许多重继承的意思是，内部类对某个类进行重写再调用它的方法。让一个类可以同时使用两个类的特性。【继承一个类，内部类继承其他类，对必要方法进行重写！可以一个类为载体，内部使用多个内部类，从而实现多继承！】

#### 10.5 内部类的继承

**外部类继承另一个外部类的内部类**

```java
class WithInner{
    class Inner{}
}
public class ExtendInnerClass extends WithInner.Inner{
    // public ExtendInnerClass(){} won't compile  写这个构造会导致编译不成功
    public ExtendInnerClass(WithInner w){
        w.super();
    }
    
    public static void main(String[] agrs){
        WithInner wi = new WithInner();
        ExtendInnerClass c = new ExtendInnerClass(wi);
    }
}
```

**内部类的覆盖**

```java
class WithInner2{
    public class Inner{
        public void say(){ System.out.println("I am say1"); }

        public void walk(){ System.out.println("I am walk"); }
    }
}

public class ExtendInnerClass2 extends WithInner2{
    public class Inner2 extends WithInner2.Inner{
        @Override
        public void say(){ System.out.println("I am say2"); }
    }

    public static void main(String[] args) {
        ExtendInnerClass2 class2 = new ExtendInnerClass2();
        Inner inner2 = new ExtendInnerClass2().new Inner2();
        inner2.say();   // I am say2
        inner2.walk(); // I am walk
    }
}

```

## 第十一章

### 11.1 Object

- `toString()方法`
- `equals()方法`
  - 注意，当需要比较对象时，覆写equals方法，以便于比较对象的大小。
  - 当需要使用Arrays工具类进行大小比较时，需要继承Comparable。

### 11.2 日期相关

```
Calendar采用了枚举，见名知意。可避免不必要的错误。似乎不常用，不学
```

- 日期类都采用单例模式？保证所有时间的一致？
- 不要求掌握的一些

```java
@Test
public void fn1(){
    final Calendar instance = Calendar.getInstance();
    DateFormat dateInstance = DateFormat.getDateInstance(DateFormat.FULL);
    System.out.println(dateInstance.format(new Date())); // 地理位置的判断？输出的中文？
}

@Testjava
public void fn2(){
    // G表示公元  字母大小写不能错，不知道为什么，无责任猜测，解析了字符串，提取的ascill码？
    SimpleDateFormat sdf = new SimpleDateFormat("Gyyyy MM dd"); // 日期格式
    System.out.println(sdf.format(new Date()));
}
```

- Java8 提供的日期类 **都是final修饰的**

```
Instant——它代表的是时间戳

LocalDate——不包含具体时间的日期，比如2014-01-14。它可以用来存储生日，周年纪念日，入职日期等。

LocalTime——它代表的是不含日期的时间

LocalDateTime——它包含了日期及时间，不过还是没有偏移信息或者说时区。

ZonedDateTime——这是一个包含时区的完整的日期时间，偏移量是以UTC/格林威治时间为基准的
```

- Date仅仅含日期。不包含具体时间，有time的才有具体的时间（精确到时分秒）

```java
public void fn3(){
    //无时区
    LocalDate now = LocalDate.now();
    System.out.println(now); // 2020-02-06

    int year = now.getYear();
    int month = now.getMonthValue();
    Month monthE = now.getMonth();
    int day = now.getDayOfMonth();
    System.out.println("year:"+year+" month:"+month+" day:"+day);

    int dayOfYear = now.getDayOfYear();
    System.out.println("2020年的第"+dayOfYear+"天");
}

@Test
public void fn4(){
    LocalDateTime now = LocalDateTime.now();
    System.out.println(now);
    LocalDateTime now2 = now.plusDays(10);
    System.out.println(now2);

    LocalDateTime plus = now.plus(1, ChronoUnit.YEARS);java
    System.out.println(plus);
}
```

- 要用再查API
- System类

```java
@Test
public void fn5(){
    Properties properties = System.getProperties(); // 获取System的properties对象
    Enumeration<?> enumeration = properties.propertyNames();// 获得所有的key
    while(enumeration.hasMoreElements()){ //是否还有元素
        // 安全的强转
        Object c= enumeration.nextElement();
        if(c instanceof String){
            System.out.println("key:"+(String)c+" ---value:"+System.getProperty((String)c));
        }

        Object cc = null;
        if((cc = enumeration.nextElement()) instanceof String){
            System.out.println("key:"+(String)cc+" ---javavalue:"+System.getProperty((String)cc));
        }
    }
}
**回忆迭代器的元素遍历，回忆为什么迭代器遍历元素时可以进行元素移除的操作不会发生异常！**
```

- arraycopy的使用

```java
@Test
public void fn6(){
    int[] fromArray = {1,2,3,4,5,4};
    int[] toArray = {50,60,70,80,90};
    /**
    * @param      src      源数组
    * @param      srcPos   源数组的其实位置 
    * @param      dest     目标数组
    * @param      destPos  目标数组的开始位置
    * @param      length   拷贝的长度
    */
    System.arraycopy(fromArray,1,toArray,2,2);
    for (int i = 0; i <toArray.length ; i++) {
        System.out.println(toArray[i]);
    }
}
```

## 第十二章 集合框架

### 0 引言（Think in Java 第11章 持有对象）

> **Java容器类类库的用途是“保存对象”。可细分为单列集合，双列集合！**

- Collection。一个独立元素序列
- Map。一组成队的“键值对”对象，允许使用键来查找值

- 添加一组元素的 

```java
@Test
public void fn2(){
    Collection<Integer> c = new ArrayList<Integer>(Arrays.asList(1,2,3,4,5,6,7));
    Integer []arr = {1,2,3,4,5};
    // 这里需要Integer类型的数组！
    Collection<Integer> c1 = new ArrayList<Integer>(Arrays.asList(arr)); 
    for (Integer i: c)
        System.out.print(i+"\t");
    for ( Integer i: c1)
        System.out.print(i+"\t");
}

@Test
public void fn1(){
    Integer []array = {1,23,4,5};
    Collection<Integer> coll = new ArrayList<Integer>(Arrays.asList(array));
    System.out.println(coll.size());
    coll.addAll(Arrays.asList(array));
    System.out.println(coll.size());

    // 可变长参数 自动装箱拆箱
    Collections.addAll(coll,1,2,3,5);
    System.out.println(coll.size());
}

@Test
public void fn2(){
    Integer []array = {1,23,4,5};
    Collection<Integer> coll = new ArrayList<Integer>(Arrays.asList(array));
    Iterator<Integer> iterator = coll.iterator();
    //迭代器遍历
    while(iterator.hasNext()){ // hasNext仅仅判断是否有元素
        Integer next = iterator.next(); // 依稀记得有指针后移的操作 后面补充
        System.out.println(next);
        iterator.remove();
    }
}
```

- 这节就一个核心观点，使用泛型，安全！强制类型转换时使用instanceof进行检测！

### 12.1 List集合

```java
@Test
public void fn1(){
    // 不使用多态，便于测试特有的实现方法
    ArrayList<Integer> list = new ArrayList<Integer>();
    /**
         * ArrayList初始化时有容量。默认10.当用到了一定比例的空间会自行进行扩充
         * 简而言之：可变长数组！
         * 如果存储空间不足，会扩大至原来大小的2倍
         * int newCapacity = oldCapacity + (oldCapacity >> 1);
         * 看remove方法的代码，似乎没有发现明显的当用的空间不多时，对数组大小进行缩减！
         *
         * 可变长数组代码的策略
         *      长度不够时进行数组长度的扩充，创建一个新的，大小时原来2-3倍的，把数组copy进行曲
         *      所用的空间不多时，对数组长度进行缩减。创建一个新的长度小的数组，把oldValue复制进去
         *
         *      度的把握：用了2/3时就进行数组的扩充
         *               元素只剩1/3（好像是1/2）时才进行数组的缩减.缩减的策略比较保守
         *               主要是因为程序的局部性原理！
         */
    for (int i = 0; i <5 ; i++) {
        list.add(10);
    }
    Integer i = 10;
    list.remove(i);// remove(Object o)  这个是对象。
    // list.remove(10); 这个识别成了 index 所以报错
    System.out.println(list.size()); // 0
    list.set(0,100);
    System.out.println(list.get(0));
    System.out.println(list.lastIndexOf(10));
    // list.forEach(); 函数式编程 后期补充
    // list.equals() 对象比较时 记得按需求考虑是否重写对象的equals方法
}

@Test
public void fn2(){
    // 可充当 队列 / 栈？
    LinkedList<Integer> list = new LinkedList<>();
    for (int i = 0; i <5 ; i++) {
        list.add(i);
    }
    list.offer(100);
    System.out.println(list.getLast()); // 记单词 tail 尾部
}

@Test
public void fn3(){
    LinkedList<Integer> list = new LinkedList<>();
    for (int i = 0; i <5 ; i++) {
        // 插入 begin stack后进先出FIFO
        list.addFirst(i);
    }
    // 后进入的在head 故获取First
    System.out.println(list.getFirst());
}
```

### 12.2 Set集合

> **set无重复元素**

- `TreeSet` 有序，红黑树

```java
@Test
public void fn4(){
    /**
     * 内部使用的红黑树，我也不知道红黑树是啥
     * 二叉排序树 --> AVL --> 红黑树
     * 应该都满足，中序遍历结果是有序的！
    */
    TreeSet<Integer> set = new TreeSet<>();
    for (int i = 0; i <100 ; i++) {
        set.add((int)(Math.random()*100));
    }
    System.out.println(set.size());
    for(int i : set){
        System.out.println(i);
    }
}
```

- `HashSet` 散列表，无序

```java
@Test
public void fn5(){
    HashSet<Integer> set = new HashSet<>();
    for (int i = 0; i <100 ; i++) {
        // 看不懂代码。不看了。知道散列表的基本写法就算了.
        set.add((int)(Math.random()*100));
    }
    System.out.println(set.size());
    for(int i : set){
        System.out.println(i);
    }
}
```

- 对象之间用Set

```java
package com.bbxx.list;

import java.util.Objects;
import java.util.TreeSet;
/**
 * 类大小比较
 * 依据年龄 姓名进行比较
 */
public class Student implements Comparable {

    public static void main(String[] args) {
        TreeSet<Student> set = new TreeSet<Student>();
        for (int i = 0; i <10 ; i++) {
            set.add(new Student(i+5,i+"s"));
        }
        set.add(new Student(6,null));
        for(Student ss : set){
            System.out.println(ss);
        }
        /**
         * 总结
         * TreeSet采用的红黑树。其应该是符合二叉排序树的性质。中序遍历是有序的。
         * 中序遍历为从小到大的顺序。所以是从小到大来输出。
         *
         * comparable的compareTo方法返回值的解释。
         * 返回正数表示大于。返回0等于，返回负数表示小于!
         *
         * 查看TreeSet add的源码试试 发现 看不懂！
         * 采取代码测试
         */
        Student obj1 = new Student(6, "kkx");
        Student obj2 = new Student(6, "kkx1");
        Student obj3 = new Student(7, "kkx3");
        Student obj4 = new Student(8, "kkx1");
        // -1 如果是表示小于那么set集合的输出顺序是obj1在前
        System.out.println(obj1.compareTo(obj2));
        set.clear();
        set.add(obj1);
        set.add(obj2);
        //测试结果表明 的确是小于。
        for(Student ss : set){
            System.out.println(ss);
        }
        /**
         * 总结：
         *  comparable的compareTo方法返回值的解释。
         *   返回正数表示大于。返回0等于，返回负数表示小于!
         *   obj1.compareTo(obj2) 比较1 和 2的大小。返回正数则 obj1大
         */
    }
    //方便操作
    public int age;
    public String name;

    public Student(){}
    public Student(int age,String name){
        this.age = age;
        this.name = name;
    }
    @Override
    public String toString() {
        return "Student{" +
                "age=" + age +
                ", name='" + name + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Student student = (Student) o;
        return age == student.age &&
                Objects.equals(name, student.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(age, name);
    }

    @Override
    public int compareTo(Object o) {
        Object obj;
        // 不属于该类
        if (!((obj = o) instanceof Student)) {
            System.out.println("对象错误");
            return -1;
        }
        o = (Student) o;
        if (this.equals(o)) return 0;
        // 优先通过年龄判断
        if (this.age > ((Student) o).age) return 1;
        // 其次通过姓名判断
        if (this.age == ((Student) o).age) {
            if(this.name==null && ((Student) o).name==null) return 0;
            if(this.name == null && ((Student) o).name!=null) return -1;
            int len = this.name.compareTo(((Student) o).name);
            if (len == 0) return 0;
            else if (len > 0) return 1;
        }
        return -1;
    }
}
```

### 12.3 Map集合

> **常用的有 `HashMap`和`TreeMap`**

- `HashMap`相关

  - 基本原理：`Java1.8`后是 红黑树+散列表。最开始是散列表的拉链法，链长度超过八是链转为红黑树！
  - `HashMap`的key可以存入null**
  - 基本操作：

  ```java
  public void fn1(){
      // map的存储 遍历  指定泛型，安全
      Map map = new HashMap<Integer,String>();
      map.put(1,"AA");
      map.put(12,"BB");
      map.put(13,"CC");
      map.put(1,"DD");
  
      // map的基本遍历有两种方式
      // 先获取所有的key  @return a set view of the keys contained in this map
      Set set = map.keySet();
      Iterator iterator = set.iterator();
      while(iterator.hasNext()){
          System.out.println(map.get(iterator.next()));
      }
      System.out.println("*************华丽的分割线*************");
  
      // @return a set view of the mappings contained in this map
      // 记不清就点进去看他的返回值回忆具体操作
      Set set1 = map.entrySet();
      Iterator iterator1 = set1.iterator();
      while(iterator1.hasNext()){
          // Map.Entry<Integer, String> 内部接口
          Map.Entry<Integer, String> next = (Map.Entry<Integer, String>)iterator1.next();
          System.out.println(next.getKey()+"=="+next.getValue());
      }
  }
  ```

  - **PS**：开始没必要学太细，第17章有深入理解集合的内容！

    **`HashMap`对象的key、value值均可为null。且`HashMap`是线程不安全的**

    **`HahTable`对象的key、value值均不可为null。且`HashTable`是线程安全的**，put方法用synchronized锁了！好多方法也用synchronized锁了。如remove这些方法！

  ```java
  public void fn1(){
      Hashtable<Integer, String> table = new Hashtable<>();
      // Make sure the value is not null
      // 测试时 发现 key也不能为null，key为null时，没有对应的处理策略
      table.put(null,"ss");
  
      // map的存储 遍历  指定泛型，安全
      HashMap map = new HashMap<Integer,String>();
      map.put(1,"AA");
      map.put(12,"BB");
      map.put(13,"CC");
      map.put(1,"DD");
      // 如果key为null时有处理策略的 return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
      map.put(null,null);
  ```

- `TreeMap`基本内容

  ```java
  public void fn2(){
      // 盲猜 TreeMap的key有二叉排序树的性质 中序遍历为从小到大 内部采用的红黑树。
      // 暂时用二叉排的性质去理解。
      // String 内部的排序 比较的时ASCII码值 Unicode包含ASCII的所有码值
      TreeMap<String, String> map = new TreeMap<String, String>();
      map.put("AA","AA");
      map.put("BB","BB");
      map.put("B123B","CC");
      map.put("23BB","DD");
      Set<Map.Entry<String, String>> entries = map.entrySet();
      Iterator<Map.Entry<String, String>> iterator = entries.iterator();
      while(iterator.hasNext()){
          Map.Entry<String, String> next = iterator.next();
          // 有时候不用泛型 代码返回值就是舒服
     System.out.println(next.getKey()+":"+next.getValue());
      }
  }
  ```

- Properties集合

  > `HashTable`的子类。常用于存储一些配置信息。回忆`properties`文件，好像是的。还有一个properties流？果不其然，有load方法传入的对象是输入流！

  -----
  
  ```java
  public void fn3(){
      Properties properties = new Properties();
      // 仅仅可以为String，应该是专门为配置文件所产生的一个map
      properties.setProperty("name","kkx");
      properties.setProperty("age","18");
      properties.setProperty("sex","xxx");
      Set<Map.Entry<Object, Object>> entries = properties.entrySet();
      Iterator<Map.Entry<Object, Object>> iterator = entries.iterator();
      while(iterator.hasNext()){
          Map.Entry<Object, Object> next = iterator.next();
          System.out.println(next.getKey()+":"+next.getValue());
      }
      Runtime runtime = Runtime.getRuntime();java
  }
  ```

### 12.4 集合工具类

集合工具类 Collections：排序、复制、翻转等操作

数据工具类 Arrays：排序、复制、翻转等操作，Arrays.sort(数组)

排序默认是字典顺序，从小到大。

> **Collections**

```java
Collections.max(list);
Collections.min(list);
Collections.binarySearch(list,find_value);
Collections.shuffle(list); // 洗牌，打乱数据的顺序
Collections.reverse(list); // 反转
Collections.swap(list,2,3);// 2  3 位置的数据交换
Collections.replaceAll(list,"a","A"); // 所有小写a替换成大写A
Collections.fill(list,"h"); // 全部填充为h
```

> **Arrays**

```java
// 与Collections没什么区别
```

### 12.5 比较器

用户自定义对象需要排序的话就需要比较器了~

自定义比较器：

- Comparable：内部比较器，需要修改被比较的对象Person
- Comparator：外部比较器，不需要修改被比较的对象Person

```java
// 内部比较器
/*
返回值
    1  正数 当前对象大 [降序，怎么理解，，，]
    0  一样大
    -1 负数 当前对象小，传入的对象大
    
    这样记忆吧。假设当前对象位置是0。
    当前对象大，返回1，新对象就在1了，降序，就是大-->小
    当前对象小，返回-1，那么新对象就插在-1处，就是：小-->大
*/
```

思路：将比较的对象（Person）实现Comparable接口，重写compareTo方法，在该方法内写比较的逻辑。重点返回值是：-1，0，1

```java
// 外部比较器，无侵入性，传给集合
// 这种没必要记，写个demo测一下就可以了~~~
public class myxx implements Comparator{
    public int compare(Object o1,Object o2){
        // 强转
        return s1.age - s2.age;
    }
}
```



## 第十三章 异常

## 第十四章 注解

## 第十五章 IO流

> IO流可大致分为字节流和字符流。字节是按字节进行输入输出的，适用于各种文件。字符流是按字符进行输入输出的，适用于文本文件。

> **IO流文件的创建读取，采用相对路径是以当前项目为基准的！**

- 输入流：其他地方向内存中输入。 	xx--->内存
- 输出流：从内存中输出到其他地方。 内存--->其他

### 15.1 字节流

无论何种文件，都是以二进制（字节）的形式存储在计算机中。可操作Computer中的任何文件。

**字节流通常以`InputStream`或`OutputStream`结尾**

#### 15.1.1 文件的输入（读取文件）

```java
public void fn1() throws IOException {
    // 通过类加载器获得classpath下的文件（就是src目录下的文件）
    InputStream in1 = this.getClass().getClassLoader().getResourceAsStream("test.txt");
    InputStream in2 = new FileInputStream("E:\\xx\\JavaDay08( IO )\\src\\test.txt");
    // 断言是否为空 不为空 说明找到了文件
    Assert.assertNotNull(in2);
    int b = 0;
    while ((b = in1.read()) != -1) {
        System.out.print((char)b);
    }
}
```

- 关于文件的路径问题

```
1、对于File类构造方法，他需要的传入的是一个“路径名字符串”，而并不是一个单纯的文件名，对吧兄弟。

2、对于jvm来说，在classloader加载时候，你所以存放的d.txt也会随classloader进行加载，因此他们属于同级目录。

3、如果楼主真心想采用d.txt来读取的话。可以使用classloader加载原理来读取。

此方法需要注意，静态方法（通过当前的classloader加载的类来获取当前d.txt被加载的路径）
```

#### 15.1.2 文件的输出（写入）

```java
@Test
public void fn2() throws IOException {
    // 只写文件是默认创建在与src同级目录。就是a.txt的目录和src同级
    FileOutputStream fos1 = new FileOutputStream("a.txt");
    // 写绝对路径的话
    FileOutputStream fos2 = new FileOutputStream("E://a.txt");
    String str = "!23";
    // 直接写一个字节数组
    fos1.write(str.getBytes());
    // 一个一个字节写
    byte[] bytes = str.getBytes();
    for (int i = 0; i < bytes.length; i++) {
        fos2.write(bytes[i]);
    }
}
```

```java
// 追加写入
@Test
public void fn3() throws Exception{
    // public FileOutputStream(String name, boolean append) append = true 追加写入
    FileOutputStream fio = new FileOutputStream("aaaa.txt", true);
    fio.write("liujiawei".getBytes());
}
```

#### 15.1.3 文件的复制

- 获取输入流，将内容读入内存
- 获取输出流，将读入的内容写到磁盘

```java
@Test
public void fn4() throws Exception{
    // 1. 创建输入流 准备读入文件
    FileInputStream fis = new FileInputStream("E://note.docx");
    // 2. 创建输出流 准备写文件到外存
    FileOutputStream fos = new FileOutputStream("copyNote.docx");
    // 3. 逐步将读到的文件 写到外存
    int b = 0;
    while((b = fis.read())!=-1){
        fos.write(b);
    }
    fis.close();
    fos.close();
}

@Test
public void fn5() throws IOException {
    // 加强版，依次读一串。
    FileInputStream fis = new FileInputStream("E://note.docx");
    FileOutputStream fos = new FileOutputStream("copy2Note.docx");
    byte[] bytes = new byte[1024];
    int read = fis.read(bytes);
    while((read = fis.read(bytes))!=-1){
        fos.write(bytes,0,bytes.length);
    }
    fis.close();
    fos.close();
}
```

#### 15.1.4 字节缓冲流

```java
@Test
public void fn6() throws Exception {
    // 字节缓冲流 看源码可以知道 bf默认有一个8192的字节数组。
    // bis读取时一次读取8192字节
    // bos 写入时 write(len) 写入指定长度的数据。 bis的buff字节数组用volatile修饰了，应该是给当前线程的xx
    // 查看资源用的，bos写入时好获得要写入的字节数组
    BufferedInputStream bis = new BufferedInputStream(new FileInputStream("E://note.docx"));
    BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("note3.docx"));
    int len = 0;
    while((len = bis.read())!=-1){
        bos.write(len);
    }
}
```

### 15.2 字符流

- 为什么出现字符流？

> 有些是好几个字节组成一个字符，一个一个字节读，输出的数据可能不对！文件的复制时，由于是连续的操作，所以没出现问题！（同时，一个一个字节的读取，写入，频繁的进行系统调用，在申请调用上太费时了。）

- 小例子

```java
@Test
public void fn7() throws Exception{
    Properties properties = System.getProperties();
    Enumeration<?> enumeration = properties.propertyNames();
    while (enumeration.hasMoreElements()){
        String o = (String) enumeration.nextElement();
        System.out.println(o);
    }
    //得到当前系统的默认编码格式 得到的是UTF-8
    System.out.println(System.getProperty("file.encoding"));

}

@Test
public void fn8() throws UnsupportedEncodingException {
    String str = "详细信息显示";
    //using the platform's default charset
    byte[] bytes = str.getBytes();
    byte[] bytess = str.getBytes("GBK");
    // U8
    for (int i = 0; i <bytes.length ; i++) {
        System.out.print(bytes[i]);
    }
    System.out.println("\r\n"+"*****************");
    // GBK
    for (int i = 0; i <bytess.length ; i++) {
        System.out.print(bytess[i]);
    }
}
```

> **字符流 = 字节流 + 编码表**

> 用字节流复制文本文件时，文本文件的中文没有问题。原因是最终底层操作会自动进行字节拼接成中文，如何识别中文呢？

> 汉字在存储时，无论时那种编码存储，第一个字节都是负数！

```java
// 代码验证
public void fn9() throws Exception{
    FileInputStream fis = new FileInputStream("E:\\Eclipse_javaee_workspace\\JavaSE\\JavaDay08( IO )\\src\\test.txt");
    int fisRead = fis.read();
    fis.close();
    System.out.println((char)fisRead+":"+fisRead); // 乱码 ä:228

    System.out.println("**************");

    InputStreamReader reader = new InputStreamReader(new FileInputStream("E:\\Eclipse_javaee_workspace\\JavaSE\\JavaDay08( IO )\\src\\test.txt"));
    int readerRead = reader.read();
    System.out.println((char)readerRead+":"+readerRead);// 不乱码 中:20013

    reader.close();
    int i = 20013; //
    System.out.println((char)i);// 输出“中”
}
```

- 汉字的码值很大！字节流的返回值在-1到255直接，无法正确识别大的数值。

#### 15.2.1 字符流的输出（写入文本文件）

```java
public void fn1() throws IOException {
    OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream("charTest.txt"));
    osw.write(97);
    /**
     * @param  cbuf  Buffer of characters
     * @param  off   Offset from which to start writing characters
     * @param  len   Number of characters to write 写入的数据的数目 写len个
     * 其他的大同小异 不赘述
    */
    char[] ch = {'a','b','c','d','e','f','h','h','1','2'};
    osw.write(ch,5,2);
    osw.flush();
}
```

```java
@Test
public void fn9() throws Exception{

    String absolutePath = new File(".").getAbsolutePath();
    System.out.println(absolutePath);//
    /**
     * E:\Eclipse_javaee_workspace\JavaSE\JavaDay08(IO)\. 打印当前文件的路径。
     * 如果用 new FileInputStream("test.txt") 他是从    E:\Eclipse_javaee_workspace\JavaSE\JavaDay08(IO)\.这里找！
     * 而test.txt实际在E:\Eclipse_javaee_workspace\JavaSE\JavaDay08(IO)\src\test.txt
     * 路径不一致，所以找部分指定文件！
    */
    FileInputStream fis = new FileInputStream("E:\\Eclipse_javaee_workspace\\JavaSE\\JavaDay08(IO)\\src\\test.txt");
    int fisRead = fis.read();
    fis.close();
    System.out.println((char)fisRead+":"+fisRead); // 乱码 ä:228

    System.out.println("**************");

    InputStreamReader reader = new InputStreamReader(new FileInputStream("E:\\Eclipse_javaee_workspace\\JavaSE\\JavaDay08(IO)\\src\\test.txt"));
    int readerRead = reader.read();
    System.out.println((char)readerRead+":"+readerRead);// 不乱码 中:20013
}
```

> **获取src下的文件请用类加载器进行加载！**

#### 15.2.2 字符流的输入（读取到内存）

```java
public void fn3() throws IOException{
    FileReader r = new FileReader("charTest.txt");
    int read = r.read();
    System.out.println(read);
}
```

#### 15.3 字符缓冲流的使用

> **与字节缓冲流类似，也是用到了装饰模式，且内部有一个8192大小的数组（不过是char数组）**

```java
@Test
public void fn4() throws Exception {
    // 读文本到内存中
    BufferedWriter bw = new BufferedWriter(new FileWriter("bw.txt"));
    bw.write("hwllo woafasdfs");
    bw.newLine();
    bw.write("asfhashfasfhoihasff");
    bw.newLine();
    bw.flush();
    BufferedReader br = new BufferedReader(new FileReader("bw.txt"));
    System.out.println(br.readLine());
    bw.close();
    br.close();
}
```

### 15.3 File类概述

#### 15.3.1 概述

> `java.io.File` 类是文件和目录路径名的抽象表示，主要用于文件和目录的创建、查找和删除等操作。 

**注意：**

**一个点 . 表示当前目录**

**两个点  .. 表示上一级目录**

```java
File fil = new File(".");
System.out.println(fil.isDirectory() + ":"+fil.getAbsolutePath());
File file = new File("..");
System.out.println(file.isDirectory()+":"+file.getAbsolutePath());
```

- 要求
  - 梳理File的基本操作
  - 遍历指定目录的所有文件（单级）
  - 遍历指定目录的所有文件（多级）
  - 复制单级目录文件
  - 复制多级目录文件
  - `JDK7`的异常处理
- File的基本操作

```java
/**
* File概述 及其基本操作
* Java文件类以抽象的方式代表文件名和目录路径名。该类主要用于文件和目录的创建、文件的查找和文件的删除等。
* File对象代表磁盘中实际存在的文件和目录。OS中文件和目录似乎是一个性质。Linux中将目录看作一种特殊的文件
*      回忆FCB 及其处理策略（OS）
*      回忆文件的存储方式（OS）
*/
public void fn1() throws IOException {
    // 与IO流一致，默认为相对路径。
    File file= new File("file.txt");
    if(!file.exists()) file.createNewFile();
    
    // E:\Eclipse_javaee_workspace\JavaSE\JavaDay08( IO )\file.txt
    System.out.println(file.getAbsolutePath()); 
    System.out.println(file.isAbsolute()); // false
    System.out.println(file.isDirectory()); // false
    System.out.println(file.isFile());  // true
    System.out.println(file.toString());// file.txt
}
```

- 遍历指定目录的所有文件（单级别目录）

```java
/**
 * 遍历单级文件夹下的所有文件
 */
public void fn3() {
    File file = new File("");
    // E:\Eclipse_javaee_workspace\JavaSE\JavaDay08( IO )
    System.out.println(file.getAbsolutePath());
    System.out.println(file.isDirectory());// false
    File file2 = new File(".");
    // E:\Eclipse_javaee_workspace\JavaSE\JavaDay08( IO )\.
    System.out.println(file2.getAbsolutePath());
    System.out.println(file2.isDirectory());// true
    System.out.println("**************************");
    String[] list = file2.list();
    for (String str : list) {
        System.out.println(str);
    }
}
```

- 遍历指定目录的所有文件（多级）

```java
/**
* 遍历指定文件夹下的所有文件。仅输出文件名称+文件绝对路径
* 递归
* 遇到目录就继续访问
* 遇到文件就打印输出
* 递归的判断条件是是否为目录
*/
@Test
public void fn4() {
    // 获得单曲目录 即项目名的目录 xxx\JavaSE\JavaDay08( IO )\.
    File file = new File(".");
    getAllFile(file);
}

public void getAllFile(File file) {
    if(file == null) return;
    File[] files = file.listFiles();
    if(files == null) return;
    for (File tempFile : files) {
        // 不存在传入null
        if (tempFile.isDirectory()) {
            getAllFile(tempFile);
        } else {
            System.out.println("fileName = " + tempFile.getName());
        }
    }
}
```

- 复制单级目录文件
  - 找到源文件
  - 找到目的地，目的地不存在则创建

```java
public void fn5() {
    File file = new File(".");
    File dest = new File("E:\\copyTemp");
    if(!dest.exists()) dest.mkdirs();
    File[] files = file.listFiles();
    for (File tempFile : files) {
        if (!tempFile.isDirectory()) {
            // 执行复制操作
            copyFile(new File(file,tempFile.getName()), new File(dest, tempFile.getName()));
        }
    }
}

/**
 * @param src  源文件
 * @param dest 目的文件
 */
public void copyFile(File src, File dest) {
    try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(src));
         BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dest));) {
        int b = 0;
        while ((b = bis.read()) != -1) {
            bos.write(b);
            bos.flush();
        }
        bis.close();
        bos.close();
    } catch (Exception e) {
        e.printStackTrace();
    }
}
```

- 复制多级目录
  - 递归遍历源文件夹
  - 当遇到文件时则进行复制
  - 遇到文件夹时则继续遍历

```java
/**
* 1.遍历源文件。
*      遍历过程中，如果遇到的是文件夹，则在dest创建对应的文件夹
*      遇到的是文件，则在dest创建对应的文件。
*      注意路径的保存
*/
public void fn7() throws Exception {
    File file = new File(".");
    File file1 = new File("E://copy2");
    if(!file1.exists()) file1.mkdirs();
    copy(file,file1);
}

public void copy(File src, File dest) throws Exception {
    File[] files = src.listFiles();
    if (files == null) return;
    for (File temp : files) {
        String curName = temp.getName();
        if (temp.isDirectory()) {
            // 如果是目录 则创建 创建后递归遍历
            File file = copyDirectory(dest, curName);
            copy(new File(src,curName),file);
        } else {
            // 是文件则复制文件，该层递归结束
            copyFile(new File(src, curName), new File(dest, curName));
        }
    }
}

private File copyDirectory(File dest, String curName) {
    File file = new File(dest, curName);
    if(!file.exists()) file.mkdir();
    return file;
}
```

- `jdk7`的异常处理

```java
public void copyFile(File src, File dest) {
    try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(src));
         BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dest));) {
        int b = 0;
        while ((b = bis.read()) != -1) {
            bos.write(b);
            bos.flush();
        }
        bis.close();
        bos.close();
    } catch (Exception e) {
        e.printStackTrace();
    }
}
```

### 15.4 标准输入输出流

#### 15.4.1 标准输入流

```java
InputStream in = System.in;

/**
* 模拟Scanner读入一个char 读入String
* @throws IOException
*/
public static void fn3() throws IOException {
    // 只能安全键盘录入字节
    InputStream in = System.in;
    // 转换流 装饰模式
    InputStreamReader isr = new InputStreamReader(in);
    // -1 到 0xFFFF
    int read = isr.read();
    // 可以安全地读入一个中文
    System.out.println((char)read);
    // 读一个串地话，自己设置char数组
}
```

#### 15.4.2 标准输出流

> **客户端的输入内容，直接写入文本？？！！重定向牛批**

```java
PrintStream out = System.out;

/**
* 标准输入流的重定向
* 从键盘输入 打印到文本中
*/
public static void fn2() throws FileNotFoundException {
    PrintStream printStream = new PrintStream("target.txt");
    System.setOut(printStream);
    String str = "999";
    while(!str.equals("exit")){
        Scanner sc = new Scanner(System.in);
        str = sc.next();
        System.out.println(str);
    }
}
```

### 15.4 打印流====用于写入数字，写入对象哈希值什么的

#### 15.4.1 字节打印流

```java
PrintStream
@Test
public void fn6() throws FileNotFoundException {
    PrintStream ps = new PrintStream(new FileOutputStream("xxx.txt"),true);
    ps.println(99);
    ps.close();
}
```

#### 15.4.2 字符打印流

```java
PrintWriter
/**
* 字符打印流
*/
@Test
public void fn4() throws IOException {
    PrintWriter printWriter = new PrintWriter(new FileWriter("a.x"));
    printWriter.write(99);
    printWriter.close();
    // 不刷新看一看
}

@Test
public void fn5() throws IOException {
    // true 调用 print println时自动刷新 而且写入的时99 不进行转型（char）99 也可以写入对象？
    PrintWriter printWriter = new PrintWriter(new FileWriter("a.txt"),true);
    // 写入数字
    printWriter.println(99);
    // 写入对象的哈希值
    printWriter.println(new Object());
}
```

### 15.5 其他流对象

#### 15.5.1 对象序列化

- 用于将不常用的又不能销毁的对象存入文本，要用时在从文本读取。可以节约内存？
- 类想要被序列化需要实现**`Serializable`**接口
- 类的个别字段不想被序列化的话使用**transient**关键字
- 若因为类进行了更改导致反序列化失败，如何解决？
  - 定义一个`private static final long serialVersionUID = -6849794470754660L;`进行是否是同一个类的判断
  - 无责任猜测：应该是计算了类的信息指纹，用信息指纹的比较来判断是否是同一个类。【密码学】

```java
@Test
public void fn1() throws IOException, ClassNotFoundException {
    // 测试序列化流的基本方法
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("objectDemo.txt"));
    oos.writeObject(new Student("hello1",52));
    oos.writeObject(new Student("hello2",52));
    oos.writeObject(new Student("hello3",52));
    oos.flush();
    oos.close();

    // 读取序列化对象
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("objectDemo.txt"));
    Student o = (Student) ois.readObject();
    System.out.println(o.toString());
    ois.close();
}

// 如果对象被更改了，能否再次正确读出？ 无法正确读出！！
// 怎么办？ 使用private static final long serialVersionUID = -6849794470754660L; 标识是否是同一个对象
// 不仅识别了，多余的方法还可以调用
@Test
public void fn2() throws IOException, ClassNotFoundException {
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("objectDemo.txt"));
    Student o = (Student) ois.readObject();
    Object o1 = ois.readObject();
    Object o2 = ois.readObject();java
        // 应该报错吧
        Object o3 = ois.readObject();
    o.say();
    System.out.println(o.toString());
    ois.close();
}

class Student implements java.io.Serializable {
    private static final long serialVersionUID = -6849794470754660L;
    // 不想被序列化的字段用transient
    private transient String name;
    private int age;
    private int weight = 10;
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public Student(){

    }

    public Student(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void say(){
        System.out.println("我是多余的方法！");
    }

    @Override
    public String toString() {
        return "Student{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
```

#### 15.5.2 Properties与IO流的结合

- 用于配置文件，防止硬编码

```java
/**
 * Properties与IO流的结合使用
 *  之前看他的方法 发现有传入IO对象的方法
 */
public class PropertiesDemo {
    @Test
    public void fn1() throws IOException {
        // 存入数据！  想一想数据库连接池的配置文件，就是这么个意思。防止硬编码。
        // 我真是个小天才
        Properties p = new Properties();
        p.setProperty("name","xiaoming");
        p.setProperty("jdbc","xxx");
        p.store(new FileOutputStream("PropertiesDemo.properties"),"无备注");

        p.load(new FileInputStream("PropertiesDemo.properties"));
        String name = p.getProperty("name");
        System.out.println(name);
    }
}
```

## 第十六章 线程

> **进程与线程的概念，看操作系统课本！看以前做的笔记，画的思维导图。**

- 多线程的实现方式

  - 继承Thread类
  - 实现Runnable接口

- 线程调度模型

  - 分时调度模型：所有线程轮流使用CPU使用权，平均分配每个线程占用CPU的时间（RR）
  - 抢占式调度模型：优先让优先级高的线程使用CPU，优先级相同则随机选取一个。优先级高的线程获取CPU的时间更多。
  - 我记得操作系统里有一个调度模型：**找书看一下 “多级反馈队列”** **汤子瀛 P94**

- 线程优先级的设置

  - `getPriority()`获得此线程的优先级
  - `setPriority()`更改此线程的优先级
  - 优先级高，仅仅代表获取`cpu`的几率高！回忆OS防止进程饥饿！

- 线程控制

  - `sleep(long millis)`：使当前正在执行的线程暂停`millis`毫秒
  - join()：等待这个线程死亡。`A.join(),`只有A这个线程执行完毕后，后面的代码/线程才会执行
  - **`setDaemon(boolean on)`：将线程标记为守护线程，当运行的线程为守护线程是，Java虚拟机将退出【其他线程执行完毕后，如果剩下的都是守护线程，则`jvm`不会等待守护线程执行完，会直接退出！】**

- 线程同步

  - 回忆OS的`pv`操作！
  - 线程同步案例
  - 线程同步的关键字synchronized锁，volatile保证数据可见但是不保证数据的准确性【回忆os磁盘的延迟写，cache的写回法什么的】

- 线程安全类 

  - `StringBuffer`【字符串的拼接，编译器会自动优化！我测试过！】
  - Vector
  - `Hashtable`===键值都不能为null。【Properties也是线程安全的】

- Lock锁 `jdk5`提供

  - `ReentrantLock()`

- `ThreadLocal`===Java线程本地存储

### 16.1 线程的运行

方式一：继承Thread类

- 为什么要重写run方法？
  - 因为run方法是用来封装被线程执行的代码
- run方法和start方法的区别
  - run封装线程执行的代码，直接调用，相当于普通方法调用
  - start，启动线程，然后由`jvm`调用此线程的run方法

```java
public class ThreadDemo {
    public static void main(String[] args) {
        // 线程抢夺CPU权限，交替执行。回忆CPU是如何分配的？ FIFS SJF RR
        MyThread myThread1 = new MyThread();
        MyThread myThread2 = new MyThread();
        myThread1.start();
        myThread2.start();
    }
}
class MyThread extends Thread{
    @Override
    public void run() {
        for (int i = 0; i <100 ; i++) {
            System.out.println(this.getName()+":"+i);
        }
    }
}
```

方式二：实现Runnable接口

- 相比于继承Thread，实现Runnable接口的优势
  - 避免了Java单继承的局限性【多继承可以用内部类实现】
  - 适合多个相同程序的代码去处理同一个资源。【Thread用静态定义资源也可以】，把线程和程序的代码，数据，进行了有效分类，较好体现了面向对象的设计思想！
    - 数据，代码分离体现在哪里？？？

### 16.2 线程的控制

#### 16.2.1 join()

```java
public static void fn1(){
    MyThread m1 = new MyThread();
    MyThread m2 = new MyThread();
    MyThread m3 = new MyThread();
    m1.setName("AA");
    m2.setName("BB");
    m3.setName("CC");
    m1.start();
    m2.start();
    m3.start();
}
public static void fn2() throws InterruptedException {
    MyThread m1 = new MyThread();
    MyThread m2 = new MyThread();
    MyThread m3 = new MyThread();
    m1.setName("AA");
    m2.setName("BB");
    m3.setName("CC");
    m1.start();
    m1.join();
    System.out.println("hello  world");
    m2.start();
    m3.start();
}
    
class MyThread extends Thread {
    @Override
    public void run() {
        for (int i = 0; i < 10; i++) {
            try {
                Thread.sleep(5);
                System.out.println(this.getName() + ":" + this.getPriority());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
```

#### 16.2.2 daemon()

> 设置当前线程为守护线程！当只剩守护线程时，jvm会退出，不会等待守护线程执行完毕

```java
// 非守护线程全部执行后 守护线程不一定能执行完毕，可能会被jvm直接终止
public static void fn1() {
    MyThread m1 = new MyThread();// for 10
    MyThread2 m2 = new MyThread2();// for 100
    MyThread2 m3 = new MyThread2();// for 100
    m1.setName("大哥");
    m2.setName("守护大哥一号");
    m3.setName("守护大哥二号");
    m2.setDaemon(true);
    m3.setDaemon(true);
    m1.start();
    m2.start();
    m3.start();
}
```

### 16.3 线程的同步

#### 16.3.1 使用synchronized

- synchronize（）中的应该就是充当信号量的。

```java
public static void main(String[] args) {
    SaleTicket sale = new SaleTicket();
    Thread t1 = new Thread(sale, "窗口一");
    Thread t2 = new Thread(sale, "窗口二");
    Thread t3 = new Thread(sale, "窗口三");
    t1.start();
    t2.start();
    t3.start();
}

public class SaleTicket implements Runnable {
    private int ticket = 100;
    private Object o = new Object();

    @Override
    public void run() {
        /**
         * OS中所谓的管程
         * OS中pv操作心得：pv中包裹的不影响同步的代码尽可能地少，多了影响程序性能。
         * java多线程应该也是如此！
         * */
        synchronized (o){
            while (ticket > 0) {
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("出售了一张，还有" + (--ticket) + "张");
            }
        }
    }
}
```

#### 16.3.2 同步代码块与同步方法

```java
public synchronized void run() {} // 同步方法！ 看视频！
```

同步方法默认用this或者当前类class对象作为锁；
同步代码块可以选择以什么来加锁，比同步方法要更细颗粒度，我们可以选择只同步会发生同步问题的部分代码而不是整个方法；
同步方法使用关键字 synchronized修饰方法，而同步代码块主要是修饰需要进行同步的代码，用 synchronized（object）{代码内容}进行修饰；

----

### 16.4 Local锁

```java
public class SaleTicket implements Runnable {
    private int ticket = 100;
    private Lock lock = new ReentrantLock();

    @Override
    public void run() {
        // PV 细想-
        lock.lock();
        while (ticket > 0) {
            lock.lock();
            System.out.println("出售了一张，还有" + (--ticket) + "张");

        }
        lock.unlock();
    }
}
```

### 16.5 生产者 消费者

> **生产者生产，消费者消费。有同步有互斥。**

- empty 告诉生产者还可以放多少
- full 告诉消费者还可以拿多少
- 同时只能一个拿或一个放
- 不能拿/放则等待 用wait()
- 有东西了，可以拿了就notify() == 【应该是随机唤醒一个等待的线程，可以指定唤醒某个吗？】
- 明天写！ 吃饭了！

## 第十七章 网络编程

### 17.1 网络编程入门

- 网络编程
  - 在网络通信写一下，实现网络互连的不同计算机上，进行数据交换。
- 网络编程三要素
  - IP地址。为计算机指定标识（即IP地址）
  - 端口。用于区分一台计算机的不同程序。如：A程序网络通信就用xx端口。
  - 协议。网络通信需要遵从的通信规则【常见协议有 UDP协议和TCP协议】

### 17.2 IP地址

`IP地址分为 IPv4和IPv6`

- IPv4 地址4字节 4*8=32位
- IPv6 地址128位地址长度，每16字节一组，分成8组十六进制数

`常见命令【windows】`

- ipconfig
- ping IP地址

#### 17.2.1 InetAddress

> **IP地址的获取和操作，InetAddress表示Internet协议（IP）地址**

- getHostName() 获得主机名称
- getHostAddress() 返回文本显示中的IP地址字符串

```java
@Test
public void fn1() throws UnknownHostException {
    // 通过计算机名称得到InetAddress对象
    InetAddress byAddress = InetAddress.getByName("DESKTOP-R0ENAIP");
    // 获得主机地址
    String hostAddress = byAddress.getHostAddress();
    System.out.println(hostAddress);
    // 获得主机名称
    String hostName = byAddress.getHostName();
    System.out.println(hostName);
}
```

### 17.3 端口

- 端口：设备上应用程序的唯一标识
- 端口号：用两个字节表示的整数。范围是0到65535.其中0到1023被用于一些知名的网络服务和应用。普通程序建议使用1024以上的端口号，防止端口被占用启动失败！

### 17.4 协议（`UDP`）

> **协议：计算机网络中，连接和通信的规则称之为网络通信协议**

#### 17.4.1 `UDP`协议

- 用户数据报协议（User Datagram Protocol）
- `UDP`是无连接通信协议。数据传输时发送端和接收端不建立逻辑。回忆计组和计网的相关内容。发送和接受都不确认对面有没有人。`UDP`协议资源消耗小，通信效率高，常用于音频，视频和普通数据的传输！
- `UDP`是无连接性，不保证数据的完整性。传输重要数据不建议使用`UDP`.

#### 17.4.2 UDP通信原理

UDP协议是一种不可靠的网络协议，它在通信的两端各建立一个Socket对象，但是这两个Socket只是发送，接收数据的对象因此对于基于UDP协议的通信双方而已，没有所谓的客户端，服务器的概念。

#### 17.4.3 UDP发送，接收数据的步骤

> **先有接收端，再有发送端！**

**发送数据的步骤**

- 创建发送端的套接字对象（DatagramSocket）
- 创建数据，并把数据打包
- 调用DatagramSocket对象的发送方法
- 关闭发送端

```java
@Test
public void fn1() throws IOException {
    DatagramSocket ds = new DatagramSocket();
    byte[] bytes = "你好，我是xxx".getBytes();
    InetAddress byName = InetAddress.getByName("DESKTOP-R0ENAIP");
    DatagramPacket dp = new DatagramPacket(bytes,0,bytes.length,byName,8888);
    dp.setData(bytes,0,bytes.length);
    ds.send(dp);
    ds.close();
}
```

**接收数据的步骤**

- 创建接收端的Socket对象（DatagramSocke）
- 创建一个数据包，用于接收数据
- 调用DatagramSocke的方法接收数据
- 解析数据包，把数据在控制台显示

```java
public static void main(String[] args) throws Exception {
    DatagramSocket ds = new DatagramSocket();
    byte[] bytes = "你好，我是xxx".getBytes();
    InetAddress byName = InetAddress.getByName("DESKTOP-R0ENAIP");
    DatagramPacket dp = new DatagramPacket(bytes,0,bytes.length,byName,8888);
    dp.setData(bytes,0,bytes.length);
    ds.send(dp);
    ds.close();
}
```

PS : 不记得具体的xx，就点进源码去看构造方法上面的注释！！

### 17.5 协议（TCP）

#### 17.5.1 TCP通信原理

TCP协议是一种可靠的网络协议，它在通信的两端各建立一个Socket对象，从而在通信的两端形成网络虚拟链路，一旦建立了虚拟的网络链路，两端的程序就可以通过虚拟链路进行通信！

> **Java对基于TCP协议的网络提供了良好的封装，使用Socket对象来代表两端的通信端口，并通过Socket产生IO流进行网络通信**

#### 17.5.2 TCP发送数据

- 创建客户端的Socket对象（Socket）

  - ```java
    Socket socket = new Socket("192.168.1.106",8888);
    ```

- 获取输出流，写数据

  - ```java
    OutputStream os = socket.getOutputStream();
    os.write("TCP我来了".getBytes());
    ```

- 释放资源

  - ```java
    socket.close();
    ```

#### 17.5.3 TCP接收数据

- 创建服务器端的Socket对象**（ServerSocket）**
  - `ServerSocket(int port)`  指定端口即可
- 监听客户端连接，返回一个Socket对象
  - `Socket.accept();`
- 获取输入流，读数据，把数据显示在控制台
  - `InputStream getInputStream()`
- 释放资源
  - `void close()`
- TCP读数据的方法是阻塞式的
- 解决办法：自定义结束标记；使用`shutdownOutput（）`方法【推荐】

**第一版代码**

```java
public class ClientDemo {
    public static void main(String[] args) throws IOException {
        // 发送数据 内存向外 输出流
        Socket socket = new Socket("192.168.1.106",8888);
        OutputStream os = socket.getOutputStream();
        os.write("TCP我来了".getBytes());
        socket.close();
        // 有用有三次握手的确认，所以需要客户端 服务器端都开启才行
    }
}

public class ServerDemo {
    public static void main(String[] args) throws IOException {
        ServerSocket s = new ServerSocket(8888);
        Socket accept = s.accept();
        // 可以用xx流一次读一行！
        InputStream is = accept.getInputStream();
        byte[] bytes = new byte[4096];
        int read = is.read(bytes, 0, bytes.length);
        System.out.println(new String(bytes, 0, read));
        accept.close();
        s.close();
    }
}
```

**第二版代码**

```java
public class ClientDemo {
    public static void main(String[] args) throws IOException {
        // 发送数据 内存向外 输出流
        Socket socket = new Socket("192.168.1.106",8888);
        OutputStream os = socket.getOutputStream();
        os.write("TCP我来了".getBytes());
        socket.close();
        // 有用有三次握手的确认，所以需要客户端 服务器端都开启才行
    }
}

public class ServerDemo {
    public static void main(String[] args) throws IOException {
        ServerSocket s = new ServerSocket(8888);
        Socket accept = s.accept();
        // 构造方法中要传入一个Reader对象,带Reader后缀的都继承了Reader
        BufferedReader br = new BufferedReader(new InputStreamReader(accept.getInputStream()));
        String line;
        while((line=br.readLine())!=null){
            System.out.println(line);
        }
        accept.close();
        s.close();
    }
}
```

### 17.6 上传文件到服务器

- 客户端，读取文件，并逐一发送数据
- 服务器端，接收数据
- 由于网络传输，服务器端是一直在等待客户端的数据的，所以会一直等。当客户端数据传输完毕后，给出停止标记！

```java
// 客户端代码
public class ClientDemo {
    public static void main(String[] args) throws IOException {
        Socket socket = new Socket("192.168.1.106", 9999);
        OutputStream outputStream = socket.getOutputStream();
        FileInputStream fis = new FileInputStream("demo5.txt");
        byte[] bytes = new byte[2048];
        int len = 0;
        while ((len = fis.read(bytes, 0, bytes.length)) != -1) {
            outputStream.write(bytes, 0, len);
            outputStream.flush();
        }
        // 停止自身的Output 这个写了，对方才知道 不要一直获取了，可以跳槽while循环
        socket.shutdownOutput();

        System.out.println("文件上传完毕了！要通知服务器关闭了！");
        byte[] b = new byte[1024];
        int read = socket.getInputStream().read(b, 0, b.length);
        System.out.println(new String(b, 0, read));

        fis.close();
        outputStream.close();
        socket.close();
    }
}

// 服务器端代码
public class ServerDemo {
    public static void main(String[] args) throws IOException {
        ServerSocket server = new ServerSocket(9999);
        Socket accept = server.accept();
        InputStream inputStream = accept.getInputStream();
        FileOutputStream fos = new FileOutputStream("Server.txt");
        byte[] bytes = new byte[2048];
        int len;
        while ((len = inputStream.read(bytes, 0, bytes.length)) != -1) {
            fos.write(bytes, 0, len);
            fos.flush();
        }
        accept.shutdownInput();// 停止input
        System.out.println("文件接收完毕");
        accept.getOutputStream().write("完成了".getBytes());

        fos.close();
        inputStream.close();
        accept.close();
        server.close();
    }
}
```

## 第十八章 反射

### 18.0 类加载器前置知识概述

**类加载的时机**

类从被加载到虚拟机内存中开始，到卸载出内存为止，整个生命周期包括：

加载，验证，准备，解析，初始化，使用和卸载 七个阶段

####  **18.0.1 加载**

加载是类加载过程的一个阶段，加载阶段需要完成以下三件事情

- 通过一个类的全限定名来获取定义此类的二进制字节流
- 将这个字节流所代表的静态存储结构转化为方法区的运行时数据结构
- 在内存中生成一个代表这个类的java.lang.Class对象，作为方法区，这个类的各种数据的访问入口。

任何类被使用时，系统都会为之建立一个java.lang.Class对象

#### **18.0.2 连接**

- 验证阶段：用于检测被加载的类是否有正确的内部结构【符合JVM规范】。【不是必要阶段，可省略】

- 准备阶段：负责为类的类变量分配内存，设置默认初始化值。

  - 这时候进行的内存分配仅包含类变量（被static修饰的变量），不包括实例变量，实例变量将会在对象实例化时随对象一起分配在Java堆中。

  - 这里的初始值“通常情况”下时数据类型的零值

  - ```java
    public staatic int value = 123
    value在准备阶段过后 初始值为0，而非123
    ```

- 解析阶段：将常量池内的符号引用替换为直接引用

  - 符号引用（Symbolic References）：符号引用以一组符号来描述所引用的目标，符号引用可以是任何形式的字面量，只要使用时能无歧义的定位到目标即可。与虚拟机的内存布局无关
  - 直接引用（Direct References）：直接引用可以是直接指向目标的指针、相对偏移量或是一个能间接定位到目标的句柄。与虚拟机的内存布局有关。如果有直接引用了，那么目标一定在内存中！

#### 18.0.3 初始化

类加载过程的最后一步。到了初始化阶段，才开始执行类中定义的Java程序代码（或者是是字节码）

**对类变量进行初始化**

**类的初始化步骤**

- 假如类还未被加载和连接，则程序先加载并连接该类
- 假如该类的直接父类还未被初始化，则先初始化其直接父类
- 假如父类中有初始化语句，则系统依次执行这些初始化语句

**类的初始化时机**【视频中的】

- 创建类的实例
- 调用类的类方法
- 访问类或接口的类的变量，或者为该类变量赋值
- 使用反射方式来强制创建某个类或接口对应的java.lang.Class对象
- 初始化某个类的子类
- 直接使用java.exe命令运行某个主类

### 18.1 类加载器

#### **18.1.1 类加载器作用**

- 负责将.class文件加载到内存中，并为之生成对应的java.lang.Class对象

#### **18.1.2 JVM的类加载机制**

- 全盘负责：当一个类加载器负责加载某个Class时，该Class所依赖和引用的其他Class也将由该类加载器负责载入，除非显示使用另一个类加载器来载入
- 父类委托：当一个类加载器负责某个Class时，先让父类加载器试图加载该Class，只有在父类加载器无法加载该类时才尝试从自己的类路径中加载该类
- 缓存机制：保证所有加载过的Class都会被缓存，当程序需要使用某个Class对象时，类加载器先从缓存区中搜索该Class，只有当缓存中不存在该Class对象时，系统才会读取该类对应的二进制数据，并将其转换成Class对象，存储到缓存区。

#### **18.1.3 ClassLoader：**

- 负责加载类的对象

#### **18.1.4 Java运行时的内置类加载器**

- **Bootstrap class loader**：它是虚拟机的内置类加载器，通常表示为null，并且没用父
- **Platform class loader**：平台类加载器可以看到所有平台类，平台类包括由平台类加载器或其祖先定义的JavaSE平台API，其实现类和JDK特定的运行时类
- **System class loader**：也被称为应用程序类加载器，与平台类加载器不同，系统类加载器通常定义应用程序类路径，模块路径和JDK特定工具上的类
- 类加载器的继承关系：System的父加载器为Platform，而Platform的父加载器为Bootstrap

```java
@Test
public void fn1(){
    // 获得系统加载
    ClassLoader c = ClassLoader.getSystemClassLoader();
    System.out.println(c);//sun.misc.Launcher$AppClassLoader@18b4aac2

    //获得父类加载
    ClassLoader c2 = c.getParent();
    System.out.println(c2);//sun.misc.Launcher$ExtClassLoader@4a574795

    //获得父类加载
    ClassLoader c3 = c2.getParent();
    System.out.println(c3);// null
}
```

### 18.2 反射概述

​		Java的反射机制是指在运行时去获取一个类的变量和方法信息，然后通过获取到的信息来创建对象，从而调用方法的一种机制。由于这种动态性，可以极大的增强程序的灵活性，程序不用在编译期就完成确定，在运行期仍然可以扩展。

### 18.3 反射操作

#### 18.3.1 获取Class类的对象

要使用反射，先要获取该类的字节码文件对象

- 使用类的class属性获取Class对象
- 调用对象的getClass()方法，该方法是Object类中的方法，所有Java对象都可以调用该方法
- 使用Class类中的静态方法forName(String className) 传入的是完整包名路径

```java
@Test
public void getClazz() throws ClassNotFoundException {
    // 最方便
    Class<Student> c1 = Student.class;
    Class<Student> c2 = Student.class;
    System.out.println(c1 == c2); //true

    Student s = new Student();
    Class<? extends Student> c3 = s.getClass();
    System.out.println(c2 == c3); //true

    // 灵活 可以把xx写在配置文件中
    Class<?>  c4 = Class.forName("com.bbxx.demo1.Student");
    System.out.println(c3 == c4); //true
}
```

#### 18.3.2 获取构造方法

- **自行查看api【暴力访问时需要setAccessible（true）】**

```java
@Test
public void getConstructors() throws Exception {
    Class<Student> c1 = Student.class;
    // 获得指定的构造方法
    Constructor<Student> con1 = c1.getConstructor(String.class,String.class,int.class);
    // 创建对象
    Student student = con1.newInstance("xxx", "swx", 15);
    System.out.println(student);

    // 获得所有非私有构造方法
    Constructor<?>[] con2 = c1.getConstructors();
    for(Constructor c: con2 ){
        System.out.println(c.getParameterTypes().length);
    }

    // 暴力反射
    Constructor<Student> c3 = c1.getDeclaredConstructor(String.class);
    // 取消访问检查
    c3.setAccessible(true);
    Student s3 = c3.newInstance("xx");
    System.out.println(s3.getName());
}
```

#### 18.3.3  获取成员变量

|        方法名称         |               方法说明                |
| :---------------------: | :-----------------------------------: |
|     `getFields（）`     |   获得所有公共字段（public修饰的）    |
| `getDeclaredFields（）` | 获得所有字段（包括protected private） |
| `age.set(student,18);`  |    为student对象的age字段设置值18     |

```java
@Test
public void getFiled() throws Exception {
    Class<Student> stu = Student.class;
    // 获得所有公有字段。public修饰的
    Field[] fields = stu.getFields();
    for (Field f: fields) {
        System.out.println(f.getName());
    }
    System.out.println("**********");
    // 获得所有字段 包括 protected private
    Field[] declaredFields = stu.getDeclaredFields();
    for (Field f: declaredFields) {
        System.out.println(f.getName());
    }
    System.out.println("**********");
    // 给student对象的age字段赋值为18
    Student student = stu.newInstance();
    Field age = stu.getDeclaredField("age");
    age.setAccessible(true);
    age.set(student,18);
    System.out.println(student.getAge());
}
```

#### 18.3.4 获取成员方法

| 方法名                                                       | 说明                                        |
| :----------------------------------------------------------- | :------------------------------------------ |
| `Method[] getMethods()`                                      | 返回所有公共成员方法对象的数组，包 括继承的 |
| `Method[] getDeclaredMethods()`                              | 返回所有成员方法对象的数组，不包括 继承的   |
| `Method getMethod(String name, Class<?>... parameterTypes)`  | 返回单个公共成员方法对象                    |
| `Method getDeclaredMethod(String name, Class<?>... parameterTypes)` | 返回单个成员方法对象                        |

#### 18.3.5 反射越过泛型检查

```java
@Test
public void refelectDemo() throws Exception {
    ArrayList<Integer> list = new ArrayList<Integer>();
    // list.add("123"); 抱错，有泛型检查
    Class<? extends ArrayList> clazz = list.getClass();
    // 是Object.class
    Method add = clazz.getMethod("add", Object.class);
    add.invoke(list,"asdf");
    System.out.println(list.get(0));
}
```



## 第十九章 函数式编程

### 19.1 体验Lambda表达式

### 19.2 Lambda表达式的标准格式

- Lambda表达式的代码分析
  - () 里面没用内容，可以看成是方法形式参数为空！
  - **->** 用箭头指向后面要做的事情！
  - { } 包含一段代码，称之为代码块，可以看成是方法体中的内容！

- Lambda表达式的格式
  - 格式：（形式参数）-> （代码）。
  - 形式参数：如果有多个参数，参数之间用逗号隔开；如果没用参数，留空即可。
  - **->** ：由英文中画线和大于符号组成，固定写法，代表指向动作。
  - 代码块：是我们具体要做的事情，也就是以前我们写法的方法体内容。

### 19.3 Lambda表达式的练习

- 使用前提

  - 有一个接口
  - 接口中有且仅有一个抽象方法

- 练习一

  - 定义一个接口Eatable，里面定义一个抽象方法：void eat（）；
  - 定义一个测试类，测试类中提供两个方法
    - 一个是useEatable（Eatable e）
    - 一个是主方法，在主方法中调用useEatable

  ```java
  public interface Eatable {
      void eat();
  }
  
  public class EatableImpl implements Eatable{
      public void eat(){
          System.out.println("eat eat eat");
      }
  }
  
  public class EatableDemo {
      public static void main(String[] args) {
          // 常规方式
          EatableImpl eatable = new EatableImpl();
          useEatable(eatable);
          // 匿名内部类写法
          useEatable(new Eatable() {
              @Override
              public void eat() {
                  System.out.println("eat eat eat");
              }
          });
          /**
           * 记忆方法
           *  因为只有一个抽象方法，所以不用写方法名称！
           *  （）没用参数就空着，有就写！
           *  ->{}指向要执行的代码块
           */
          useEatable(() -> {
              System.out.println("xxxx");
          });
  
      }
      public static void useEatable(Eatable e) {
          e.eat();
      }
  }
  ```

- 练习二

  - 定义一个接口Flyable，里面定义一个抽象方法：void fly（String s）；
  - 定义一个测试类，测试类中提供两个方法
    - 一个是`useFlyable（Flyable f）`
    - 一个是主方法，在主方法中调用useFlyable

```java
public interface Flyable {
    void fly(String str);
}

public class FlyableImpl implements Flyable {
    public void fly(String str){
        System.out.println("fly fly fly");
    }
}

public class FlyableDemo {
    public static void main(String[] args) {
        useFlyable((String str) -> {
            System.out.println(s);
            System.out.println("Fly");
        },"123");
        /**
         * 输出
         * 风和日丽，晴空万里123
         * Fly
         */
    }
	// 风和日丽，晴空万里被赋值给了lambda表达式中的String str。两个参数的情形也是一样的 
    public static void useFlyable(Flyable e,String s) {
        e.fly("风和日丽，晴空万里");
    }
}
```

### 19.4 Lambda表达式的省略模式

**（）中的数据类型可以省略！但是有多个参数的情况下，不能只省略一个！要省略就都省略！**

```java
// （）中的数据类型可以省略！
useFlyable((s) -> {
    System.out.println(s);
    System.out.println("Fly");
},"123");
```

**（）中如果参数只有一个，那么小括号可以省略**

```java
useFlyable(s -> {
    System.out.println(s);
},"123");
```

**如果代码块的语句只有一条，可以省略大括号和分号**

```java
useFlyable(s -> System.out.println(s) ,"123");
```

如果有return，renturn也要省略掉

```java
userAddable((x,y) -> x+y);
```

### 19.5 Lambda 表达式的注意事项

- 使用lambda必须要有接口，并且接口中有且仅有一个抽象方法

- 必须有上下文环境，才能推导出Lambda对于的接口
  - 根据局部变量的赋值得知Lambda对应的接口：Runnable r = () -> System.out.println("Lambda表达式")；
  - 根据调用方法的参数得知Lambda对应的接口：new Thread( () - >System.out.println("Lambda表达式") ).start();

```java
public interface Inter {
    void eat();
}

public class InterDemo {
    public static void main(String[] args) {
        useInter(() -> System.out.println("hello world"));
        // 直接写没用上下文环境 直接生成本地变量是Runnable
        Runnable runnable = () -> {
            while(true)
                System.out.println("hello world 2");
        };
        new Thread(runnable).start();
    }

    public static void useInter(Inter i) {
        i.eat();
    }
}
```

### 19.6 Lambda表达式和你们内部类的区别

- 匿名内部类调用方法的形参可以是：类，抽象类，接口
- Lambda调用方法的形参必须是接口
- 匿名内部类编译时会生成一个单独的class文件。lambda表达式不会，它对应的字节码会在运行时动态生成。

## 第二十章 接口组成更新

#### **概述：接口的组成**

- 常量

  - ```
    public static final
    ```

- 抽象方法

  - ```
    public abstract
    ```

- 默认方法（jdk 8）

  - ```
     public default void walk(){ System.out.println("hello"); }
    ```

  -  **default可以不被重写。方便在不影响已有接口的情况下更新接口**

- 静态方法（jdk 8）

- 私有方法（jdk 9）

### 20.1 方法引用

在使用Lambda表达式的时候，我们实际上传递进去的代码就是一种解决方案：拿参数做操作

若在Lambda中所指的的操作方案，已经有地方存在相同的方案，这时候是没用必要再写重复逻辑的。

那如何使用已经存在的方案？

使用方法引用来使用已经存在的方案！

- 方法引用符号

```java
public interface Animal {
    void say(Object o);
}

public interface Printable {
    void printString(String s);
}

public class PrintableDemo {
    public static void main(String[] args) {
        // 常规写法
        userPrintable(s -> System.out.println(s));
        // 方法引用符 ，：：   其实方法应用是直接把参数传给了这个方法【print】
        userPrintable(System.out::print);
        // 可推导的就是可省略的
        useAnimal(System.out::print);
    }

    // 我要打印数据
    private static void userPrintable(Printable p) {
        p.printString("hehahhh");
    }

    private static void useAnimal(Animal a){
        a.say(new Object());
    }
}
```

- ##### 静态方法引用
  
  - 格式：类名：：静态方法
  
  - 示例：Integer::parseInt（）
  
- 练习
  - 定义一个接口（Converter），定义抽象方法int convert(String s)
  - 定义测试类

```java
public interface Convert {
    int convert(String s);
}

public class ConvertDemo {
    public static void main(String[] args) {
        useConver(Integer::parseInt,"105");
    }
	// lambda表达式被类方法替代的时候，它的形式参数全部传递给静态方法作为参数！
    public static void useConver(Convert c,String str){
        int convert = c.convert(str);
        System.out.println(convert*10);
    }
}
```

- **特定对象的实例方法引用**

  - 格式：对象：：成员方法
  - 示例：“HelloWorld”.toUpperCase

- ##### 类的任意对象的实例方法引用

  - 格式：类名：：成员方法

  - 示例：String::substring

  - 格式：类名：：成员方法
  - 示例：String::substring

- ##### 构造器引用

  - 格式：**类名::new** ，

  - 示例： () -> new ArrayList<String>() 等价于 ArrayList<String>::new，代码示例：

  - ```java
    Supplier<List<String>> supplier1= () -> new  ArrayList<String>();
    ```

### 20.1 函数式接口

有且仅有一个抽象方法的接口

```java
@FunctionalInterface
public interface MyInterface {
    void say();
    default void saa(){
        System.out.println("hello");
    }
}

// 可以不写注解，但是写注解可以检测到是否只有一个抽象方法，安全些。
// 如果接口时函数式接口，编译通过！
// 建议加上这个注解
```

#### 20.1.1 函数式接口作为返回值

**return 推到式就可以了**

#### 20.1.2 常用的函数式接口

- `Java8`在`java.util.function`包下预定义了大量的函数式接口

- **Supplier接口**

  - Supplier<T>：包含一个无参的方法
  - T get（）：获得结果
  - 该方法不需要参数，它会按照某种实现逻辑（由Lambda表达式实现）返回一个数据
  - Supplier<T>接口也被称为生产型接口，如果我们指定了接口的泛型式是什么类型，那么接口中的get（）方法就会产生什么类型的数据供我们使用！
  - **简单说来，他就是一个容易，用来存Lambda表达式生成的数据的。可用get方法得到这个生成的数据**

  ```java
  public class Student {
      private int age;
      public Student(){}
      public Student(int age){
          this.age = age;
      }
  }
  
  public class SupplierDemo {
      public static void main(String[] args) {
          String string = getString(() -> "lqx");// 生成String
          Integer integer = getInteger(() -> 20 + 50);// 生成int
          System.out.println(string);
          System.out.println(integer);
      }
  
      // 生成Supplier示例
      private static void fn2(){
          Supplier<Student> s1 = Student::new; // 生成 对象放进去
          Student student = s1.get(); // 获得这个对象
          System.out.println(student.toString());
      }
  
      // 返回integer
      private static Integer getInteger(Supplier<Integer> sup){
          return sup.get();
  
      }
      // 返回String
      private static String getString(Supplier<String> sup){
          return sup.get();
      }
  }
  
  ```

- **Consumer接口**

  - Consumer<T>：包含两个方法
  - void accept(T t)：对给的的参数执行此操作
  - default Consumer<T>andThen(Consumer after)：返回一个组合的Consumer，依次执行此操作，然后执行after操作
  - Consumer<T>接口也被称为消费型接口，它消费数据的数据类型由泛型指定
  - **大概意思就是，你定义消费规则。然后调用它的消费方法，他会按这个消费规则进行消费**

```java
public class ConsumerDemo {
    public static void main(String[] args) {
        Consumer<Integer> c = x -> {
            for (int i = 0; i < x; i++) {
                System.out.println(i);
            }
        };
        c.accept(10);
    }
}
```

- **Predicate接口**

  - Predicate 接口是一个谓词型接口，其实，这个就是一个类似于 bool 类型的判断的接口。

  - Predicate常用的四个方法

    - ```java
      boolean test(T t) 对给定参数进行逻辑判断，判断表达式由Lambda实现。
      ```

    - ```java
      default Predicate<T>negate(); 返回一个逻辑的否定， 对应逻辑非
      ```

    - ```java
      default Predicate<T>and(Predicate other) 返回一个组合逻辑判断，对应短路与
      ```

    - ```java
      default Predicate<T>or(Predicate other) 返回一个组合判断，对应短路或
      ```

  - **Predicate常用于判断参数是否满足xx条件**

```java
public class PredicateDemo {
    @Test
    public void fn1() {
        Predicate<String> pre = (s) ->  s .equals("hello") ;
        System.out.println(pre.test("ss"));
        System.out.println(pre.negate().test("hello"));
    }

    @Test
    public void fn2(){
        Predicate<String> pre1 = (s) ->  s .equals("hello") ;
        Predicate<String> pre2 = (s) ->  s .equals("hello") ;
        // test(参数) 这个参数是赋值给s的 pre1 和 pre2 是否都等于hello
        System.out.println(pre1.and(pre2).test("hello"));
    }

    @Test
    public void fn3(){
        Predicate<String> pre1 = (s) ->  s .equals("hello") ;
        Predicate<String> pre2 = (s) ->  s .equals("123") ;
        // 把hello 赋值给s pre1 和 pre2 是否满足他们的比较关系
        System.out.println(pre1.or(pre2).test("hello"));
    }
}

// 用lambda筛选数据
@Test
public void fn4() {
    String[] strArray = {"理解,30", "wul123i,20","wui,20"};
    ArrayList<String> strings = myFilter(strArray, s -> s.split(",")[0].length() >= 2, s -> Integer.parseInt(s.split(",")[1]) > 23);
    System.out.println("ss");
    for(String s : strings){
        System.out.println(s);
    }
}

public static ArrayList<String> myFilter(String[] strArray, Predicate<String> pre1, Predicate<String> pre2) {
    ArrayList<String> list = new ArrayList<>();
    for (String s : strArray) {
        if (pre1.and(pre2).test(s)) {
            list.add(s);
            System.out.println("asfs");
        }
    }
    return list;
}
```

- **Function接口**
  - Function 接口是一个功能型接口，是一个转换数据的作用。
  - Function 接口实现 `apply` 方法来做转换。

  - 常用方法
    - R apply（T t）将此函数应用于给定的参数
    - default<V> Function andThen(Function after) ：返回一个组合函数，首先将函数应用于输入，然后将after函数应用于结果

  ```java
  public class FunctionDemo {
  
      public static void main(String[] args) {
          convert("132", Integer::parseInt);
          convert("132", Integer::parseInt);
  
  
          // 直接使用 String是传入数据的类型，Integer是apply处理后返回的数据类型
          Function<String,Integer> fn = (s)->Integer.parseInt(s)*10;
          Integer apply = fn.apply("10");
          System.out.println(apply);
  
      }
      // 要求 把一个字符串转换为int类型并乘以10输出
      private static void convert(String s, Function<String,Integer> fun){
          Integer apply = fun.apply(s);
          System.out.println(apply*10);
      }
  }
  ```

### 20.3 体验Stream流

 ```java
public class StreamDemo {
    @Test
    public void fn1() {
        ArrayList<String> list = new ArrayList<>();
        list.add("张三");
        list.add("李三");
        list.add("兆三");
        list.add("科学三");
        list.add("学习三");
        // stream 筛选出符合条件的数据进行输出
        list.stream().filter((ss) -> ss.length() == 3).forEach(ss -> System.out.println(ss));
    }
}
 ```

### 20.4 Stream流的简单Demo

**Stream流的使用**

- 生成流

  - 通过数据源（集合，数组等）生成流
  - `list.stream();`

- 中间操作

  - 一个流后面可以跟0个或多个中间操作，其主要是打开流，然后返回一个新的流，交给下一个操作使用
  - `filter()`

- 终结操作

  - **一个流只能有一个终结操作，当这个操作执行后流会被关闭**【forEach是void，无返回值的意思】

  - ```java
    public class StreamDemo {
        @Test
        public void fn1() {
            ArrayList<String> list = new ArrayList<>();
            list.add("张三");
            list.add("李三");
            list.add("兆三");
            list.add("科学三");
            list.add("学习三");
            // stream 筛选出符合条件的数据进行输出
            list.stream().filter((ss) -> ss.length() == 3).forEach(ss -> System.out.println(ss));
            Stream<String> stringStream = list.stream().filter((ss) -> ss.length() == 3);
            Assert.assertNotNull(stringStream);// 通过测试，不为空
            stringStream.forEach(System.out::print);
            Assert.assertNotNull(stringStream);// 通过测试，不为空
        }
        
        @Test
        public void fn4() {
            ArrayList<String> arr = new ArrayList<>();
            arr.add("1111");
            arr.add("2222");
            arr.add("3333");
            arr.add("4444");
            arr.add("5555");
            arr.add("6666");
            arr.add("7777");
            Stream<String> limit = arr.stream().limit(5);
            Stream<String> skip = arr.stream().skip(3);
            Stream<String> concat = Stream.concat(limit, skip);
            // stream has already been operated upon or closed
            // concat.forEach(System.out::println);
            System.out.println("***********");
            concat.distinct().forEach(System.out::println);
        }
    }
    ```

### 20.5 Stream流的生成方式

#### **20.5.1 Stream流的常见生成方式**

- Collection体系的集合可以使用默认方法stream（）生成流
  - default Stream<E>() stream()
- Map体系的集合间接生成流
- 数组可以通过Stream接口的静态方法of(T ...value)生成流

```java
@Test
public void fn1() {
    // Collection的 直接生成流
    ArrayList<String> arr = new ArrayList<String>();
    Stream<String> arrStream = arr.stream();

    HashSet<String> set = new HashSet<>();
    Stream<String> setStream = set.stream();

    // Map体系间接的生成流
    HashMap<String, Integer> map = new HashMap<>();
    Stream<Map.Entry<String, Integer>> mapStream = map.entrySet().stream();
    mapStream.filter(s -> s.getKey().length() > 2).forEach(System.out::println);

    // 数组变为Stream流
    String[] str = {"12313", "asda"};
    Stream<String> strSteam1 = Stream.of(str);
    Stream<String> strSteam2 = Stream.of("123", "!231", "!!");
}
```

#### 20.5.2 中间流操作

- **filter**：过滤，满足条件的保留，不满足的不保留。传入的是Predicate

- **limit**：取前xx个元素

- **skip**：跳过前xx个元素

- **concat**：`concat(Steam a,Stream b)` 合并a，b两个流

- **distinct**：基于`hashCode（）`和`equals（）`去重

- **sorted**：**按照指的规则排序，无参数按照自然排序，有参数按照指的排序规则**

  - ```java
    sorted(Comparator<? super T> comparator)
    ```

- **mapToInt**：将xx转为intStream

  - ```java
    IntStream mapToInt(ToIntFunction<? super T> mapper);
    ```

- **forEach**：遍历元素

  - 属于终结方法

  - ```java
    void forEach(Consumer<? super T> action);
    ```

- **count**：返回元素总和
  
  - 属于终结方法

```java
// filter操作
@Test
public void fn2(){
    ArrayList<Object> arr = new ArrayList<Object>();
    arr.add(new Object());
    arr.add(new Object());
    arr.add("!@#123");
    //  Stream<T> filter(Predicate<? super T> predicate);
    // 进行布尔判断，为真就保存？
    arr.stream().filter(s->(s instanceof String)).forEach(System.out::println);
}
```

```java
// limit和skip操作
@Test
    public void fn3(){
        ArrayList<String> arr = new ArrayList<>();
        arr.add("qeasfsa");
        arr.add("456");
        arr.add("789");
        arr.add("/45662*");
        arr.add("/asfg*");
        arr.add("/阿斯弗*");
        arr.add("/撒法发*");
        // limit 取前三个元素
        arr.stream().limit(3).forEach(System.out::println);
        // skip 跳过前3个元素
        arr.stream().skip(3).forEach(System.out::println);
        System.out.println("**********");
        // 跳过两个，剩下的前三个输出
        arr.stream().skip(2).limit(3).forEach(System.out::println);
    }
```

```java
// concat 和 distinct
@Test
public void fn4() {
    ArrayList<String> arr = new ArrayList<>();
    arr.add("1111");
    arr.add("2222");
    arr.add("3333");
    arr.add("4444");
    arr.add("5555");
    arr.add("6666");
    arr.add("7777");java
        Stream<String> limit = arr.stream().limit(5);
    Stream<String> skip = arr.stream().skip(3);
    Stream<String> concat = Stream.concat(limit, skip);
    // stream has already been operated upon or closed
    // concat.forEach(System.out::println);
    System.out.println("***********");
    concat.distinct().forEach(System.out::println);
}
```

```java
// sorted
@Test
public void fn5() {
    ArrayList<String> arr = new ArrayList<>();
    arr.add("1111");
    arr.add("22122");
    arr.add("3332333");
    arr.add("4123444");
    arr.add("a5555");
    arr.add("6dsf666");
    arr.add("7ds777");
    //在测试一次
    // arr.stream().sorted().forEach(System.out::println);

    // 默认从小到大排序。返回正数代表大！
    arr.stream().sorted((s1, s2) -> s1.length() - s2.length()).forEach(System.out::println);
    System.out.println("***");
    // 反过来就是从大到小
    arr.stream().sorted((s1,s2)-> s2.length()-s1.length()).forEach(System.out::println);
}
```

```java
// mapToInt
@Test
public void fn6(){
    ArrayList<String> list = new ArrayList<>();
    list.add("123");
    list.add("12");
    list.add("32");
    list.add("45");
    IntStream intStream = list.stream().mapToInt(Integer::parseInt);
    int sum = intStream.sum();
    intStream.forEach(System.out::println);
    System.out.println(sum);
}
```

#### 20.5.3 Stream流的收集操作

对数据使用Stream流的方式进行操作后，把流中的数据收集到集合中

- R collect（Collector c）
- 参数是一个Collector接口
- 常用方法

```java
@Test
public void fn8() {
    ArrayList<String> list = new ArrayList<>();
    list.add("123");
    list.add("12");
    list.add("32");
    list.add("45");
    // 收集到List中
    Stream<String> stream = list.stream().filter(s -> s.length() > 3);
    List<String> collect = stream.collect(Collectors.toList());

    // 收集到Set中
    HashSet<String> set = new HashSet<>();
    set.add("!2312313");
    set.add("!13");
    set.add("!453");
    set.add("!231");
    set.add("!412");
    Stream<String> Setstream = set.stream().filter(s -> s.length() > 5);
    Setstream.collect(Collectors.toSet());

    // 收集到Map中
    // 定义一个字符串数组，每一个字符串数据由姓名数据和年龄数据组合而成
    String[] arra = {"xx,18", "ljw,20", "lh,23"};
    // 得到年龄大于等于20的
    Stream<String> age = Stream.of(arra).filter(s -> Integer.parseInt(s.split(",")[1]) > 19);
    // 姓名作为key 年龄作为value
    // age.collect(key,value);
    // 年龄大于等于20的被封装成了map
    Map<String, String> collect1 = age.collect(Collectors.toMap(s -> s.split(",")[0], s -> s.split(",")[1]));
    Stream<Map.Entry<String, String>> stream1 = collect1.entrySet().stream();
    stream1.forEach( s -> System.out.println(s.getKey() + ":" + s.getValue()) );
}
```

# 第三部分 加强

## 第一章 枚举

### 1.1 枚举的使用Demo

下面看一段骚气的代码

```java
public String judge(String str){
    if("AAA".equals(str)){
        return "AAA";
    }else if("BBB".equals(str)){
        return "BBB";
    }else if("CCC".equals(str)){
        return "CCC";
    }else if("DDD".equals(str)){
        return "DDD";
    }
}
```

- 条件一多 就要该源码【扩展性弱】，有没有解决办法
- 代码看起来不优雅，有没有解决办法

**枚举！**

> **第一版，用枚举替代if else**

```java
// 直接用枚举
enum RoleOperation1 {
    ADMIN_POWER,
    NORMAL_POWER,
    SUPER_POWER
}

// 因为有返回值 所以这样定义
enum RoleOperation2 {
    ADMIN_POWER() {
        @Override
        public String toString() {
            return "Admin power";
        }
    },
    NORMAL_POWER() {
        @Override
        public String toString() {
            return "Normal power";
        }
    },
    SUPER_POWER() {
        @Override
        public String toString() {
            return "Super power";
        }
    }
}

// 因为有统一的方法，所以用接口定义规则
interface Operation {
    String op();
}

//  漂亮的枚举代码，虽然看起来长，复杂，但是拓展性特别强！
// 下面就是见证奇迹的时刻，优雅地用枚举替代if else。
public enum RoleOperation implements Operation {
    ADMIN_POWER() {
        @Override
        public String op() {
            return "Admin power";
        }
    },
    NORMAL_POWER() {
        @Override
        public String op() {
            return "Normal power";
        }
    },
    SUPER_POWER() {
        @Override
        public String op() {
            return "Super power";
        }
    }
}
```

```java
public class Demo1 {
    // 如此优雅的代码！！
    // 还有用工厂模式 策略模式的。感觉都不如枚举来的优雅。
    public String judge(String role) {
        return RoleOperation.valueOf(role).op();
    }
}
```

### 1.2 枚举的常用方法

| values()    | 以数组形式返回枚举类型的所有成员 |
| ----------- | -------------------------------- |
| valueOf()   | 将普通字符串转换为枚举实例       |
| compareTo() | 比较两个枚举成员在定义时的顺序   |
| ordinal()   | 获取枚举成员的索引位置           |

```java
package org.example.enumeration;

import org.junit.jupiter.api.Test;

// 枚举中一些常用方法
public class SomeFunc {
    @Test
    public void func1() {
        Color[] values = Color.values();
        for (Color c : values) {
            System.out.println(c);
        }
    }

    @Test
    public void func2() {
        //  将普通字符串实例转换为枚举
        Color blue = Color.valueOf("BLUE");
        System.out.println(blue);
    }

    @Test
    public void func3() {
        System.out.println(Color.BLUE.ordinal());
    }


    /**
     *     public final int compareTo(E o) {
     *         Enum<?> other = (Enum<?>)o;
     *         Enum<E> self = this;
     *         if (self.getClass() != other.getClass() && // optimization
     *             self.getDeclaringClass() != other.getDeclaringClass())
     *             throw new ClassCastException();
     *         return self.ordinal - other.ordinal;
     *     }
     */
    @Test
    public void func4() {
        // RED 和 BLUE比较， RED小于BLUE 返回负数 ；equals返回0；大于返回 正数
        System.out.println(Color.RED.compareTo(Color.BLUE)); // -1
        System.out.println(Color.RED.compareTo(Color.GREEN));// -2
    }

    @Test
    public void func() {
        System.out.println(Color.RED);
        // output RED
    }

}

enum Color {
    RED, BLUE, GREEN
}
```

## 第二章 比较对象

**Comparator和Comparable**

Comparable接口/ Comparator接口

- Comparator  函数式接口 jdk1.8引入
- Comparable 普通接口

## 第三章 单元测试

### 3.1 单元测试的优点

保证的程序代码的正确性【语法上了逻辑上】。

### 3.2单元测试的使用

@Test

- @Before 无论Test是否出现异常，都会执行 【初始化资源】
- @After 无论Test是否出现异常，都会执行 【销毁资源】

```java
public class JunitDemo {
    private OutputStream outputStream;

    @Before
    public void init() throws FileNotFoundException {
        System.out.println("IO 流初始化完毕了");
        outputStream = new FileOutputStream("junit.txt");
    }

    @Test
    /**
     * 单元测试判断数据的正确性
     * 一般用Assert里面的方法
     */
    public void fn1(){
        // 断言不为null  不是null则成功
        Assert.assertNotNull(outputStream);
    }

    @After
    public void destory() throws IOException {
        System.out.println("IO 流关闭了");
        outputStream.close();
    }
}
```

### 3.3 单元测试原理

## 第四章 反射

### 4.1 反射概述

反射可以把类的各个组成部分封装为其他对象。

反射，Java的高级特性，流行的框架基本都是基于反射的思想写成的。

Java反射机制是在程序的运行过程中，对于任何一个类，都能够知道它的所有属性和方法；对于任意一个对象，都能够知道它的所有属性和方法，**<span style="color:green">这种动态获取信息以及动态调用对象方法的功能称为Java语言的反射机制。</span>**

Java反射机制主要提供了以下这几个功能：

- 在运行时判断任意一个对象所属的类
- 在运行时构造任意一个类的对象
- 在运行时判断任意一个类所有的成员变量和方法
- 在运行时调用任意一个对象的方法

### 4.2 反射的基本操作

#### 4.2.1 获取成员变量

- `File[] getFileds()` // **获得所有公有字段，包括继承的**
- `Filed getFiled(String name)` // 获取指定name的
- `Filed[] getDeclaredFileds() `// 获取该类自己声明的，包括私有
- `Filed[] getDeclaredFileds(String name)` // 获取指定名称的

#### 4.2.2 获取构造方法

- `Constructor<?>[] getConstructors()` // 获得所有公有构造器

- `Constructor<?> getConstructor(Class<?>...parameterTypes)` //获得指定参数的公有构造器
- `Constructor<?>[]getDeclaredConstructors()`// 获得所有私有构造器
- `Constructor<T>[]getDeclaredConstructors()`//  得指定参数的构造器【包括public~~~private 】

#### 4.2.3 获取成员方法

- `Method[] getMethods()` // **获得所有public修饰的方法，包括继承的**

- `Method getMethod(String name, Class<?>... parameterTypes)` // 获得指定名称和参数类型的public修饰的方法
- `Method[] getDeclaredMethods()` //获得所有的私有方法
- `Method getDeclaredMethod(String name, Class<?>... parameterTypes)` // 获得指定名称和参数类型的方法

#### 4.2.4 获取类名

- `String getName()` // 获得类全名`com.bbxx.junits.Son`

#### 4.2.5 几个重要的类

> **Class类**

每定义一个`java` `class` 实体都会产生一个Class对象。我们编写一个类，编译完成后，在生成的 `.class`文件中，就会产生一个Class对象，这个Class对象用于表示这个类的类型信息。Class中没有公共构造器，即Class对象不能被实例化。

> **Field类**

Field类提供类或接口中单独字段的信息，以及对单独字段的动态访问。

> **Method类**

```java
invoke(Object obj, Object... args)
```

> **`ClassLoader`类**

**<span style="color:green">ClassLoader类加载器！类加载器用来把类（class）装载进JVM的。ClassLoader使用的双亲委派模型来搜索加载类的，这个模型也就是双亲委派模型。</span>**

**`ClassLoader`的类继承图如下：**

<img src="D:\69546\Documents\pics\JavaStrengthen\classLoader.png" style="float:left">

### 4.3 动态代理

#### 4.3.1 作用

运行时，动态创建一组指定的接口的实现类对象！（在运行时，创建实现了指定的一组接口的对象）

动态代理对比其他方法增强方式

<img src="D:\69546\Documents\pics\JavaStrengthen\proxy.png" style="float:left">

#### 4.3.2 基本Demo

```java
interface A{    
}
interface B{
}
Object o = 方法(new Class[]{ A.class, B.class })
o 它实现了A和B两个接口！
```

```java
Object proxyObject = Proxy.newProxyInstance(ClassLoader classLoader, Class[] interfaces, InvocationHandler h);
```

- 方法的作用：动态创建实现了interfaces数组中所有指定接口的实现类对象！
- `ClassLoader`：类加载器！
    - 它是用来加载器的，把.class文件加载到内存，形成Class对象！
- `Class[ ] interfaces`：指定要实现的接口们。
- `InvocationHandler`：代理对象的所有方法（个别不执行，一般`nativate`方法不会执行，但是`hashCode`却会执行，好奇怪）都会调用`InvocationHadnler`的`invoke()`方法
- 动态代理的作用
    - 最终是学习`AOP`（面向切面编程），它与装饰者模式有点相似，它比装饰者模式更灵活（潜在含义，动态代理更难！）

**动态代理基本Demo**

```java
interface IBase {
    public void say();

    public void sleep();

    public String getName();
}
```

```java
public class Person implements IBase {
    public void say() {
        System.out.println("hello");
    }

    public void sleep() {
        System.out.println("sleep");
    }

    public String getName() {
        return "getName";
    }
}
```

```java
public class ProxyDemo1 {
    public static void main(String[] args) {

        Person person = new Person();
        // 获得类加载器
        ClassLoader classLoader = person.getClass().getClassLoader();
        // 获得被代理对象实现的接口
        Class[] interfaces = person.getClass().getInterfaces();
        // 实例化一个处理器 用于增强方法用的
        InvocationHandler h = new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                method.invoke(person, args);
                return null;
            }
        };
        IBase p = (IBase) Proxy.newProxyInstance(classLoader, interfaces, h);
        // 获得代理类的名称 com.sun.proxy.$Proxy0
        System.out.println(p.getClass().getName());
        p.say();
    }
}
```

#### 4.3.3 invoke解释

```java
public Object invoke(Object proxy, Method method, Object[] args)
```

**这个invoke什么时候被调用？**

- 在调用代理对象所实现接口中的方法时被调用！

**参数解释**

- `Object proxy`：当前对象，即代理对象！在调用谁的方法！
- `Method method`：当前被调用的方法（目标方法）
- `Object [ ] args`：实参
- 返回的是方法的返回值。

<img src="D:\69546\Documents\pics\JavaStrengthen\invoke_explain.png" style="float:left">

----

```java
public class ProxyDemo2 {
    public static void main(String[] args) {
        Person person = new Person();
        ClassLoader classLoader = person.getClass().getClassLoader();
        Class[] interfaces = person.getClass().getInterfaces();
        System.out.println(interfaces.length);
        InvocationHandler h = new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                Object retVal = method.invoke(person, args);
                 // 这个返回了，方法才有返回值 
                return retVal; 
            }
        };
        IBase p = (IBase) Proxy.newProxyInstance(classLoader, interfaces, h);
        p.say();
        // invoke返回null的话，这里的输出就是null
        System.out.println(p.getName());
    }
}
```

### 4.4 模拟`AOP`

`Spring AOP`，感受一下什么叫增强内容可变！

- `ProxyFactory` 代理工厂
- `IBeforeAdvice` 前置通知接口【方法执行前调用前置】
- `IAfterAdvice` 后置通知接口【方法执行后调用后置】
- `IWaiter` 服务员类接口
- `ManWaiterImple` 具体的服务员类【对他进行增强】

## 第五章 注解

**注解也叫元数据**。是一种代码级别的说明，JDK1.5引入的特性，与类，接口，枚举是在同一层次。**可声明在包，类，字段，方法，局部变量，方法参数等的前面**，对这些元素进行说明。

**作用分类：**

1.代码分析，通过代码里标识的元数据对代码进行分析【结合反射技术】

2.编译检查，通过代码里标识的元数据让编译器能够实现机泵的编译检查【Override】

3.编写文档，通过代码里标识的元数据生成文档【生成文档doc文档】

### 5.1 内置注解

- `Override`：检测被标注的是否继承自父类
- `Deprecated`：表示方法过时
- `SuppressWarnings`：压制警告
    - 一般传递参数all

### 5.2 自定义注解

> **元注解`public @interface annotationName{}`**

反编译发现，本质就是一个接口。

```java
import java.lang.annotation.Annotation;

public interface Annotation extends Annotation {
}
```

#### 5.2.1 属性的返回值

**基本数据类型	String	枚举	注解	以上类型的数组**

#### 5.2.2 赋值问题

设置默认值`String sex() default "1";`

使用注解，数组类型的赋值 `str={xx,xx,xx}`，若数组中只有一个，大括号可省略。回忆Spring中注解

```
* 基本数据类型
		* String
		* 枚举
		* 注解
		* 以上类型的数组
```

### 5.3 元注解

> **用于描述注解的注解**

`@Target`：描述注解的位置

- `ElementType`取值
    - TYPE：可以作用于类上
    - METHOD：可以作用于方法上
    - FIELD：可以作用于成员变量上

`@Retention`：描述注解是被保留的阶段

`@Retention(RetentionPolicy.RUNTIME)`：当前被描述的注解，会保留到class字节码文件中，并被`JVM`读取到

`@Documented`：描述注解是否被抽取到api文档中

`@Inherited`：描述注解是否被子类继承

### 5.4 注解的解析

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Pro {
    String className();
    String methodName();
}

@Pro(className = "com.bbxx.Demo1",methodName = "show1")
public class RefelectDemo {
    public static void main(String[] args) throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, IllegalAccessException, InstantiationException {
        /**
         * 解析注解
         */
        Class<RefelectDemo> refelectDemoClass = RefelectDemo.class;
        Pro annotation = refelectDemoClass.getAnnotation(Pro.class);
        String s = annotation.className();
        String s1 = annotation.methodName();
        Class<?> aClass = Class.forName(s);
        Object o = aClass.newInstance();
        Method declaredMethod = aClass.getDeclaredMethod(s1);
        declaredMethod.setAccessible(true);
        declaredMethod.invoke(o);
    }
}
```

## 第六章 类加载器

### 6.1 分类

`ClassLoad`分类

- 引导 类加载器----->负责加载类库 rt中的jar 【最高，Bootstrap】
- 扩展 类加载器----->负责加载扩展jar包  ext下的都是扩展jar
- 系统 类加载器----->应用下的类，包含开发人员写的类和三方jar包【最低】

`ClassLoad`有个双亲委派模型，会先问父   类加载器/上级类加载器，向上级委托，没有就自己加载，没找到就抛出`ClassNotFound`。永远不会出现类库中的类被系统加载器加载，应用下的类 被引导加载。

委托父加载器加载，父可以加载就让父加载。父无法加载时再自己加载。

- 可避免类的重复加载，父类加载器已经加载了该类时，就没必要子`ClassLoader`再加载一次了/
- 考虑到安全因素，`java`核心`api`中定义类型不会被随意替换。

### 6.2 类加载的顺序

```java
class MyApp{
    public static void main(String[]args){ // 系统加载
        // 也由系统加载
        A a = new A(); 
        // 也由系统加载 （从系统开始匹配，最终会委托上去， ...由引导加载）
        String s = new String();
    }
}

class String{ // 引导加载， String类，类库中的
    private Integer i;// 直接引导加载，毕竟无法委托了！
}
```

其实还得分线程，每个线程都有一个当前的类加载器来负责加载类。

### 6.3 流程

基础阶段 **了解**，中级阶段 **熟悉**，高级阶段，**不清楚**。

继承`ClassLoader`类完成自定义类加载器。自定义类加载器一般是为了加载网络上的类，class在网络中传输，为了安全，那么class需要加密，需要自定义类加载器来加载（对class做解密工作）

`ClassLoader`加载类都是通过==`loadClass()`==方法来完成的。`loadClass()`方法的工作流程如下：

- 调用==`findLoadedClass()`==方法查看该类是否已经被加载过了，如果该类没有加载过，那么这个方法返回null。
- 判断`findLoadedClass()`返回的是否为null,如果不是null那么直接返回，可避免同一个类被加载两次。
- 如果`findLoadedClass()`返回的是null, 那么就启动代理模式（委托机制），即调用上级的`loadClass()`方法，获取上级的方法是`getParent()`，当然上级可能还有上级，这个动作就一直向上走；（==双亲委派机制==，tomcat破坏了双亲委派模型）
- 如果`getParent().loadClass()`返回的不是null，这说明上级加载成功了，那么就加载结果；
- 如果上级返回的是null，说明需要自己出手，`loadClass()`方法会调用本类的`findClass()`方法来加载类
- 这说明我们只需要重写`ClassLoader`的`findClass()`方法，这就可以了！如果重写了`loadClass()`方法覆盖了代理模式！

我们要自定义一个类加载器，只需要继承`ClassLoader`类。然后重写它的`findClass()`方法即可。在`findClass()`中我们需要完成如下的工作！

- 找到class文件，把它加载到一个byte[]中
- 调用`defineClass()`方法，把byte[]传递给这个方法即可

### 6.4 自定义类加载器

>**文件类加载器**

```java
public class MyClassLoader extends ClassLoader {
    private String directory;

    public MyClassLoader(String _directory, ClassLoader paraent) {
        super(paraent);
        this.directory = _directory;
    }

    protected Class<?> findClass(String name) throws ClassNotFoundException {
        try {
            // 把类名转为目录
            String file = directory + File.separator + name.replace(".", File.separator) + ".class";
            // 构建输入流
            InputStream fis = new FileInputStream(file);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int len = -1;
            while ((len = fis.read(buf)) != -1) {
                baos.write(buf, 0, len);
            }
            byte[] byteArray = baos.toByteArray();
            fis.close();
            baos.close();

            return defineClass(name, byteArray, 0, byteArray.length);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

热部署，越过双亲委派，就是不用`loadClass` 用`findClass`

> **复杂例子**

```java
package org.example.classloader;

import java.io.File;
import java.io.FileInputStream;
import java.lang.reflect.Method;

/**
 * 类加载器学习
 * 注意maven中的单元测试只能写在 test下面！
 * 字节码文件请自己生成一个 然后调用对应的方法哦！！
 */
public class ClassLoaderDemo extends ClassLoader {

    // 类加载器的地盘，指明加载那个地方的class文件
    private String classpath;

    public ClassLoaderDemo() {
    }

    public ClassLoaderDemo(String classpath) {
        this.classpath = classpath;
    }

    public static void main(String[] args) throws Exception {
        ClassLoaderDemo classLoaderDemo = new ClassLoaderDemo();
        classLoaderDemo.fun2();
    }

    // 执行字节码的非静态方法
    public void fun1() throws Exception {
        ClassLoaderDemo classLoaderDemo = new ClassLoaderDemo("D:\\");
        Class<?> clazz = classLoaderDemo.loadClass("org.example.classloader.ClassLoaderTest");
        // loaderSay是一个非静态方法，需要一个实例调用
        Method loaderSay = clazz.getMethod("loaderSay");
        ClassLoaderTest o = (ClassLoaderTest) clazz.newInstance();
        // 非静态方法需要一个实例进行调用
        loaderSay.invoke(o);
    }


    // 执行字节码的静态方法
    public void fun2() throws Exception {
        ClassLoaderDemo classLoaderDemo = new ClassLoaderDemo("D:\\");
        Class<?> clazz = classLoaderDemo.loadClass("org.example.classloader.ClassLoaderTest");
        // loaderSay是一个非静态方法，需要一个实例调用
        Method loaderSay = clazz.getMethod("loaderStaticFunction");
        // 静态方法不用实例
        String result = (String) loaderSay.invoke(null);
        System.out.println(result);
    }


    // 重写这个方法即可
    @Override
    public Class<?> findClass(String name) throws ClassNotFoundException {
        try {
            // 自定义的方法，通过类名找到class文件，把文件加载到一个字节数组中
            byte[] datas = getClassData(name);
            if (datas == null) {
                throw new ClassNotFoundException("类没有找到：" + name);
            }
            return this.defineClass(name, datas, 0, datas.length);

        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            throw new ClassNotFoundException("类找不到:" + name);
        }
    }

    private byte[] getClassData(String name) {
        // 把名字换成文件夹的名字
        name = name.replace(".", "\\") + ".class";
        File classFile = new File(classpath, name);
        System.out.println(classFile.getAbsoluteFile());
        return readClassData(classFile);
    }

    private byte[] readClassData(File classFile) {
        if (!classFile.exists()) return null;
        byte[] bytes = null;
        try {
            FileInputStream fis = new FileInputStream(classFile);
            bytes = fis.readAllBytes();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bytes;
    }
}
```

### 6.5 Tomcat类加载器

tomcat提供了两种类加载器。

**第一种 服务器类加载器**

- ${CATALINA-HOME}\lib\，tomcat类加载器，它负责加载下面的类

**第二种 应用类加载器**

- ${CONTEXT}\WEB-INF\lib  
- ${CONTEXT}\WEB-INF\classes

**总结**

tomcat破坏了双亲委派模型

引导

扩展

系统

服务器类加载器：先自己动手，然后再去委托

应用类加载器：先自己动手，然后再去委托

<img src="D:\69546\Documents\pics\JavaStrengthen\tomcat_classLoader.png" style="float:left">

## 第七章 并发

### 7.1 注意

> **不要调用Thread类或Runnable对象的run方法**。直接调用run方法会在同一个线程中执行----不会启动新的线程。调用`Thread.start()`方法会创建一个执行run方法的新线程。

> **线程的六种状态**

- New：新建
- Runnable：可运行【可能在运行 或 准备运行】
- Blocked：【阻塞】
- Waiting：【等待】
- Timed waiting：【具有指定等待时间的等待线程的线程状态。一个线程处于定时等待状态，这是由于调用了以下方法中的一种，具有指定的正等待时间】
- Terminated：【终止】

```java
//Thread内部的枚举类
public enum State {
    NEW,
    RUNNABLE,
    BLOCKED,
    WAITING,
    TIMED_WAITING,
    TERMINATED;
}
```

> **lock和unlock，unlock要放在finally中，确保锁可以被释放。**

> **可重入锁，获得锁的方法（代码）可以调用持有相同锁的方法**

> **`ReentrantLock()`**

- 公平锁和非公平锁。
- 公平锁倾向于选择等待时间长的线程，这种策略可能严重影响性能。
- 一般选择非公平锁。

> <span style="color:green">**Condition，用`ReentrantLock()`的实例对象获得Condition对象**</span>

- `await()` 将该线程放在这个条件的等待集中，<span style="color:green">**并放弃锁！**</span>
- `singalAll()` 激活等待这个条件的所有线程，把他们从等待集中移出，让他们重新成为可运行的线程！
- `singal()` 从该条件的等待集中随机选取一个从等待集中移出，让他们重新成为可运行的线程！
- <span style="color:green">**用if做条件判断不合适，存在虚假唤醒的问题，用while。【`JDK`注释中有说明】**</span>

> **synchronized**

> **线程就是一个单独的资源类，没有任何附属的操作。**

> **线程局部变量 `ThreadLocal`**

- `ThreadLocal.withInitial()`为函数式编程提供的方法

**Unsafe类啊！**

## 第八章 网络编程

采用windows的`telent`工具作为客户端进行发起连接。

### 8.1 入门

> **Client**

```java
/**
 * 测试服务器连接
 */
public class SocketTest {

    public static void fun1() {
        // jdk 7 try catch用法
        try (var socket = new Socket("time-a.nist.gov", 13)) {
            var scanner = new Scanner(socket.getInputStream());
            while (scanner.hasNextLine()) {
                System.out.println(scanner.nextLine() + "==");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void fun2() throws UnknownHostException, UnsupportedEncodingException {
        String host = "www.bilibili.com";
        InetAddress[] localhosts = InetAddress.getAllByName(host);
        for (InetAddress tmp : localhosts) {
            System.out.println(tmp.getHostAddress());
            System.out.println(tmp);
        }
    }

    public static void fun3() throws IOException {
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress("time-a.nist.gov", 13), 10000);
        Scanner scanner = new Scanner(socket.getInputStream());
        // Scanner类不是很熟悉
        while (scanner.hasNextLine()) {
            System.out.println(scanner.nextLine());
        }
    }

    public static void main(String[] args) throws IOException {
        fun3();
    }
}
```

> **Server**

```java
public class EchoServer {
    /**
     * 服务器端的 inputStream 和 outPutStream
     * inPutStream 输入流，输入到Server
     * outPutStream 输出流，输出到client
     *
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(8189);
        Socket accept = serverSocket.accept();
        // 控制台读入数据
        Scanner in = new Scanner(accept.getInputStream(), StandardCharsets.UTF_8);
        // 输出 IO流还是不熟悉 类的组合太复杂了
        // PrintWriter out = new PrintWriter(new OutputStreamWriter(accept.getOutputStream(), StandardCharsets.UTF_8), true);
        OutputStreamWriter out = new OutputStreamWriter(accept.getOutputStream(), StandardCharsets.UTF_8);
        out.write("connected");
        out.flush();

        boolean done = false;
        while (!done && in.hasNextLine()) {
            // 控制台输入数据
            String line = in.nextLine();
            // 输出到客户端
            out.write("Echo:" + line);
            out.flush();
            if ("BYE".equals(line.trim())) done = true;
        }
    }
}
```

---

## 第九章 `Servlet3.0`

- 注解
- 文件上传
- 异步处理 需要 `asyncSupported=true`，有过滤器的话，过滤器也要设置`asyncSupported = true`

使用型特性就是在保护你的Java职业生涯。

### 9.1 注解替代`xml`

```java
@WebServlet("/index.do")
public class IndexServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setAttribute("data", fakeData());
        request.getRequestDispatcher("/demo.jsp").forward(request, response);
    }

    public ArrayList<User> fakeData() {
        ArrayList<User> users = new ArrayList<>();
        users.addAll(Arrays.asList(
                new User("111", "111"),
                new User("222", "222"),
                new User("333", "333")));
        users.forEach(System.out::println);
        return users;
    }
}
```

### 9.2 异步响应

异步响应如果不设置编码格式 可能会导致异步失败（有乱码，异步可能会失败；主要是告诉它响应文本是什么。）测试了一下，的确是设置好响应文本即可。

异步响应如果过滤器这些东西没有设置为异步状态，也会导致异步失败

```text
 * 类型 异常报告
 * 消息 当前链的筛选器或servlet不支持异步操作。
 * 描述 服务器遇到一个意外的情况，阻止它完成请求
 
 错误的原因就是过滤器没有设置  asyncSupported = true
```

**代码案例**

```java
@WebServlet(urlPatterns = "/async", asyncSupported = true)
public class AsyncServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doGet(request, response);
    }

    private char[] getOutPutChar(String str) {
        return str == null ? "   2020年 10月24日，祝各位程序员节日快乐！ 2020-1024=996，想不到吧！".toCharArray() : null;
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        // 不加设置响应的类型的话，就无法异步。
        response.setContentType("text/html");
        AsyncContext asyncContext = request.startAsync(request, response);
        threadOutPut(asyncContext, response, getOutPutChar(null));
    }

    /**
     * @param asyncContext
     * @param response
     * @param outputStr    需要输出给浏览器的数据
     */
    private void threadOutPut(AsyncContext asyncContext, HttpServletResponse response, char[] outputStr) {
        asyncContext.start(() -> {
            try {
                PrintWriter print = response.getWriter();
                TimeUnit.MILLISECONDS.sleep(600);
                for (char c : outputStr) {
                    TimeUnit.MILLISECONDS.sleep(180);
                    print.print(c); print.flush();
                }
                asyncContext.complete();
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                asyncContext.complete();
            }
        });
    }
}
```

### 9.3 文件上传

> **几个重要的API**

```java
- request.getPart("file_name") // 获得文件对象Part
- part.getName() // 获得文件上传时的 name <input name="xx"> 中的name
- part.getSize() // 获得文件的大小
- part.getSubmittedFileName() // 获得提交的文件的名字。上传的是 demo.txt 那么得到的就是 demo.txt
- part.getInputStream(); // 获得文件输入流。

获取文件输入流后，在用输出流 存入磁盘。
```

**文件上传的简单Demo**

文件上传用绝对路径【公司】

```java
@WebServlet("/upload")
@MultipartConfig // 表示它支持文件上传
public class FileUpload extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        Part part = request.getPart("file_name");
        System.out.println(part.getName());
        System.out.println(part.getSize());
        System.out.println(part.getSubmittedFileName());
        InputStream inputStream = part.getInputStream();
        // new FileOutputStream("filename") 这样是无法定位位置的，不能正常存储？
        //D:\citespace.projects.txt
        FileOutputStream fos = new FileOutputStream("D://" + part.getSubmittedFileName());
        // citespace.projects.txt
        // FileOutputStream fos = new FileOutputStream(part.getSubmittedFileName());
        byte[] bys = new byte[1024];
        int len = 0;
        while ((len = inputStream.read(bys)) != -1) {
            fos.write(bys, 0, len);
        }
        inputStream.close();
        fos.close();
    }
}
```

```html
<html>
<head>
    <title>Title</title>
</head>
<body>
    enctype 说明有文件要提交过去
<form action="/Tomcat/upload" method="post" enctype="multipart/form-data">
    <input type="file" name="file_name">
    <input type="submit">
</form>
</body>
</html>
```

# 第四部分 源码阅读

- 修改idea配置
    - Build，Execution，Deployment --> Debugger --> Stepping 的Do not step into the classes不要勾选

- 把jdk的源码，解压，然后放在项目的目录下，不要放在src下就行，非项目目录也可以，然后idea源码关联改成我们解压后的那些文件，这样就可以修改源码注释了。

----

<span style="color:red">**JDK8核心源码目录说明**</span>

- java.applet：applet所必须的类
- java.awt：创建用户界面和绘制图形图像的所有类
- java.beans：包含与开发javabeans有关的类

- <span style="color:red">**java.lang：**</span>Java基础类【类装载（Class等）】、反射、字符串、线程类
- java.math：主要是BigInteger、BigDecimal这两个的源码
- java.net：网络应用出现提供类（阻塞式）
- <span style="color:red">**java.nio：**</span>多路的、无阻塞的I/O
- java.rmi：远程方法调用相关
- java.security：安全框架类和接口
- java.sql：操作数据库，闲得无聊可以看看
- <span style="color:red">**java.util：**</span>重要的有：集合、JUC（Atomic、Lock、concurrent）、Stream（函数式操作）、工具类
- java.text：处理文本、日期、数字和消息的类和接口（常见DataFormat、SimpleDataFormat）
- java.time：日期时间工具库
- javax.java：java扩展包，为了保持版本兼容，但有了更好的解决方案。如swing
- launcher：和底层交互的c代码
- org：其他企业组织提供的java类库，大部分不是oracle公司提供的。如w3c提供的对xml解析类和接口
- com.sun：hotspot虚拟机中java.* 和javax.*的实现类，其他jdk中没有，这些类可能不向后兼容。

Java native方法是用c c++实现的，无法查看到代码~要看的话去下open jdk的源码

<span style="color:red">**用open jdk的源码。**</span>

- corba：不流行的多语言、分布式通讯接口
- hotspot：Java虚拟机
- hotspot-cpu：CPU相关代码（汇编器、模版解释器、部分runtime函数）
- hotspot-os：操作系统相关代码
- hotspot-os_cpu：操作系统+CPU的代码
- hotspot-share：平台无关的通用代码
- jaxp：xml处理
- jaxws：一组XML web services的Java API
- <span style="color:green">**jdk**</span>：Java开发工具包（share\class Java的实现 share\native目录里的是C++的实现）
- langtools：Java语言工具
- nashorn：JVM上的JavaScript运行时





