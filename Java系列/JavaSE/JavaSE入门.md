# 第一部分 入门

## 开发环境

### Java虚拟机

- JVM：Java 虚拟机。Java 代码是运行在虚拟机上的。
- 跨平台：代码运行在虚拟机上，不同版的 OS（linux，windows，mac）对应不同的虚拟机。虚拟机本身不具备跨平台功能，每个 OS 下都有不同版本的虚拟机。『可理解为，各个 OS 下的虚拟机都是采用一套编码指令集，JVM 是面向操作系统的，它负责把 Class 字节码解释成系统所能识别的指令并执行，同时也负责程序运行时内存的管理』

### JRE&JDK

- JRE (Java Runtime Environment）Java 程序的运行时环境，包含 JVM 和运行时所需要的核心类库
- JDK（Java Development Kit）Java 程序开发的工具包，包含 JRE 和开发人员使用的工具。
- 运行 Java 程序有 jre 就行，开发 Java 程序需要 JDK。
- Windows 会把 `%CC%` 中的 CC 当作变量进行翻译

## 入门程序

### 程序开发步骤说明

- 编写、编译、运行
- Java 源程序-->Java 字节码文件-->JVM 运行
- Javac.exe 编译器，将 Java 文件变成字节码文件
- Java.exe 解释器，解释字节码的内容

## 第四章 常量

- 常量；在程序运行期间固定不变的量。
- 常量的分类
  - 字符串常量：凡是用双引号引起来的部分叫字符串常量。“asdfas”，可以是空串
  - 整数常量：直接写上数字的，没有小数点。
  - 浮点数常量：直接写上数字的，有小数点。
  - 字符常量：用单引号引起来的 ‘A’，不能是空字符‘’。
  - 布尔常量：只有两种取值。true，false
  - 空常量：null，代表没有任何数据。不能直接用来打印。syso(null) 是错的。

## 变量&数据类型

### 数据类型

#### 基本数据类型

- 整数
  - byte   1 个字节
  - short  2 个字节
  - int      4 个字节
  - long   8 个字节
- 浮点数
  - float      4 个字节
  - double   8 个字节
- 字符型
  - char    2 个字节
- 布尔型
  - boolean   1 个字节

><b>Java 中默认类型：整型是 int，浮点类型是 double</b>
>想要精确的数字不推荐用 double，用 BigDemical。

#### 引用数据类型

> <b>字符串，数组，类，接口，Lambda</b>

注意事项：

- 字符串不是基本数据类型
- 浮点数可能是一个近似值
- 数据范围与字节数不一定相关。如 float 数据范围比 long 更大，但 float 是 4 字节，long 是 8 字节
- 浮点数默认是 double，如果一定要用 float，需要加上一个后缀 F(推荐大写)
- 如果是整数，默认为 int，如果一定要用 long，需要加上一个后缀 L(推荐大写)

### 变量

- 变量：程序运行期间，内容可以发生改变的量
- 编译原理，左值与右值

### 强制数据类型转换

- 强制类型转换一般不推荐使用，因为可能发生精度损失
- byte，short，char 这三种数据类型可发生数学运算。
- byte，short，char 在运算时都会被首先提升为 int 类型，然后再计算。

```java
byte num1 = 40;
byte num2 = 50;
//byte + byte --> int + int
int result = num1 + num2;
//如果用 byte接收 需要强转
byte result = (byte)(num1 + num2);
// short 同理
```

### ASCII码表

```
0 -- 48
A -- 65
a -- 97
```

### 数字和字符的对照关系表（编码表）

ASCII 码表：American Standard Code for Information Interchange

Unicode 码表：万国码。也是数字和符号对照关系，开头 0-127 部分和 ASCII 完全一样，但是从 128 开始包含更多字符。

### 易错点

byte short char 这些在计算的时候，会有类型提升，提升为 int 进行计算。

```java
public static void main(String[] args) {
    byte a = 8;
    byte b = 127;
    b = a+b; 
    // 会报错，提示你要进行强制类型转换。因为计算的时候，a和b会被提升为int类型，然后再进行计算，得到的结果也是int的，要把int类型的赋值给byte类型的变量需要进行强制类型转换。
}
```

包装类型的比较

```java
public class Demo {
    public static void main(String[] args) {
        Integer b = new Integer(47);
        Integer a = new Integer(47);
        System.out.println(a == b);  // false 因为 a b 是不同的对象。
        Integer c = Integer.valueOf(47);
        Integer d = Integer.valueOf(47);
        System.out.println(c == d); // true 因为 valueOf 创建对象是会先从 IntegerCache 缓存中找，有就返回缓存中的对象。
        							// IntegerCache 是静态内部类。静态内部类在你使用的时候才会进行加载。注意：是说的静态内部类。
    }
}
```

静态内部类加载时机的测试。遇到 new、getstatic、putstatic 或 invokestatic 这四条字节码指令执行的时候，如果类没有进行过初始化，则需要先触发其初始化。

```java
public class Demo {
    // VM参数 -XX:+TraceClassLoading
    public static void main(String[] args) {
        int a = 0x2f;
        // 调用静态内部类，控制台输出，它被加载了。不用静态内部类，它就不加载。
        System.out.printf("", TestClassLoading.d); 
    }

    static class TestClassLoading {
        static int d = 10;
        { System.out.println("d"); }
    }
}
```

## 常用运算

- 一元运算符：只需要一个数据就可以进行操作的运算符。

  - 取反
  - 自增
  - etc

- 二元运算符：需要两个数据才可以进行操作的运算符。

  - 加法
  - 减法
  - 赋值

- 三元运算符：需要三个数据才可以进行的运算符。

  - 变量名称 = 条件判断?表达式 A : 表达式 B;
  - int max = a > b ? a : b;

- 拓展

  - 对于 byte/short/char 三种类型来说，如果右侧赋值的数值没有超过范围，那么 java 编译器会自动隐含地为我们补上一个 (byte)(short)(char).

  ```
  short = 5 + 8;(都要是常量才行)
  等同于
  short = 13; // 编译优化
  先计算出的结果在进行赋值的
  称为编译器的常量优化。
  ```

## 基本语法

### switch & 循环

#### switch

> 基本语法

```java
public class Demo {
    public static void main(String[] args) {
        int a = 5;
        switch (a) {
            case 1:
                System.out.println(1);
            case 2:
                System.out.println(2);
            default:
                System.out.println("over!");
        }
    }
}
```

- 多个 case 后面的数值不可以重复
- switch 后面小括号中只能是下列数据类型
  - 基本数据类型 byte/short/char/int
  - 引用数据类型 String 字符串、enum 枚举

> Java 7 之前想用 String 判断的话

```java
package tij.chapter5;

public class StringSwitch {
    public static void main(String[] args) {
        String color = "red";
        // 老的方式: 使用 if-then 判断
        if ("red".equals(color)) {
            System.out.println("RED");
        } else if ("green".equals(color)) {
            System.out.println("GREEN");
        } else if ("blue".equals(color)) {
            System.out.println("BLUE");
        } else if ("yellow".equals(color)) {
            System.out.println("YELLOW");
        } else {
            System.out.println("Unknown");
        }
        // 新的方法: 字符串搭配 switch
        switch (color) {
            case "red":
                System.out.println("RED");
                break;
            case "green":
                System.out.println("GREEN");
                break;
            case "blue":
                System.out.println("BLUE");
                break;
            case "yellow":
                System.out.println("YELLOW");
                break;
            default:
                System.out.println("Unknown");
                break;
        }
    }
}
```

> switch 中可以用 String 的原理

调用字符串的 hashCode 方法，switch 中先比较 hashCode 的值，hashCode 的值一致再比较字符串的值。

```java
public class TestSwitch {
    public static void main(String[] args) {
        String str = "dd";
        switch (str) {
            case "dd":
                System.out.println("odk");
                break;
            case "cc":
            default:
                System.out.println("over!");
        }
    }
}
// 反编译后的代码
public class TestSwitch {
    public TestSwitch() {}

    public static void main(String[] args) {
        String str = "dd";
        byte var3 = -1;
        switch(str.hashCode()) {
            case 3168:
                if (str.equals("cc")) { var3 = 1; }
                break;
            case 3200:
                if (str.equals("dd")) { var3 = 0; }
        }

        switch(var3) {
            case 0:
                System.out.println("odk");
                break;
            case 1:
            default:
                System.out.println("over!");
        }
    }
}
```

#### 循环

for 循环，最常用的迭代形式

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

逗号操作符：在 for 循环的初始化和步进控制中定义多个变量。

```java
public class CommaOperator {
    public static void main(String[] args) {
        for(int i = 1, j = i + 10; i < 5; i++, j = i * 2) {
        	System.out.println("i = " + i + " j = " + j);
        }
    }
}
```

增强 for 循环 foreach：操纵数组和集合

```java
for(float x : f){
    System.out.println(x);
}
// 将每一个f的元素赋值给x
```

do-while

```java
do {
    // doing something
} while (condtion);
```

### break & continue

break 跳出一层循环，continue 开启下一次循环。IDEA 点击关键字可以看到下一步会执行到那里。

### goto

 ```java
 public class GoToDemo {
     public static void main(String[] args) {
         outer:
         for (int i = 0; i < 10; i++) {
             for (int j = 0; j < 10; j++) {
                 if (j == 5) {
                     System.out.println(j);
                     break outer;
                 }
             }
         }
     }
 }
 ```

### 重载与重写

<b>方法调用的三种格式</b>

1.单独调用：方法名称(参数)
2.打印调用：System.out.println(方法名称(参数))
3.赋值调用：数据类型 变量名称 = 方法名称(参数)

<b>方法重载 Overload</b>

- <b>方法重载：</b>指在同一个类中，允许存在一个以上的同名方法，只要它们的参数列表不同即可，与修饰符和返回值类型无关。
- 参数列表：个数不同，数据类型不同，顺序不同。
- 重载方法调用：JVM 通过方法的参数列表，调用不同的方法。

```java
// 以下参数顺序不一样也是重载！
public static void test(int a, short b){}
public static void test(short b,int a){}
```

- 实际上，println 就是一个被重载的函数

- 方法重写 Overrider 

  - 子类中出现和父类中一模一样的方法（包括返回值类型,方法名,参数列表）
  - 1.重写的方法必须要和父类一模一样（包括返回值类型,方法名,参数列表）
  - 2.重写的方法可以使用 @Override 注解来标识

#### 重载的注意事项

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

### 数组

#### 数组的初始化

- 动态初始化 -- 指定数组长度

```java
int [] array = new int[300];
```

- 静态初始化 -- 指定数组内容

```java
int [] array = new int[]{1,2,3,4,5,6}; // 标准格式
int [] array = {1,2,3,4,5,6}; // 省略格式
// 静态初始化不能拆分成
int [] array;
array = {1,2,34};
// 这样是错误得
```

- 总结

>动态初始化有默认值的过程，
>整型 默认为 0
>浮点 默认为 0.0
>字符 默认为 '\u0000'
>布尔 默认为 false
>引用 默认为 null
>
>静态初始化也有，不过系统自动马上将默认值替换为了大括号当中的具体数值。

#### 数组作为参数，返回值

```java
public static void cals(int[] arr){
    xxxx
}

public static int[] calculate(int a,int b){
    int [] array = {a,b};
    return array;
}
```

数组作为参数，作为返回值其实都是数组的地址值

### Java 内存划分

- 1.栈（stack）：存放的都是方法中的局部变量。方法的运行一定要在栈中运行
  - 局部变量：方法的参数，或者方法{}内部的变量
  - 作用域：一旦超出作用域，立刻从栈内存当中消失
- 2.堆（heap）：凡是 new 出来的东西都在堆中
  - 堆里面的数据都有默认值。默认值同上
- 3.方法区（method area）：存储 .class 相关信息，包含方法的信息。
- 4.本地方法栈（native method stack）：与操作系统相关
- 5.寄存器（register）：与 CPU 相关

Java 的垃圾回收，对于提高对象的创建速度，具有明显的效果。Java 从堆空间分配空间的速度，可以和其他语言从堆栈上分配空间的速度相媲美。在某些 Java VM 中，堆的实现截然不同，但是堆内存的分配可以看做：有一个堆指针，简单移动到尚未分配的区域，通过这种方式分配对象内存，其效率比得上 C++ 在栈上分配空间的效率。当然，在实际工作方面，还有少量额外的开销，但是比不上查找可用空间的开销。（<b>Java GC 会清理出可用的空间，堆指针在空间中移动，这样就完成了内存的分配。而 C++ 需要遍历查找可用的内存，这个查找开销较大。这样一对比，会发现，Java 分配对象的速度并不比 C++ 慢</b>）

Java 的 GC 工作的时候，一面回收内存空间，一面使堆中的对象紧凑排列。

Java 的优化技术==>JIT（Just-In-Time）：这种技术可以把程序的全部或部分代码翻译成本地机器码，提升程序速度。当要装载某个类时，编译器会先找到其 `.class` 文件，然后将该类的字节码转入内存。此时有两种方式可供选择

- 一、让即时编译器编译所有代码，但是这种做法有两个缺陷：
    - ①这种加载动作散落在整个程序的生命周期内，累加起来要花很多时间
    - ②会增加可执行代码的长度（字节码要比 JIT 展开后的本地机器码小很多），这将导致页面调度，从而降低程序速度。
- 二、惰性评估，只在必要的时候编译代码。

### 常见异常

> ArrayIndexOfBoundsException
>
> NullPointException
>
> OutOfMemmory

## OOP

- 面向对象是指把相关的数据和方法组织为一个整体来看待，从更高的层次来进行系统建模，更贴近事物的自然运行模式【来自百度】
- 当需要实现一个功能时，不关心具体的步骤，而是找一个已经具有该功能的人，来替我们做事。
- 面向对象的基本特征：继承，封装，多态

### 类&对象

- 类：一组相关属性和行为的集合。
- 对象：具体存在的实例，是真实地。 实例==对象。
- 代码层面，必须先有类，才能创建出对象。

<b>定义类的格式</b>

```java
修饰符 class 类名{
	//1.成员变量（Field 描述类和对象的属性信息）

	//2.成员方法（Method：描述类或者对象的行为信息）

	//3.构造器（Constructor：初始化一个类的对象并返回引用）

	//4.代码块

	//5.内部类
}
```

### 构造器

- 作用：初始化一个类的对象并返回。
- 构造器初始化对象的格式：<span style="color:blue">类名 对象名称 = new 构造器()</span>

### this 关键字的作用

- this 代表当前对象的引用。
- this 关键字可以用在实例方法和构造器中。
- this 用在方法中，谁调用这个方法，this 就代表谁。
- this 用在构造器，代表构造器正在初始化那个对象的引用。

### 封装

<b>封装的作用</b>

- 可以提高安全性，合理暴露合理封装。
- 可以实现代码组件化。

<b>封装的规范</b>

- 建议成员变量都私有。
- 提供成套的 getter+setter 方法暴露成员变量的取值和赋值。

<b>小结：</b>封装的核心思想=\=\>合理隐藏，合理暴露。

### static 关键字

Java 通过成员变量是否有 static 修饰来区分是属于类的还是属于对象的。

static \=\=\> 静态 \=\=\> 修饰的成员（方法和成员变量）属于类本身。

- 有 static，静态成员变量：属于类本身。
- 无 static，实例成员变量：属于每个实例对象，必须用类的对象来访问。

成员方法也类似：

- 静态方法
- 实例方法

static 修饰，属于类本身，随类的加载而加载，因为只有一份，所以可以被类和类的对象共享。

### 类与类之间的关系

- <b>依赖（user-a）</b>：一个类使用了另一个类的方法，A 调用了 B 的方法，B 出 bug 了，A 也可能出 bug，软件工程中称之为耦合。
- <b>聚合（has-a）</b>：一个对象将一个或者多个其它对象作为自己的成员
- <b>继承（is-a）</b>：一个子类继承父类，<span style="color:red">那么子类会拥有父类的非 private 修饰的方法、变量。</span>

### 对象的内存图

- 方法区中存放 class 信息。class 中的成员方法一直在方法区中。
  - 方法区的实现有两种：
    - jdk 1.8 以前（不包括 1.8）用永久代实现
    - jdk 1.8 及 1.8 以后用元空间实现
  - 1.8 及其以后，方法区中的常量池和静态变量都移动到了 JVM 堆中。具体看 openJDK 的描述 [JEP 122: Remove the Permanent Generation (java.net)](http://openjdk.java.net/jeps/122)
- 堆中拿到成员方法的地址，通过地址对方法进行调用【回忆组成原理】。
- 堆将方法区中的成员变量拿到堆中（相当于 copy 一份），对其进行初始化值的操作【不同对象的成员变量是独立的（非静态成员变量）】
- main 方法中的变量指向堆中的对象，并对对象进行赋值操作。
- stack--栈，FIFO

### 成员变量和局部变量

- 定义的位置不一样【重点】
  - 局部变量：在方法的内部
  - 成员变量：在方法的外部，直接写在类当中
- 作用范围不一样【重点】
  - 局部变量：只有方法当中才可以使用，出了方法就不能再用
  - 成员变量：整个类全都可以通用。
- 默认值不一样【重点】
  - 局部变量：没有默认值，如果要想使用，必须手动进行赋值
  - 成员变量：如果没有赋值，会有默认值，规则和数组一样
- 内存的位置不一样（了解）
  - 局部变量：位于栈内存
  - 成员变量：位于堆内存
- 生命周期不一样（了解）[ 通常是这样，但是不绝对 ]
  - 局部变量：随着方法进栈而诞生，随着方法出栈而消失（[局部内部类访问方法中的局部变量为什么加final - mjsky - 博客园 (cnblogs.com)](https://www.cnblogs.com/mjblogs/p/4971630.html)）
  - 成员变量：随着对象创建而诞生，随着对象被垃圾回收而消失

```java
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

### 变量的初始化顺序

```java
public Class Counter{
    int i;
    public Counter(){
        i=7;
        //在调用构造方法为 i 赋值为 7之前，i 就已经被初始化为 0 了。
        // 即，自动初始化发生在构造器被调用之前。
    }
}
/**
PS：回顾下类加载过程
- 加载（将字节码文件加载进内存）
- 验证（验证 Class 文件是否符合 JVM 规范，并确保它们不会危害虚拟机安全）
- 准备（为静态变量分配内存并设置类变量的初始值）
- 解析（将常量池内的符号引用替换为直接引用）
- 初始化（执行 clinit 方法），初始化指的是代码中定义的初始化，而非编译器默认的初始化
类加载是类加载，创建实例化对象是创建对象。
*/
```
