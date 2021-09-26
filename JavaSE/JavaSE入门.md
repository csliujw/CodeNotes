# 第一部分 入门

## 第二章 开发环境

### Java虚拟机

- JVM：Java虚拟机。Java代码是运行在虚拟机上的。
- 跨平台：代码运行在虚拟机上，不同版的OS（linux，windows，mac）对应不同的虚拟机。虚拟机本身不具备跨平台功能，每个OS下都有不同版本的虚拟机。【可理解为，各个OS下的虚拟机都是采用一套编码指令集，JVM是面向操作系统的，它负责把Class字节码解释成系统所能识别的指令并执行，同时也负责程序运行时内存的管理】

### JRE和JDK

- JRE (Java Runtime Environment):Java程序的运行时环境，包含JVM和运行时所需要的核心类库
- JDK(Java Development Kit):是Java程序开发的工具包，包含JRE和开发人员使用的工具。
- 运行Java程序有jre就行，开发Java程序需要JDK。
- Windows会把%CC% CC当作变量进行翻译

## 第三章 入门程序

### 程序开发步骤说明

- 编写、编译、运行
- Java源程序-->Java字节码文件-->JVM运行
- Javac.exe 编译器 处理后 Java文件变成字节码文件
- Java.exe 解释器 解释字节码的内容

## 第四章 常量

- 常量；在程序运行期间固定不变的量。
- 常量的分类
  - 字符串常量：凡是用双引号引起来的部分叫字符串常量。“asdfas”，可以是空串
  - 整数常量：直接写上数字的，没有小数点。
  - 浮点数常量：直接写上数字的，有小数点。
  - 字符常量：用单引号引起来的 ‘A’，不能是空字符‘’。
  - 布尔常量：只有两种取值。true，false
  - 空常量：null，代表没有任何数据。不能直接用来打印。syso(null)是错的。

## 第五章 变量&数据类型

### 数据类型

#### 基本数据类型

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

#### 引用数据类型

> **字符串，数组，类，接口，Lambda**

注意事项：

- 字符串不是基本数据类型
- 浮点数可能是一个近似值
- 数据范围与字节数不一定相关。如float数据范围比long更大，但float是4字节，long是8字节
- 浮点数默认是double，如果一定要用float，需要加上一个后缀F(推荐大写)
- 如果是整数，默认为int，如果一定要用long，需要加上一个后缀L(推荐大写)

### 变量

- 变量：程序运行期间，内容可以发生改变的量
- 编译原理，左值与右值

### 强制数据类型转换

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

### ASCII码表

```shell
0 -- 48
A -- 65
a -- 97
```

### 数字和字符的对照关系表（编码表）

```java
ASCII码表：American Standard Code for Information Interchange
Unicode码表：万国码。也是数字和符号对照关系，开头0-127部分和ASCII完全一样，但是从128开始包含更多字符。
```

### 易错点

byte short char 这些在计算的时候，会有类型提升，提升为 int 进行计算。

```java
public static void main(String[] args) {
    byte a = 8;
    byte b = 127;
    b = a+b; // 会报错，提示你要进行强制类型转换。因为计算的时候，a和b会被提升为int类型，然后再进行计算，得到的结果也是int的，要把int类型的赋值给byte类型的变量需要进行强制类型转换。
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

静态内部类加载时机的测试。遇到new、getstatic、putstatic或invokestatic 这四条字节码指令执行的时候，如果类没有进行过初始化，则需要先触发其初始化。

```java
public class Demo {
    // VM参数 -XX:+TraceClassLoading
    public static void main(String[] args) {
        int a = 0x2f;
        System.out.printf("", TestClassLoading.d); // 调用静态内部类，控制台输出，它被加载了。不用静态内部类，它就不加载。
    }

    static class TestClassLoading {
        static int d =10;
        {
            System.out.println("d");
        }
    }
}
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
  short = 13; // 编译优化
  先计算出的结果在进行赋值的
  称为编译器的常量优化。
  ```

## 第七章 基本语法

### switch

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

- 多个case后面的数值不可以重复
- switch后面小括号中只能是下列数据类型
  - 基本数据类型 byte/short/char/int
  - 引用数据类型 String字符串、enum枚举

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
    public TestSwitch() {
    }

    public static void main(String[] args) {
        String str = "dd";
        byte var3 = -1;
        switch(str.hashCode()) {
        case 3168:
            if (str.equals("cc")) {
                var3 = 1;
            }
            break;
        case 3200:
            if (str.equals("dd")) {
                var3 = 0;
            }
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

### 循环

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

### 方法重载与重写

方法调用的三种格式

```java
1.单独调用：方法名称(参数)
2.打印调用：System.out.println(方法名称(参数))
3.赋值调用：数据类型 变量名称 = 方法名称(参数)
```

方法重载 Overload 

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

```
动态初始化有默认值的过程，
整型 默认为 0
浮点 默认为 0.0
字符 默认为 '\u0000'
布尔 默认为 false
引用 默认为 null

静态初始化也有，不过系统自动马上将默认值替换为了大括号当中的具体数值。
```

#### 数组作为参数，返回值

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

### Java内存划分

- 1.栈（stack）：存放的都是方法中的局部变量。方法的运行一定要在栈中运行
  - 局部变量：方法的参数，或者方法{}内部的变量
  - 作用域：一旦超出作用域，立刻从栈内存当中消失
- 2.堆（heap）：凡是new出来的东西都在堆中
  - 堆里面的数据都有默认值。默认值同上
- 3.方法区（method area）：存储.class相关信息，包含方法的信息。
- 4.本地方法栈（native method stack）：与操作系统相关
- 5.寄存器（register）：与CPU相关

----

Java 的垃圾回收，对于提高对象的创建速度，具有明显的效果。Java从堆空间分配空间的速度，可以和其他语义从堆栈上分配空间的速度相媲美。在某些 Java VM 中，堆的实现截然不同，但是堆内存的分配可以看做：有一个堆指针，简单移动到尚未分配的区域，通过这种方式分配对象内存，其效率比得上 C++ 在栈上分配空间的效率。当然，在实际簿记工作方面，还有少量额外的开销，但是比不上查找可用空间的开销。（**Java GC 会清理出可用的空间，堆指针在空用空间中移动，这样就完成了内存的分配。而 C++ 需要遍历查找可用的内存，这个查找开销较大。这样一对比，会发现，Java 分配对象的速度并不比 C++慢**）

Java 的 GC 工作的时候，一面回收内存空间，一面使堆中的对象紧凑排列。

-----

Java 的优化技术\=\=\=>JIT（Just-In-Time）：这种技术可以把重新全部或部分翻译成本地机器码，提升程序速度。当要装载某个类时，编译器会先找到其 `.class` 文件，然后将该类的字节码转入内存。此时有两种方式可供选择：

- 一、让即时编译器编译所有代码，但是这种做法有两个缺陷：①这种加载动作散落在整个程序的生命周期内，累加起来要花很多时间，②会增加可执行代码的长度（字节码要比 JIT 展开后的本地机器码小很多），这将导致页面调度，从而降低程序速度。
- 二、惰性评估，只在必要的时候编译代码。

### 常见异常

> ArrayIndexOfBoundsException

> NullPointException

> OutOfMemmory