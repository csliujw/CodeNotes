# 第一章

一些基础内容，只记录一些必要的。

## 简单的C++程序

返回值 -1 通常被当作程序错误的标识。重新编译并运行你的程序，观察你的系统如何处理main返回的错误标识。将 main 函数的返回值修改为 -1，观察结果。

```cpp
#include<iostream>
using namespace std;

int main(){
    cout<<"modify return value"<<endl;
    return -1;
}
// 控制台提示
// Command executed failed（Exit code 255）
```

## 输入输出

C++ 的 iostream 库包含两个基础类型 istream 和 ostream，分别表示输入流和输出流。

### 标准IO对象

标准库定义了 4 个 IO 对象。为了处理输入

- 名为 cin（发音为 see-in）的 istream 类型的对象。这个对象被称为标准输入（standard input）。
- 名为 cout（发音为 see-out）的 ostream 类型的对象。此对象被称为标准输出（standard output）。
- 名为 cerr 和 clog（发音分别为 see-err 和 see-log）的 ostream 类型对象。我们通常用 cerr 来输出警告和错误消息，因此它也被称为标准错误（standard error）。而 clog 用来输出程序运行时的一般性信息。系统通常将程序所运行的窗口与这些对象关联起来。

当我们读取 cin，数据将从程序正在运行的窗口读入，当我们向 cout、cerr 和 clog 写入数据时，将会写到同一个窗口。

```cpp
#include<iostream>
using namespace std;

int main(){
    int n1,n2;
    // cout 使用 << 将字符串给定到 cout 对象中
    // endl 将缓冲区的数据刷到设备中。
    std::cout<<"please input two number"<<std::endl;
    // 使用 >> 将控制台的输入给定到 n1,n2 中
    std::cin>>n1>>n2;
    std::cout<<"result is:"<<n1+n2<<std::endl;
    return 0;
}
```

程序使用了 `std::cout` 和 `std::endl`，而不是直接的 cout 和 endl 。前缀 `std::` 指出名字 cout 和 endl 是定义在名为 std 的命名空间（namespace）中的。命名空间可以帮助我们避免不经意的名字定义冲突，以及使用库中相同名字导致的冲突。标准库定义的所有名字都在命名空间 std 中。

习题：解释下面程序是否合法，合法输出什么，不合法原因是什么，如何修改

```cpp
#include<iostream>

// 解释下面程序片段是否合法
int main(){
    int v1,v2;
    std::cout<<"The sum of "<<v1;
    << "and "<< v2;
    << " is "<< v1+v2 <<std::endl;
    return 0;
}
/*
不合法， ; 表示这跳语句终止了，第二条语句的 << 就不知道给定到（流的重定向？）那个对象了。
去掉 v1,v2 后面的 ; 即可
*/
```

## 注释

两种注释，单行注释和多行注释

```cpp
//
/*
*
*/
```

```cpp
#include<iostream>

// 解释下面程序片段是否合法
int main(){
    int v1,v2;
    std::cout<<"/*"<<std::endl;
    std::cout<<"*/"<<std::endl;
 	
    // 正确
    std::cout<</*"*/" /*"/*"*/<<std::endl;

    return 0;
}
```

## 控制流

if、else、for、while；

读取数量不定的输入数据

```cpp
#include<iostream>

int main(){
    int value=0,sum=0;
    while(std::cin>>value){
        sum+=value;
    }
    std::cout<<sum<<std::endl;
    return 0;
}
/*
Linux 下最后输入 EOF 就可以停止了。
*/
```

当从键盘向程序输入数据时，对于如何指出文件结束，不同操作系统有不同的约定。在 Windows 系统中，输入文件结束符的方法是敲 Ctrl+Z（按住 Ctrl 键的同时按 Z 键），然后按 Enter 或 Return 键。在 UNIX 系统中，包括 Mac OS X 系统中，文件结束符输入是用 Ctrl+D。

## 类介绍

不记

# 第Ⅰ部分-C++基础

介绍 C++ 的基本特征：

- 变量和基本类型
- 字符串、向量和数组
- 表达式
- 语句
- 函数
- 类

## 第二章-变量和基本类型

C++ 定义了一套包括算术类型（arithmetic type）和空类型（void）在内的基本数据类型。其中算术类型包含了字符、整型数、布尔值和浮点数。空类型不对应具体的值，仅用于一些特殊的场合，例如最常见的是，当函数不返回任何值时使用空类型作为返回类型。

### 基本内置类型

#### 算术类型

| 类型        | 含义              | 最小尺寸     |
| ----------- | ----------------- | ------------ |
| bool        | 布尔类型          | 未定义       |
| char        | 字符              | 8 位         |
| wchar_t     | 宽字符 16         | _位          |
| char16_t    | Unicode 字符 16   | _位          |
| char32_t    | Unicode 字符 32   | _位          |
| short       | 短整型 16         | _位          |
| int         | 整型 16           | _位          |
| long        | 长整型 32         | _位          |
| long long   | 长整型 64         | _位          |
| float       | 单精度浮点数      | 6 位有效数字 |
| double      | 双精度浮点数 10   | _位有效数字  |
| long double | 扩展精度浮点数 10 | _位有效数字  |

#### 类型转换

> 基本的类型转换

```cpp
#include<iostream>
using namespace std;
int main(){
    bool b = 42; // b 为 true

    int i = b; // i 为 1
    i = 3.14;   // i 为 3
    double p1 = i; // p1 为 3
    unsigned char c = -1; // 假设 char 占 8bit，c 的值为255
    signed char c2 = 256; // 假设 char 占 8bit，c 的值是未定义

    cout<<"b="<<b<<"\n"
        <<"i="<<i<<"\n"
        <<"p1="<<p1<<"\n"
        <<"c="<<int(c)<<"\n"
        <<"c2="<<c2<<"\n";
    return 0;
}
```

由于不同的操作系统数据类型的表现能力不一样，如某些 OS int 是 4 字节，有些则不是 4 字节。因此在进行编程的时候要避免无法预知和依赖于实现环境的行为。

> 含无符号类型的表达式

如果一个有符号的 int 和无符号的 int 进行运算，最后的数据类型会被提升为无符号数据。

```cpp
void test2(){
    int a = -1;
    unsigned int b = 0;
    cout<<a+b<<endl; // 4294967295
}
```

如果一个有符号的 long 和一个无符号的 int 进行运行，最后无符号数据类型会被提升为 long。

```cpp
void test2(){
    long a = -1;
    unsigned int b = -1;
    // b 原先是 1000 ... 0001
    // 提升为 long 后 0000 ... 1000 ... 0001
    cout<<a+b<<endl; // 4294967294
}
```

> 习题，读程序，说结果

```cpp
void test3(){
    // 42 = 32+8+2
    unsigned u = 10,u2 = 42;
    // 0000 ... 0000 0000 1010
    // 0000 ... 0000 0010 1010
    
    // 0000 ... 0000 0010 1001
    cout<< u2-u <<endl; // 32
    
    // 10-42 = -32
    // 借位做减法
    // 0000 ... 0000 0000 1010
    // 0000 ... 0000 0010 1010
    // 1111 ... 1111 1110 0000

    // 1000 ... 0000 0010 0000
    // 负数的补码 = 取反+1（无符号的运算最后还是无符号）
    // 1111 ... 1111 1101 1111 + 1
    // 最后当成无符号数来算出他的值
    // 1111 ... 1111 1110 0000
    cout<< u-u2 <<endl;

    int i = 10,i2 = 42;

    cout<< i2-i <<endl;
    cout<< i-i2 <<endl;
    cout<< i-u <<endl;
    cout<< u-i <<endl;
}
```

#### 字面值常量

> 整型和浮点型字面量

以 0 开头的整数代表八进制数，以 0x 或 0X 开头的代表十六进制数。

```cpp
void test4(){
    // 4*1+2*8 
    // 0 开头八进制
    cout<<024<<endl;

    // 0x 开头十六进制
    // 1*16+2 = 18
    cout<<0x12<<endl;
}
```

浮点型字面值表现为一个小数或以科学计数法表示的指数，其中指数部分用 E 或 e 标识：

```shell
3.14  3.14e0  0.  0e0  .001
```

注意：默认的，浮点型字面值是一个 double。

> 字符和字符串字面量

由单引号括起来的一个字符称为 char 型字面值，双引号括起来的零个或多个字符则构成字符串型字面值。

```c
'a'  // 字符字面量
"Hello" // 字符串字面值
```

> 指定字面量的类型

什么叫指定字面量的类型呢？比如你把字面量 char 指定为 wchar_t 类型的。

| 指定方式 | 说明                                                   |
| -------- | ------------------------------------------------------ |
| L'a'     | 宽字符型字面量，类型是 wchar_t                         |
| u8"hi!"  | utf-8 字符串字面值 (utf8 用 8 位编码一个 Unicode 字符) |
| 42ULL    | 无符号整型字面值，类型 unsigned long long              |
| 1E-3F    |                                                        |
| 3.14158L | long double 类型                                       |

### 变量

#### 变量定义

定义的方式和其他语言类似

```cpp
int a=0,b=0;
```

> 初始值

C++ 中，初始化是一个异常复杂的问题，在 C++ 中，初始化和赋值是两个完全不同的操作。

> 列表初始化

```cpp
void test5(){
    int n = 10;
    // 以下为列表初始化。
    int n1 = {10};
    int n2{10};
    int n3(10);
    // 10:10:10
    cout<<n1<<":"<<n2<<":"<<n3<<endl;    
}
```

<span style="color:orange">列表初始化的好处：如果我们使用列表初始化且初始值存在丢失信息的风险，则编译器将报错。</span>

```cpp
void test5(){
    double n = 10.123;
    int n1 = {n};
    int n2{n};
    int n3(n);
    
    // g++ cast.cpp -o a -std=c++11 会报错
    cout<<n1<<":"<<n2<<":"<<n3<<endl;    
}
/*
cast.cpp: In function ‘void test5()’:
cast.cpp:69:16: warning: narrowing conversion of ‘n’ from ‘double’ to ‘int’ inside { } [-Wnarrowing]
     int n1 = {n};
                ^
cast.cpp:70:13: warning: narrowing conversion of ‘n’ from ‘double’ to ‘int’ inside { } [-Wnarrowing]
     int n2{n};
*/
```

> 默认初始化

如果定义变量时没有指定初值，则变量被默认初始化（default initialized），此时变量被赋予了“默认值”。<span style="color:orange">函数外部的变量会有默认初始化，而函数内部的变量没有默认初始化！如果试图拷贝或以其他形式访问此类值将引发错误。</span>

```cpp
int out;
void test6(){
    int inner; // 没有默认初始化
    cout<<out<<":"<<inner<<endl;
}
// 0:21845
```

> 练习题

```cpp
int i = {3.14}; // C++11 报错
int i = 3.14;   // C++11 不报错
```

#### 变量声明和定义

为了允许把程序拆分成多个逻辑部分来编写，C++ 支持分离式编译（separate compilation）机制，该机制允许将程序分割为若干个文件，每个文件可被独立编译。

如果将程序分为多个文件，则需要有在文件间共享代码的方法。例如，一个文件的代码可能需要使用另一个文件中定义的变量。如 `std::cout` 和 `std::cin`，它们定义于标准库，却能被我们写的程序使用。

为了支持分离式编译，C++ 将声明和定义区分开来。声明（declaration）使得名字为程序所知，一个文件如果想使用别处定义的名字则必须包含对那个名字的声明。而定义（definition）负责创建与名字关联的实体。变量声明规定了变量的类型和名字，在这一点上定义与之相同。但是除此之外，定义还申请存储空间，也可能会为变量赋一个初始值。

<span style="color:red">如果想声明一个变量而非定义它，就在变量名前添加关键字 extern，而且不要显式地初始化变量：</span>

```cpp
extern int i; // 声明 i 而非定义 i
extern double pi = 3.14; // 声明并定义 pi
int j; // 声明并定义 j
```

在函数体内部，如果试图初始化一个由 extern 关键字标记的变量，将引发错误。

#### 作用域

和其他语言一样，不赘述。

> 习题

```cpp
#include<iostream>
using namespace std;

int i = 42;
int main(){
    int i = 100;
    int j = i; 
    // 100
    cout<<j<<endl;
    return 0;
}
```

```cpp
#include<iostream>
using namespace std;

int main(){
    int i = 100, sum = 0;
    // for 循环内部的 i 是块级作用域。而在 Java 中会报错。
    for(int i=0; i!=10; ++i)
        sum+=i;
    cout<<i<<":"<<sum<<endl;
    return 0;
}
// 100:45
```

### 复合类型

复合类型（compound type）是指基于其他类型定义的类型。C++ 有几种复合类型，本章将介绍其中的两种：引用和指针。

#### 引用

引用（reference）为对象起了另外一个名字。

```cpp
#include<iostream>
using namespace std;
int main(){
    int val = 1024;
    int &refVal = val;
    // int &refVal2; // 报错，引用在定义时必须被初始化
    refVal = 10;
    cout<<val<<endl; // 10
    return -1;
}
```

定义引用时，程序把引用和它的初始值绑定（bind）在一起。一旦初始化完成，引用将和它的初始值对象一直绑定在一起。由于无法让引用重新绑定到另外一个对象，因此引用必须初始化。

注意：引用并非对象，相反的，它只是为一个已经存在的对象所起的另外一个名字。因为引用本身不是一个对象，所以不能定义引用的引用。

```cpp
#include<iostream>
using namespace std;
int main(){
    int &refVal = 10; // 报错，引用类型的初始值必须是一个对象
    double dVal = 3.1;
    int &refValI = dVal; // 报错，引用类型的初始值必须一致，此处必须是 int 型对象
}
```

> 引用定义

```cpp
#include<iostream>
using namespace std;
int main(){
    // ri 是一个引用，与 i 绑定在了一起
    int i, &ri = i;
    i = 5;
    ri = 11; 
    // 11, 11
    cout<<i<<" "<<ri<<endl;
    return 0;
}
```

#### 指针

- 指针本身就是一个对象，允许对指针赋值和拷贝，而且在指针的生命周期内它可以先后指向几个不同的对象。
- 指针无须在定义时赋初值。和其他内置类型一样，在块作用域内定义的指针如果没有被初始化，也会有一个不确定的值。

# 第Ⅱ部分-C++标准库

# 第Ⅲ部分-类设计者的工具