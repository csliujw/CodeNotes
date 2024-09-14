# 准备

- scala 介绍
- 安装
- 执行原理
- 整合 IDEA
  - 插件市场安装 scala 或直接安装离线插件
  - 退出 idea
- scala 项目
  - 创建 java 项目，为项目添加 scala 支持

- scala 与 Java 的关系
  - scala 是 jvm 语言
  - scala 是多范式的

# quick start

scala 解释器回顾基本代码

var val 定义变量

`val i: Int = 10`

<b>条件判断语句有返回值</b>

```scala
scala> if(age>18)"jerry" else "hello"
val res0: String = jerry

// any 类型，任意类型
scala> if(age>18)"jerry" else 18
val res2: Any = jerry
```

判断闰年

```scala
```

<b>while / do-while 循环，求和</b>

```scala
object Hello {
    def main(args: Array[String]): Unit = {
        var sum = 0
        var num = 1;

        while (num <= 10) {
            sum += num;
            num += 1;
        }
        println("sum = %d".format(sum))
    }
}


object Hello {
    def main(args: Array[String]): Unit = {
        var sum = 0
        var num = 1

        do {
            sum += num
            num += 1
        } while (num <= 10)
        println("sum = %d".format(sum))
    }
}
```

<b>用户输入，输入用 StdIn</b>

练习：用户输入密码进行登录，三次错误抛出异常。学习中止循环的方式 [Scala跳出循环的三种方法-CSDN博客](https://blog.csdn.net/qq_39532946/article/details/77746348)

```scala
package org.example

import scala.io.StdIn
import scala.util.control.Breaks._

object Hello {
    def main(args: Array[String]): Unit = {
        val username = "root"
        val password = "123"
        var total = 1
        breakable { // break 跳出 breakable 包裹的内容
            while (total <= 3) {
                val curU = StdIn.readLine("请输入用户名")
                val curP = StdIn.readLine("请输入密码")
                if (curU == username && password == curP) {
                    println(s"用户${curU} 登录成功")
                } else {
                    println("错误，请重新登录")
                    break
                }
                total += 1
            }
        }
    }
}
```

<b>for 循环</b>

```scala
for( 变量 <- 集合 ){
    
}

// 防卫 if
for(...;..; if 防卫语句)
```





