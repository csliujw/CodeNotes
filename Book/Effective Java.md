# 概述

这个笔记主要是记录下平时看 Effective Java 的一些想法。不会按照书中的章节顺序阅读哦！

# 并发

## 同步访问可共享的可变数据

这里讲了下 Java 语言规范保证了读写变量是原子性的（long float 除外，long 和 float 可能为分成两次读取，写入。每次读取写入 32 位，long 和 float 是 64 位，需要操作两次），但是由于 `JMM`，一个线程所做的变化如何对其他线程可见。

我的理解是，线程在自己的工作内存空间中对变量的读写操作是原子性的，但是所有的共享数据都是从主存copy到线程工作内存，这使得不同线程操作同一个变量，在自己的工作内存中改变了值，但是没有即时同步到主存中，导致数据出现问题，这种情况需要进行并发访问。

那么 `cpp` 的并发安全？

```java
package com.payphone.thread;

import java.util.concurrent.TimeUnit;

public class Atomic {
    private static boolean stopRequested = false;

    public static void main(String[] args) throws InterruptedException {
        // lambda表达式
        Thread thread = new Thread(() -> {
            int i = 0;
            while (!stopRequested) {
                i++;
            }
        });

        thread.start();
        TimeUnit.SECONDS.sleep(1);
        stopRequested = true;
    }
}
```

`JVM` 会存在优化提升，提升后的结果为：

但是我反编译的结果并不是这样。一定要用 jad 反编译吗。

```java
if(!done){
    while(true)
        i++;
}
```

jad 因为版本的问题，不支持 jdk1.8 的语法，所以我用了 lambda 表达式的字节码文件反编译时会出错。