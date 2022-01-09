# leetcode

## 1114. 按序打印

- 考察多个线程按顺序执行。先执行 first 线程，再执行 second，再执行 third；
- 需要确保线程同步。
    - first 第一个执行；若 second 和 thrid 先执行，则先阻塞。
    - 
- synchronized、ReentrantLock、Semaphore 可以实现；LockSupport 呢？

<img src="img\image-20211113165827570.png" style="width:100%">

> Synchronized 实现线程有序执行。

# 网传题



