# 难点

ThreadLocal原理，模型。

Unsafe类

# 并发编程基础

## 线程中断

不直接种植线程的执行，让被中断的线程根据中断状态自行处理。这块不是很好理解。

### 线程中断的api及其源码

- void interrupt() 方法  如 XThread.interrupt() 将当前线程中断标志设为true，不一定是XThread线程。

- boolean isInterrupted() 方法  检测当前线程是否被中断。The interrupted status of the thread is unaffected by this method. 使用这个方法不会影响线程当前的中断状态。

  ```java
  public boolean isInterrupted() {
          return isInterrupted(false);
  }
  
      /**
       * Tests if some Thread has been interrupted.  The interrupted state
       * is reset or not based on the value of ClearInterrupted that is
       * passed.
       */
  @HotSpotIntrinsicCandidate
  private native boolean isInterrupted(boolean ClearInterrupted);
  ```

  一看源码命名，见名知意。

- boolean interrupted() 方法。 检测当前线程是否被中断。会清楚线程当前的中断状态，请看下列源码。

  ```java
  public static boolean interrupted() {
      // 清除当前线程的状态
      return currentThread().isInterrupted(true);
  }
  
  /**
       * Tests if some Thread has been interrupted.  The interrupted state
       * is reset or not based on the value of ClearInterrupted that is
       * passed.
       */
  @HotSpotIntrinsicCandidate
  private native boolean isInterrupted(boolean ClearInterrupted);
  ```

### 如何优雅的中断线程

何谓优雅的中断线程？

需要中断线程，且被中断线程需要完成的事已经完成。

```java
// 伪代码
public void run(){
    try{
        // 线程退出条件
        if(!Thread.currentThread().isInterrupted() && need to do other things){
            do other things;
        }
    }catch(Exception e){
        // thread was interrupted during sleep or wait
    }finally{
        // cleanup if require
    }
}
```

## 线程上下文切换

时间片到了切换，被剥夺了切换。

## 线程死锁

死锁：竞争资源，相互等待，谁也不让谁。

产生死锁的条件：以下都得满足！

- 互斥条件：有些资源必须互斥
- 请求保持：可破坏
- 不可剥夺：有些资源不能被剥夺
- 环路等待：可破坏

## 守护线程和用户线程

最后一个非守护进程结束的时候，JVM会正常退出。

JVM不等守护进程执行完毕就会退出。

简单的API调用就可测试，无难度，不做笔记。

## ThreadLocal

ThreadLocal内部有一个静态内部类，ThreadLocalMap。

ThreadLocal提供了线程本地变量





