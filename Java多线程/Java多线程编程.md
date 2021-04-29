# 概述

Java多线程编程。

- 线程的基本用法
- 线程池
- juc
- 死锁&常见错误加锁
- JMM
- ThreadLocal

# 一、线程基本用法

## 1.1 新建线程

### 1.1.1 继承Thread

```java
public class CreateByThread extends Thread {
    @Override
    public void run() {
        for (int i = 0; i < 100; i++) {
            System.out.println(i);
        }
    }

    public static void main(String[] args) {
        CreateByThread createByThread = new CreateByThread();
        // 让线程执行createByThread对象中的run方法
        Thread thread = new Thread(createByThread);
        // 源码注释：It is never legal to start a thread more than once；start方法只能调用一次
        thread.start();
    }
}
```

==正常编写程序，资源类和线程类应该区分开来。==

```java
public class CreateByThread02 {
    public static void main(String[] args) throws InterruptedException {
        // 实例化资源类
        Resource resource = new Resource();
        MyThread myThread = new MyThread(resource);
        Thread thread = new Thread(myThread);
        thread.start();
        thread.join();
        System.out.println(resource.getCount());

    }
}

class Resource {
    private int count = 0;

    public int getCount() {
        return count;
    }

    public void increase() {
        count++;
    }
}

class MyThread extends Thread {
    private Resource resource;

    public MyThread(Resource resource) {
        this.resource = resource;
    }

    @Override
    public void run() {
        for (int i = 0; i <10; i++) {
            resource.increase();
        }
    }
}
```

### 1.1.2 实现Runnable接口

```java
public class CreateByRunnable {
    public static void main(String[] args) throws InterruptedException {
        Resource resource = new Resource();
        RunnableDemo runnableDemo = new RunnableDemo(resource);
        Thread thread = new Thread(runnableDemo);
        thread.start();
        thread.join();
        System.out.println(resource.getCount());
    }
}

class RunnableDemo implements Runnable {
    private Resource resource;

    public RunnableDemo(Resource resource) {
        this.resource = resource;
    }

    @Override
    public void run() {
        for (int i = 0; i < 100; i++) {
            resource.increase();
        }
    }
}

class Resource {
    private int count = 0;

    public int getCount() {
        return count;
    }

    public void increase() {
        count++;
    }
}
```

### 1.1.3 FutureTask

```java
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

public class CreateByFutureTask {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        MyCallable myCallable = new MyCallable();
        FutureTask futureTask = new FutureTask(myCallable);
        Thread thread = new Thread(futureTask);
        thread.start();
        // get拿到执行完成后的数据，拿不到会在此处阻塞。我看源码用的是yeild,到底是什么原理，后面再补充。
        System.out.println(futureTask.get());
        // 前面的sout语句无法执行，后面的也无法执行，因为都是同一个main线程中执行！！
        System.out.println(123);
    }
}

// 对比Runnable而已，增加了返回值功能。
class MyCallable implements Callable<Integer> {

    @Override
    public Integer call() throws Exception {
        Integer count = 0;
        for (int i = 0; i < 20; i++) {
            TimeUnit.SECONDS.sleep(1);
            count = (count + i) % Integer.MAX_VALUE;
        }
        return count;
    }
}
```

### 1.1.4 线程池&自定义线程池

> 线程池

java自带的线程池，但是一般不用。因为自带的线程池设置的阻塞队列的大小实在是太大了，容易出问题，一般都是自定义线程池。

```java
import java.util.concurrent.*;

/**
 * 线程池可以复用线程、控制最大并发数、管理线程。
 * 线程池中有可以用的线程，就拿线程出来用，没有就先暂时阻塞任务，等待有线程了在执行那些任务。
 * 使用线程池在一定程度上可以减少上下文切换的开销。
 */
public class ThreadPoolDemo {

    public static void main(String[] args) throws InterruptedException {
        ScheduledThreadPool();
        // new 一个线程，传入实现Runnable接口的对象。调用thread对象的start方法，从而调用Runnable的run方法
        // 实现线程调用方法。我们把Runnable对象改了就可以实现线程复用？
    }

    /**
     * Executors.newFixedThreadPool(int i) ：创建一个拥有 i 个线程的线程池
     * 执行长期的任务，性能好很多
     * 创建一个定长线程池，可控制线程数最大并发数，超出的线程会在队列中等待
     */
    public static void FixedThreadPool() {
        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(5);
        // 循环100次，让5个线程处理100个业务
        for (int i = 0; i < 100; i++) {
            final int ii = i;
            fixedThreadPool.execute(() -> {
                try {
                    System.out.println(Thread.currentThread().getName() + "\t 给用户" + ii + "办理业务");
                    TimeUnit.SECONDS.sleep((int) (Math.random() * 15));
                    System.out.println(Thread.currentThread().getName() + "\t 给用户" + ii + "办理业务结束");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    // 执行完成后，关闭当前线程的服务，归还给线程池
                    fixedThreadPool.shutdown();
                }
            });
        }
    }

    /**
     * Executors.newSingleThreadExecutor：创建一个只有1个线程的 单线程池
     * 一个任务一个任务执行的场景
     * 创建一个单线程化的线程池，它只会用唯一的工作线程来执行任务，保证所有任务按照指定顺序执行
     */
    public static void SingleThreadPool() {

    }

    /**
     * Executors.newCacheThreadPool(); 创建一个可扩容的线程池
     * 执行很多短期异步的小程序或者负载教轻的服务器
     * 创建一个可缓存线程池，如果线程长度超过处理需要，可灵活回收空闲线程，如无可回收，则新建新线程
     */
    public static void CacheThreadPool() {

    }

    /**
     * Executors.newScheduledThreadPool(int corePoolSize)：
     * 线程池支持定时以及周期性执行任务，创建一个corePoolSize为传入参数，最大线程数为整型的最大数的线程池
     */
    public static void ScheduledThreadPool() {
        ScheduledExecutorService scheduledThreadPool = Executors.newScheduledThreadPool(3);
        // 周期执行。
        ScheduledFuture<?> schedule1 = scheduledThreadPool.scheduleAtFixedRate(() -> {
            System.out.println(1);
        }, 10, 10, TimeUnit.SECONDS);

        // 只执行一次
        ScheduledFuture<?> schedule2 = scheduledThreadPool.schedule(() -> {
            System.out.println(2);
        }, 10, TimeUnit.SECONDS);
    }
}
```

> 自定义线程池写法

```java
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * 线程资源必须通过线程池提供，不允许在应用中自行显式创建线程
 * - 使用线程池的好处是减少在创建和销毁线程上所消耗的时间以及系统资源的开销，解决资源不足的问题，如果不使用线程池，有可能造成系统创建大量同类线程而导致消耗完内存或者“过度切换”的问题
 * 线程池不允许使用Executors去创建，而是通过 ThreadPoolExecutor 的方式，这样的处理方式让写的同学更加明确线程池的运行规则，规避资源耗尽的风险
 * - Executors返回的线程池对象弊端如下：
 * - FixedThreadPool和SingleThreadPool：
 * ---- 运行的请求队列长度为：Integer.MAX_VALUE，可能会堆积大量的请求，从而导致OOM
 * - CacheThreadPool和ScheduledThreadPool
 * ---- 运行的请求队列长度为：Integer.MAX_VALUE，可能会堆积大量的请求，从而导致OOM
 */
public class DefineThreadPool {
    /**
     * 手写线程池
     * -从上面我们知道，因为默认的Executors创建的线程池，底层都是使用LinkBlockingQueue作为阻塞队列的，而LinkBlockingQueue虽然是有界的，但是它的界限是 Integer.MAX_VALUE 大概有20多亿，可以相当是无界的了，因此我们要使用ThreadPoolExecutor自己手动创建线程池，然后指定阻塞队列的大小
     * 下面我们创建了一个 核心线程数为2，最大线程数为5，并且阻塞队列数为3的线程池
     */

    public static void AbortPolicy() {
        final Integer corePoolSize = 2;
        final Integer maximumPoolSize = 5;
        final long keepAliveTime = 1L;

        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
                corePoolSize,
                maximumPoolSize,
                keepAliveTime,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(3),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.AbortPolicy()
        );
        try {
            // maximumPoolSize +  LinkedBlockingQueue的大小 = 5+3 = 8； 运行的+阻塞的  最多8个任务
            for (int i = 0; i < 15; i++) {
                final int tmp = i;
                threadPoolExecutor.execute(() -> {
                    System.out.println(Thread.currentThread().getName() + "\t 给用户:" + tmp + " 办理业务");
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            threadPoolExecutor.shutdown();
        }
    }

    public static void CallerRunsPolicy() {
        final Integer corePoolSize = 2;
        final Integer maximumPoolSize = 5;
        final long keepAliveTime = 1L;

        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
                corePoolSize,
                maximumPoolSize,
                keepAliveTime,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(3),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
        try {
            // maximumPoolSize +  LinkedBlockingQueue的大小 = 5+3 = 8； 运行的+阻塞的  最多8个任务
            for (int i = 0; i < 150; i++) {
                final int tmp = i;
                // 我们发现，输出的结果里面出现了main线程，因为线程池出发了拒绝策略，把任务回退到main线程，然后main线程对任务进行处理
                threadPoolExecutor.execute(() -> {
                    System.out.println(Thread.currentThread().getName() + "\t 给用户:" + tmp + " 办理业务");
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            threadPoolExecutor.shutdown();
        }
    }

    // 直接丢弃任务，不报异常。
    public static void DiscardPolicy() {
        final Integer corePoolSize = 2;
        final Integer maximumPoolSize = 5;
        final long keepAliveTime = 1L;

        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
                corePoolSize,
                maximumPoolSize,
                keepAliveTime,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(3),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.DiscardPolicy()
        );
        try {
            // maximumPoolSize +  LinkedBlockingQueue的大小 = 5+3 = 8； 运行的+阻塞的  最多8个任务
            for (int i = 0; i < 150; i++) {
                final int tmp = i;
                // 我们发现，输出的结果里面出现了main线程，因为线程池出发了拒绝策略，把任务回退到main线程，然后main线程对任务进行处理
                threadPoolExecutor.execute(() -> {
                    System.out.println(Thread.currentThread().getName() + "\t 给用户:" + tmp + " 办理业务");
                });
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            threadPoolExecutor.shutdown();
        }
    }

    /**
     * <div>线程池的合理参数</div>
     * 生产环境中如何配置 corePoolSize 和 maximumPoolSize <br>
     * 这个是根据具体业务来配置的，分为CPU密集型和IO密集型 <br>
     * - CPU密集型 <br>
     * CPU密集的意思是该任务需要大量的运算，而没有阻塞，CPU一直全速运行 <br>
     * CPU密集任务只有在真正的多核CPU上才可能得到加速（通过多线程） <br>
     * 而在单核CPU上，无论你开几个模拟的多线程该任务都不可能得到加速，因为CPU总的运算能力就那些 <br>
     * CPU密集型任务配置尽可能少的线程数量： <br>
     * 一般公式：CPU核数 + 1个线程数 <br>
     * - IO密集型 <br>
     * 由于IO密集型任务线程并不是一直在执行任务，则可能多的线程，如 CPU核数 * 2 <br>
     * IO密集型，即该任务需要大量的IO操作，即大量的阻塞 <br>
     * 在单线程上运行IO密集型的任务会导致浪费大量的CPU运算能力花费在等待上 <br>
     * 所以IO密集型任务中使用多线程可以大大的加速程序的运行，即使在单核CPU上，这种加速主要就是利用了被浪费掉的阻塞时间。 <br>
     * IO密集时，大部分线程都被阻塞，故需要多配置线程数： <br>
     * 参考公式：CPU核数 / (1 - 阻塞系数) 阻塞系数在0.8 ~ 0.9左右 <br>
     * 例如：8核CPU：8/ (1 - 0.9) = 80个线程数 <br>
     */
    public static void note(){

    }

    public static void main(String[] args) {
        DiscardPolicy();
    }
}
```

### 1.1.5 对比总结

- 继承Thread
    - Java是单继承，继承Thread后就不能继承其他类了，有很强的局限性。
- 实现Runnable接口
    - 对比继承Thread来说更为灵活
- FutureTask
    - 对比Runnable来说，允许线程执行完后，给与一个返回值。
- 线程池
    - 线程复用。频繁的创建，销毁线程和消耗系统资源，线程池可解决该类问题。
    - 但是Java内置的线程池并不好用，因为自带的线程池设置的阻塞队列的大小实在是太大了，容易出问题，一般都是自定义线程池。

## 1.2 线程的一些方法

### 1.2.1 start

start方法用于启动线程，且只可调用一次。

### 1.2.2 stop

stop方法用于强制停止线程的运行。`不推荐使用。`原因是stop()方法过于暴力，强行把执行到一半的线程终止、释放持有的锁，这样`可能会引起一些数据不一致的问题`。

### 1.2.3 wait&notify

用于支持多线程之间的协作。这两个方法是Object类中的方法。

### 1.2.4 线程中断

先了解。

stop()方法停止线程过于保留，在jdk中有更强大的支持---->线程中断！

`线程中断并不会使线程立即退出，而是给线程发送一个通知，告知目标线程，有人希望你退出啦！至于目标线程接到通知后如何处理，则完全由目标线程自行决定。这点很重要，如果中断后，线程立即无条件退出，我们就又会遇到stop()方法的老问题。`

```java
public void interrupt() {
    if (this != Thread.currentThread()) {
        checkAccess();

        // thread may be blocked in an I/O operation
        synchronized (blockerLock) {
            Interruptible b = blocker;
            if (b != null) {
                interrupt0();  // set interrupt status
                b.interrupt(this);
                return;
            }
        }
    }

    // set interrupt status
    // native 方法
    interrupt0();
}
public static boolean interrupted() {
    // native 方法
    return currentThread().isInterrupted(true);
}
public boolean isInterrupted() {
    // native 方法
    return isInterrupted(false);
}
```

----

```java
public void interrupt() // 中断线程

public boolean Thread.isInterrupted() // 判断线程是否被中断
    
public static boolean Thread.interrupted() //  判断线程是否被中断，并清除当前中断状态
```

----

> interrupt方法

一个实例方法。它通知目标线程中断，也就是设置中断标志位。`仅仅是设置一个标志位~`并不会导致线程停止，想要线程停止可对标志位进行判断，然后进行其他操作

```java
public class InterruptDemo {
    public static void testInterrupt() throws InterruptedException {
        Thread thread = new Thread(() -> {
            while (true)
                Thread.yield();
        });
        thread.start();
        Thread.sleep(2000);
        // thread.interrupt() 仅仅只是设置中断标志位
        thread.interrupt();
    }

    public static void main(String[] args) throws InterruptedException {
        // 死循环。
        testInterrupt();
    }
}
```

> Thread.isInterrupted()方法

一个静态方法。判断当前线程是被设置了中断状态。所以我们可以对设置了中断状态的线程进行需要的操作，`如：当前线程被设置了中断状态，那么在某个时刻，我们就让线程退出执行！`

```java
import java.util.concurrent.TimeUnit;

    public static void testIsInterrupted() throws InterruptedException {
        Thread thread = new Thread(() -> {
            int count = 0;
            while (true) {
                count = (int) (Math.random() * 100_000);
                System.out.println(count);
                if (Thread.currentThread().isInterrupted() && count > 99_997) {
                    System.out.println("break current thread");
                    break;
                }
                // 放弃cpu执行权限
                Thread.yield();
            }
        });
        thread.start();
        thread.interrupt();
    }

    public static void main(String[] args) throws InterruptedException {
        testIsInterrupted();
    }
}
```

>Thread.interrupted()方法

判断线程是否被中断，并清除当前中断状态

----



