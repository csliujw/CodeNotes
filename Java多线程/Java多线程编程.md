# 概述

Java多线程编程。

- 线程的基本用法
- 线程池
- juc
- 死锁&常见错误加锁
- JMM
- ThreadLocal

## 几个概念

### 同步和异步

同步方法调用一旦开始，调用者必须等到方法调用返回后，才能继续后续的行为。异步方法调用更像一个消息传递，一旦开始，方法调用就会立即返回，调用者就可以继续后续的操作。

`多线程是异步的一种实现方式`

### 并发和并行

并发偏重于多个任务交替执行，而多个任务之间有可能还是串行的，而并行是真正意义上的“同时执行”

### 临界区

临界区用来表示一种公共资源或者说共享数据，可以被多个线程使用。但是每一次，只能有一个线程使用它，一旦临界区资源被占用，其他线程要想使用这个资源就必须等待。

如：打印机一次只能让一个人使用。

### 阻塞和非阻塞

一个线程占用了临界区资源，那么其他所有需要这个资源的线程就必须在这个临界区中等待。等待会导致线程挂起，这种情况就是阻塞。

非阻塞的意思与之相反，它强调没有一个线程可以妨碍其他线程执行，所有的线程都会尝试不断前向执行。

### 死锁、饥饿、活锁

死锁是多个线程彼此牵制都无法正常推进。

饥饿是指某一个或者多个线程因为种种原因无法获得所需要的资源，导致一直无法执行。

活锁是指资源不断地在两个线程间跳动，而没有一个线程可以同时拿到所有资源正常执行。

## 并发级别

阻塞、无饥饿、无障碍、无锁、无等待

### 阻塞

一个线程是阻塞的，那么在其他线程释放资源之前，当前线程无法继续执行

### 无饥饿

在一定时间内所有线程可以得到执行 [ 个人看法 ]

### 无障碍

无障碍：最弱的非阻塞调度。

大家都可以进入临界区，一起修改共享数据。若数据改坏了，会立即对自己所做的修改进行回滚，确保数据安全。`当临界区中存在严重的冲突时，所有的线程可能都会不断地回滚自己的操作，而没有一个线程可以走出临界区`

阻塞是一种悲观策略，非阻塞调度是一种乐观策略。

### 无锁

所有的线程都能尝试对临界区进行访问，但是无锁的并发保证必有一个线程能在有限步内完成操作离开临界区。

### 无等待

所有的线程都必须在有限步内完成。典型的有CopyOnWrite。可以多个线程同时读数据，这个就是无等待的。写数据时上锁，保证读写互斥，写写互斥。

## 定律

- Amdahl定律：`定义了串行系统并行化后的加速比的计算公式和理论上限`
    - 加速比 = 优化前系统耗时 / 优化后系统耗时
- Gustafson定律：`处理器个数、串行化比例和加速比之间的关系`

## JMM

Java的内存模型（JMM）

### 原子性

原子性是指一个操作是不可中断的，一气呵成。

### 可见性

当一个线程修改了某一个共享变量的值时，其他线程是否能够立即知道这个修改。volatile保证了可进行。

举例：4核CPU，每个核都有一个自己独立的缓存。这4个核共享一个变量。操作完数据后把数据放入自己核的缓存里，其他核不能及时得知修改。除非你把其他核的缓存也刷新了。

### 有序性

程序在执行时，可能会进行指令重排，重排后的指令与原指令的顺序未必一致。

提了下指令的执行步骤，指令流水线。

### 那些指令不能重排序

HappenBefore规则

- 程序顺序原则：一个线程内保证语义的串行性。
- volatile规则：volatile变量的写先于读发生，这保证了volatile变量的可见性。
- 锁规则：解锁（unlock）必然发生在随后的加锁（lock）前。
- 传递性：A先于B，B先于C，那么A必然先于C。
- 线程的start()方法先于它的每一个动作。
- 线程的所有操作先于线程的终结（Thread.join()）。
- 线程的中断（interrupt()）先于被中断线程的代码。
- 对象的构造函数的执行、结束先于finalize()方法。

# 线程基础

## 线程的所有状态

Java中的。

```java
 public enum State {
	// 表示刚刚创建的线程，这种线程还没开始执行。等到线程的start()方法调用时，才表示线程开始执行
        NEW,
	// 线程所需的一切资源都已经准备好了
        RUNNABLE,
	// 线程阻塞，暂停执行
        BLOCKED,
	// WAITING会进入一个无时间限制的等待
        WAITING,
	// TIMED_WAITING会进行一个有时限的等待
        TIMED_WAITING,
	// 当线程执行完毕后，则进入TERMINATED状态，表示结束。
        TERMINATED;
    }
```

`PS：从NEW状态出发后，线程不能再回到NEW状态，同理，处于TERMINATED状态的线程也不能再回到RUNNABLE状态。`

## 基本API

### 创建线程

- 继承Thread
- 实现Runnable





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



#  二、特别篇

## 常见API

### start与run的区别

- start是异步调用，是另外一个线程执行噢，start内部会调用run方法，但是run的调用时机未知。
- run是同步调用，是同一个线程执行噢。
- `jdk源码中，只要当前线程的状态不是以0状态进行调用start，都会抛出IllegalThreadStateException异常，所以start只能调用一次，多次调用会抛出异常`
  - 第一次调用后状态改变，不是0（NEW）了，第二次调用就抛出异常了！

```java
public class StartAndRun {
    public static void main(String[] args) {
        StartAndRunThread startAndRunThread = new StartAndRunThread();
        Thread thread = new Thread(startAndRunThread);
        thread.start();
    }
}

class StartAndRunThread implements Runnable{
    @Override
    public void run() {
        System.out.println("I am run!");
        System.out.println("当前线程为："+Thread.currentThread().getName());
    }
}

// out put
I am run!
当前线程为：Thread-0

// 调用run 将上述代码xxx.start()改成xxx.run()即可
I am run!
当前线程为：main
```

### resume与suspend

- 暂停线程用 `suspend()`
- 恢复线程用`resume()`
- `两个 API 是过期的，也就是不建议使用的。`
  - suspend() 在导致线程暂停的同时，并不会去释放任何锁资源。其他线程都无法访问被它占用的锁。
  - 对应的线程执行 resume() 方法后，被挂起的线程才能继续，从而其它被阻塞在这个锁的线程才可以继续执行。
  - 如果 resume() 操作出现在 suspend() 之前执行，那么线程将一直处于挂起状态，同时一直占用锁，这就产生了死锁。`而且，对于被挂起的线程，它的线程状态居然还是 Runnable，不好排查问题！！`

都是独占资源的，可能会导致死锁。都是过期方法，不推荐使用。线程的暂停恢复推荐使用：

- `wait`
- `notify`
- `notifyAll`

### wait与notify

一种线程通信机制。

notify和wait都只能在`synchronzied`修饰的代码中使用。且在使用前都必须要保证已经获取到锁！获取到了锁才能执行notify/wait方法

wait（）使当前指向wait()方法的线程等待！在wait所在的代码行处暂停执行并释放锁。直到接到通知或被中断。

notify释放锁需要执行完synchronized修饰的代码才会释放锁！

notifyAll唤醒线程的顺序是未知的（随机的）

wait(long)等待long这么长的时间，在继续执行。注意，wait(long)也是会释放锁的，在过来long时间后，它想继续执行需要重新获得锁，才可以。

```java
// 同一把锁内 才可以实现wait/notify相互通信。简而言之，只能唤醒和暂停统一把锁内的线程，不同锁的线程的暂停唤醒互不影响！
public class WaitAndNotify {
    public static void main(String[] args) {
        Object lock = new Object();
        Thread th1 = new Thread(() -> {
            synchronized (lock) {
                try {
                    System.out.println(Thread.currentThread().getName() + "wait start");
                    lock.wait();
                    System.out.println(Thread.currentThread().getName() + "wait end");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, "th1");

        Thread th2 = new Thread(() -> {
            synchronized (lock) {
                try {
                    System.out.println(Thread.currentThread().getName() + "wait start");
                    lock.wait();
                    System.out.println(Thread.currentThread().getName() + "wait end");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, "th2");

        Thread th3 = new Thread(() -> {
            synchronized (lock) {
                try {
                    System.out.println(Thread.currentThread().getName() + "notify start");
                    lock.notify();
                    System.out.println(Thread.currentThread().getName() + "notify end");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, "th3");
        th1.start();
        th2.start();
        th3.start();
    }
}
// out put notify的唤醒顺序就是先进先出【队列！】
th1wait start
th3notify start
th3notify end
th2wait start
th1wait end
```

### 停止线程

- 退出标志
- stop强行停止，不推荐。强行停止可能会有数据不安全的问题。线程执行了一半，还没做完所有操作就中断了。
  - 如 数组新增元素，添加了，但是size还没++，就stop了，数据就有问题了！
- interrupt中断线程【只是设置一个中断的标记，非立即中断】
  - <span style="color:green">在一个线程中调用另一个线程的interrupt()方法，即会向那个线程发出信号——线程中断状态已被设置。我们可以通过判断这个标记确定线程需不需要被中断，至于何时中断就由我们自己写代码决定了！</span>

```java
// interrupt示例 不好测。。cpu太快了
// 测试内容解释：
run方法里打印数据是为了证明，我调用了interrupt中断线程，线程仍会运行，仍会继续打印数据i。
我们重写了interrupt方法，在里面输出了一句话也是为了告诉自己，已经调用了中断方法。通过这两种输出结果的
顺序，推断是否是立即中断。
public class Interrupt {
    public static void main(String[] args) throws Exception {
        InterruptThread th1 = new InterruptThread();
        th1.start();
        th1.interrupt();
    }
}

class InterruptThread extends Thread {
    @Override
    public void interrupt(){
        super.interrupt();
        System.out.println("interrupt 执行了======================");
    }
    @Override
    public void run() {
        for (int i = 0; i < 200000; i++) {
            System.out.println("i=" + (i + 1));
        }
    }
}
```

-----

```text
interrupt 执行了====================== 调用interrupt线程并未停止！
i=138008
........
i=199998
i=199999
i=200000

Process finished with exit code 0
```

#### 如何判断线程是否中止？

查看Java Thread类源码可知有如下两种方法

- 静态方法 interrupted，因为是静态的，所以判断的是`currentThread，`即当前线程是否被标记为中断状态！
- 非静态 `isInterrupted`，判断的是运行 `this.isInterrupted` 这个方法的线程是否停止

```java
public static boolean interrupted() {
    return currentThread().isInterrupted(true);
}

public boolean isInterrupted() {
    return this.isInterrupted(false);
}
```

测试代码及解释

```java
public class JugeInterrupted {

    public static void main(String[] args) throws InterruptedException {
        System.out.println(Thread.currentThread().getId()); // main线程的id是1
        JugeThread thread = new JugeInterrupted().new JugeThread();
        thread.start();
        Thread.sleep(800);
        thread.interrupt(); // 中断线程，中断的是thread类 运行run方法的那个线程，即中断的id为14的线程。
        System.out.println("over");
        System.out.println(thread.interrupted());
        System.out.println(thread.interrupted());
    }
    class JugeThread extends Thread{
        @Override
        public void run(){
            for (int i = 0; i <200000 ; i++) {
                // Thread-0线程的id是14
                System.out.println(i+":"+this.isInterrupted()+":"+this.getId());
            }
        }
    }
}
```

输出结果分析

```java
thread.interrupt(); // 中断线程，将中断状态标记为true，可通过代码测试
System.out.println("over"); // 随意打印一句话
System.out.println(thread.interrupted()); // 查看中断状态 返回true，同时将状态设置为false
171644:false:14
171645:false:14
over
false
171646:true:14
false
171647:true:14
171648:true:14
    
// 源码
public static boolean interrupted() {
    return currentThread().isInterrupted(true);
}
```

`thread.interrupt()`中断的是运行thread run方法的那个线程，可以从输出结果看出。调用前后的输出结果为

`171645:false:14`
`over`
`false`
`171646:true:14`
`false`
`171647:true:14`

`thread.interrupted()`查看的是当前线程是否被中断，查看源码

```java
// 源码
public static boolean interrupted() {
return currentThread().isInterrupted(true);
}
// 我们的代码
public static void main(String[] args) throws InterruptedException {
    System.out.println(Thread.currentThread().getId()); // main线程的id是1
    JugeThread thread = new JugeInterrupted().new JugeThread();
    thread.start();
    Thread.sleep(800);
    thread.interrupt(); 
    System.out.println("over");
    System.out.println(thread.interrupted());//false
    System.out.println(thread.interrupted());//false
}
// 当前线程是main线程 我们并未对main线程进行状态标记，所以输出结果是false false
// 再看下面的代码
class JugeThread extends Thread{
    @Override
    public void run(){
        for (int i = 0; i <200000 ; i++) {
            // Thread-0线程的id是14 这块的输出结果改变了！因为这里的当前线程是Thread-0
            System.out.println(i+":"+this.isInterrupted()+":"+this.getId());
        }
    }
}

171645:false:14
171646:true:14
false
171647:true:14
```

<span style="color:red">总而言之，`this.isInterrupted()`是判断this所在的类对象，运行这个对象的线程是否已经标记为中断。</span>

#### 如何终止线程？

上面判断线程是否终止给了我们提示，我们可以在run方法中获取“是否需要被中断”，来决定是否停止线程。停止线程可采用抛出异常的方法进行终止！

可以用return，但是抛出异常的写法，方便统一管理日志！

```java
public class StopThread extends Thread {
    public static void main(String[] args) throws InterruptedException {
        System.out.println(Thread.currentThread().getName()); // 当前线程为main线程
        StopThread stopThread = new StopThread();
        stopThread.start();
        // 设置休眠 让run方法运行一会，中断方法晚一点执行
        Thread.sleep(200);
        // 设置为中断状态
        stopThread.interrupt();
    }

    @Override
    public void run() {
        // 当前线程为 Thread-0线程
        System.out.println(this.getName());
        try {
            for (int i = 0; i < 200000; i++) {
                System.out.println("hello!");
                if (this.isInterrupted()) {
                    throw new InterruptedException();
                }
            }
        } catch (InterruptedException e) {
            System.out.println("終於結束了！");
            e.printStackTrace();
        }
    }
}
```

#### 到底该怎样终止线程？

* stop方法可暴力终止线程，但是可能会使一些清理性工作无法完成！造成数据不完整！
* 而interrupt可以在run中进行逻辑判断，需要中断了，在抛出中断异常之前，把扫尾工作完成！
* 总而言之 推荐 interrupt + throw excetion的组合方式

 ### `getName()`分析

- `Thread.currentThread().getName(）`获取当前正在运行的线程的name
- `this.getName()`获取当前对象的名字（这个对象继承了Thread类，有自己的名字！）

`Thread.currentThread().getName()与this.getName()的区别分析`

先看一个例子

```java
public class DistinguishGetName {
    public static void main(String[] args) {
        DistinguishThread name = new DistinguishThread();
        name.start();
    }
}

class DistinguishThread extends Thread{

    @Override
    public void run(){
        System.out.println("this.getName():"+this.getName());
        System.out.println("Thread.currentThread().getName():"+Thread.currentThread().getName());
    }

}
// out put
this.getName():Thread-0
Thread.currentThread().getName():Thread-0
```

```java
public class DistinguishGetName {
    public static void main(String[] args) {
        DistinguishThread name = new DistinguishThread();
        Thread thread = new Thread(name);
        thread.setName("new name");
        thread.start();
    }
}

class DistinguishThread extends Thread{

    @Override
    public void run(){
        System.out.println("this.getName():"+this.getName());
        System.out.println("Thread.currentThread().getName():"+Thread.currentThread().getName());
    }

}

// out put   name不一样！！！原因是什么？
this.getName():Thread-0 
Thread.currentThread().getName():new name
```

`this.getName`获得的是当前对象的name，默认为Thread-number

`Thread.currentThread.getName`获得的是执行线程的name thread是执行线程，thread.setName("new name");

简而言之：this是当前对象，Thread.currentThread是当前线程！

要想修改this线程的名字 这样即可【注意需要继承Thread才可！实现Runnable不行，因为没有get/setName方法】

```java
public class DistinguishGetName {
    public static void main(String[] args) {
        DistinguishThread name = new DistinguishThread("new name");
        Thread thread = new Thread(name);
        thread.setName("new name");
        thread.start();
    }
}

class DistinguishThread extends Thread{
    public DistinguishThread(String threadName){
        super(threadName);
    }

    @Override
    public void run(){
        System.out.println("this.getName():"+this.getName());
        System.out.println("Thread.currentThread().getName():"+Thread.currentThread().getName());
    }

}
// out put  
this.getName():new name
Thread.currentThread().getName():new name
```

### 其他API

isAlive() ==> 是否存活
sleep(long millis) ==> 休眠指定毫秒
sleep(long millis, int nanos) ==> 休眠指定毫秒+纳秒  

## 常见对比分析

### join(long)与sleep(long)

x.join(long)方法的内部用的是wait来实现的。当线程x执行完long毫秒后，会调用wait释放锁。<br/>
sleep则是通过synchronized来实现的，不会释放锁。

这个可以深究一下

```java
// 网上的解释 是 this 指的是main线程。传入参数为0时，main线程不断放弃锁，想办法让one执行
public final synchronized void join(long millis) throws InterruptedException {
    long base = System.currentTimeMillis();
    long now = 0L;
    if (millis < 0L) {
        throw new IllegalArgumentException("timeout value is negative");
    } else {
        if (millis == 0L) {
            // 如果this所在的线程如果Alive 则一直释放锁，让其他线程执行
            while(this.isAlive()) {
                this.wait(0L); //理解释放锁？
            }
        } else {
            while(this.isAlive()) {
                long delay = millis - now;
                if (delay <= 0L) {
                    break;
                }
                this.wait(delay); // 等待 delay毫秒后释放锁
                now = System.currentTimeMillis() - base;
            }
        }
    }
}
```

### println的安全问题

```java
/**
 * @author payphone
 * @date 2020/6/12 19:15
 * @Description system.out.println()
 */
public class Other {
    public static void main(String[] args) {
        int i = 10;
        // 非线程安全 只会打印方法，i自减这个操作不在锁的范围内
        System.out.println(--i);
        // 非线程安全 先进入打印方法，i自减这个操作不在锁的范围内
        System.out.println(i--);
    }
}

// System.out.println源码
public void println(int x) {
    synchronized(this) {
        this.print(x);
        this.newLine();
    }
}
```

### wait与sleep

**1、来自不同的类**

wait==>Object

sleep==>Thread

**2、关于锁的释放**

wait会释放锁，sleep是不会释放锁。

**3、使用范围不同**

wait必须在同步代码块中

sleep可以在任何地方使用

### Lock与synchronized

> **Lock锁和synchronized的区别**

1、synchronized 内置的Java关键字，Lock是一个类

2、synchronized无法判断获取锁的状态，Lock可以判断是否获取到了锁

3、synchronized可以自动释放锁，Lock必须要手动释放锁，如果不释放，可能会死锁。

4、Synchronized 线程 1（获得锁，阻塞）、线程2（等待，傻傻的等）；Lock锁就不一定会等待下 去； 

> tryLock 是防止自锁的一个重要方式。tryLock()方法是有返回值的，它表示用来尝试获取锁，如果获取成功，则返回true，如果获取失败（即锁已被其他线程获取），则返回false，这个方法无论如何都会立即返回。在拿不到锁时不会一直在那等待。

5、Synchronized 可重入锁，不可以中断的，非公平；Lock ，可重入锁，可以 判断锁，非公平（可以 自己设置）；

> 什么是 “可重入”，可重入就是说某个线程已经获得某个锁，可以再次获取锁而不会出现死锁。

 6、Synchronized 适合锁少量的代码同步问题，Lock 适合锁大量的同步代码！

> **锁是什么，如何判断锁的是谁**

“ 锁 ” 的 本 质 其 实 是 `monitorenter` 和 `monitorexit` 字 节 码 指 令 的 一个 Reference 类 型 的 参 数 ， 即 要 锁 定 和 解 锁 的 对 象 。 我 们 知 道 ， 使 用Synchronized 可 以 修 饰 不 同 的 对 象 ， 因 此 ， 对 应 的 对 象 锁 可 以 这 么 确定 。

查看synchronized的汇编代码，可以发现，synchronized会被解释为`monitorenter` 和`monitorexit` 

### 一篇优秀的博客

**为什么wait,notify和notifyAll要与synchronized一起使用？**

`Object.wait(),Object.notify(),Object.notifyAll()`都是Object的方法，换句话说，就是每个类里面都有这些方法。

`Object.wait()`：释放当前对象锁，并进入阻塞队列
`Object.notify()`：唤醒当前对象阻塞队列里的任一线程（并不保证唤醒哪一个）
`Object.notifyAll()`：唤醒当前对象阻塞队列里的所有线程
**为什么这三个方法要与synchronized一起使用呢？解释这个问题之前，我们先要了解几个知识点**

==每一个对象都有一个与之对应的监视器==
===每一个监视器里面都有一个该对象的锁和一个等待队列和一个同步队列==
**wait()方法的语义有两个，一是释放当前对象锁，另一个是进入阻塞队列**，可以看到，这些操作都是与监视器相关的，当然要指定一个监视器才能完成这个操作了

notify()方法也是一样的，用来唤醒一个线程，你要去唤醒，首先你得知道他在哪儿，所以必须先找到该对象，也就是获取该对象的锁，当获取到该对象的锁之后，才能去该对象的对应的等待队列去唤醒一个线程。值得注意的是，只有当执行唤醒工作的线程离开同步块，即释放锁之后，被唤醒线程才能去竞争锁。

notifyAll()方法和notify()一样，只不过是唤醒等待队列中的所有线程

**因wait()而导致阻塞的线程是放在阻塞队列中的，因竞争失败导致的阻塞是放在同步队列中的，notify()/notifyAll()实质上是把阻塞队列中的线程放到同步队列中去**

为了便于理解，你可以把线程想象成一个个列车，对象想象成车站，每一个车站每一次能跑一班车，这样理解起来就比较容易了。

> 值得提的一点是，synchronized是一个非公平的锁，如果竞争激烈的话，可能导致某些线程一直得不到执行。

版权声明：本文为CSDN博主「Crazy丶Mark」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_39907763/article/details/79301813

## ThreadLocal

### 简述

我们软件复杂度设计的核心往往是数据的不一致。我们要保证单一数据源，确保数据保持一致，避免数据不一致带来的各种问题。典型的是并发问题！

保持数据的一致性有很多种方法：加锁，数据只读，统一更改等【操作系统&组成原理中有相应策略】

一致性问题的解决办法

- 排队（例如：锁、互斥量、管程）
- 投票（例如：Paxos，Raft等）

初中级 了解基本概念、原理（再别人做好的基础上进行开发）

高级 应对不同场景、正确使用、到达预期结果（了解每种数据结构合适的使用场景，理解为什么要这样用）

高专 深度掌握原理、本质，对不满足自身需求的可以进行改进，扩展（为什么要有这种数据结构，为什么要这样实现）

### 开篇

ThreadLocal，线程局部变量，每个线程都绑定了自己的值，互不干扰。

场景介绍。【每个线程都会有自己的ThreadLocal，不同的线程访问的ThreadLocal是不一样的！】

- 资源持有
- 线程安全
- 线程一致
- 并发计算

### 实现原理



### API用法

- set

- get

- `initialValue`.在获取时，若发现没有数据，则会调用`initalValue`方法,`initalValue`方法源码如下：

  ```java
  protected T initialValue() {
      return null;
  }
  ```

- ThreadLocal的内部结构如下：

- ThreadLocal

  - ThreadLocalMap：静态内部类，多线程共享。为什么是安全的？

    ```java
    static class ThreadLocalMap {
    
            /**
             * The entries in this hash map extend WeakReference, using
             * its main ref field as the key (which is always a
             * ThreadLocal object).  Note that null keys (i.e. entry.get()
             * == null) mean that the key is no longer referenced, so the
             * entry can be expunged from table.  Such entries are referred to
             * as "stale entries" in the code that follows.
             */
            static class Entry extends WeakReference<ThreadLocal<?>> {
                /** The value associated with this ThreadLocal. */
                Object value;
    
                Entry(ThreadLocal<?> k, Object v) {
                    super(k);
                    value = v;
                }
            }
    }
    ```

    

### 使用场景

> 保证数据一致

工具 apache2-util

```java
static Integer c = 0;

@RequestMapping("/start")
public Integer stat(){
    retur c;
}

@RequestMapping("/add")
public Integer add(){
    c++;
    return c;
}
```

在并发场景中，数据不一致。

ThreadLocal可以保证数据安全的操作，但是不同线程的ThreadLocal是不一样的。因为一个线程可能有多个ThreadLocal变量，所以需要一个数据结构进行维护这些ThreadLocal，Map是一个不错的选择。每个ThreadLocal内部都持有一个ThreadLocalMap静态类，一个线程，ThreadLocalMap只会被实例化一次，所以可以做到维护多个ThreadLocal。

如何解决？

让每个线程进行安全的计算，在获取的时候，获取所有线程的值进行累加！我们用ThreadLocal安全的操作每个线程处理的变量，在返回值的时候，对所有ThreadLocal进行累加。可以采用静态的Set集合，存储所有ThreadLocal中的值，但是，在Set存入值的时候可能会出现并发问题，Set.add()方法需要进行加锁。

从大面积的加锁，转向了小面积的加锁！

- 完全避免同步（难）
- 缩小同步范围（简单）+ThreadLocal解决问题

> MyBatis中的ThreadLocal

`MyBatis`的`SqlSessionFactor`是将连接存入到`ThreadLocal`中，获取也是直接从`ThreadLocal`中获取，从而保证拿到的是同一个连接。

## LockSupport

- 为什么要学：
  - JUC--->AQS--->(前置知识。可重入锁、LocSupport)
  - AQS是jdk自带的。
- 是什么
  - 用于创建锁和其他同步类的基本线程阻塞原语。所以它的主要作用就是挂起和唤醒线程，是创建锁和其他同步工具类的基础。
  - 线程等待唤醒机制（wait/notify）
- `LockSupport`类与每个使用它的线程都会关联一个许可证，默认情况下调用`LockSupport`类是不持有许可证的。`LockSupport`是使用Unsafe类实现的。
- 存在的方法
  - `void park`：<span style="color:green">申请拿许可证，拿不到就阻塞。（阻塞线程）</span>
  - `void unpark`：如果参数thread线程没有持有thread与`LockSupport`类关联的许可证，则让thread线程持有。如果thread之前因调用park而被挂起，则调用`unpark`后会被唤醒。<span style="color:green">简单说就是给你许可证；（解除阻塞线程）</span>
- `LockSupport`类使用许可这种概念来做到阻塞和唤醒线程的功能，<span style="color:green">许可（Permit）只有两个值1和0，默认是0</span>

`LockSupport`的通知可以在阻塞之前，因为他是按许可证的数量来决定阻塞还是不阻塞的。故可以先唤醒后等待。且`Park`无需锁化。归根结底，`LockSupport`调用的是`Unsafe`的`native`方法

`ReentrantLock`和基本的wait，notify则不是这样。他们只能先有等待的线程，然后唤醒等待的线程。

<span style="color:red">这种part拿许可证用的是轮询的方式看是否可以拿到锁吗？</span>

### 线程阻塞唤醒的方法

- 使用`Object`中的 `wait()`方法让线程等待，使用Object中的`notify()`方法唤醒线程
- 使用`JUC`包中`Condition`的`await()`方法让线程等待，使用`signal()`方法唤醒线程
- `LockSupport`类可以阻塞当前线程以及唤醒指定被阻塞的线程

-----

`RenntrantLock`是可重入锁，有多少lock就要有多少unlock，不然会无法释放锁。

`RenntrantLock`的`Condition`不满足条件时会进入该条件的阻塞队列。

```java
ReentrantLock lock = new ReentrantLock();
Condition empty = lock.newCondition();
Condition full = lock.newCondition();
empty.await();//当前线程进入empty对象的阻塞队列
full.await();//当前线程进入full对象的阻塞对立

// 具体的测试代码，可运行的。debug看每个条件阻塞队列中有那些线程就可以了~
public void test3() throws InterruptedException {
    ReentrantLock lock = new ReentrantLock();
    Condition empty = lock.newCondition();
    Condition full = lock.newCondition();

    Thread t1 = new Thread(() -> {
        try {
            lock.lock();
            empty.await();
        } catch (Exception e) {
        } finally {
            lock.unlock();
        }
    });

    Thread t2 = new Thread(() -> {
        try {
            lock.lock();
            empty.await();
        } catch (Exception e) {
        } finally {
            lock.unlock();
        }
    });

    Thread t3 = new Thread(() -> {
        try {
            lock.lock();
            empty.await();
        } catch (Exception e) {
        } finally {
            lock.unlock();
        }
    });


    Thread t4 = new Thread(() -> {
        try {
            lock.lock();
            full.await();
        } catch (Exception e) {
        } finally {
            lock.unlock();
        }
    });


    Thread t5 = new Thread(() -> {
        try {
            lock.lock();
            full.await();
        } catch (Exception e) {
        } finally {
            lock.unlock();
        }
    });

    t1.start();
    t2.start();
    t3.start();
    t4.start();
    t5.start();

    TimeUnit.SECONDS.sleep(10);

    System.out.println("123");
    for (; ; ) {

    }
}
```

## AQS

> `AQS（AbstractQueuedSynchronizer）`抽象队列同步器

### 前置知识

- 公平锁非公平锁
- 可重入锁
- `LockSupport`
- 自旋锁
- 数据结构链表
- 模板设计模式

### 概念普及

技术解释：`AQS`是用来构建锁或者其它同步器组件的重量级基础框架及整个<span style="color:red">`JUC`体系的基石</span>，通过内置的FIFO队列来完成资源获取线程的排队工作，<span style="color:red">并通过一个int型变量表示持有锁的状态。</span>

<img src="D:/69546/Documents/JavaEE/pics/JavaStrengthen/juc/AQS01.png" styyle="float:left">

<span style="color:green">`AQS`是`JUC`内容中最重要的基石</span>

- 与`AQS`有关的

  - `ReentrantLock`

  - `CountDownLatch`

  - `ReentraantReadWriteLock`

  - `Semaphore`

  - `CyclicBarrier`

    ```java
    // 这几个里面都有一个内部类Sync 继承了AQS
    // 内部类 Sync
    abstract static class Sync extends AbstractQueuedSynchronizer {
        
    }
    ```

<span style="color:green">进一步理解锁和同步器的关系</span>

- 锁，面向锁的<span style="color:red">使用者</span>，用户层面的`API`
- 同步器，面向锁的<span style="color:red">实现者</span>，
  - 比如`DougLee`，提出统一规范并简化了锁的实现，屏蔽了同步状态管理、阻塞线程排队和通知、唤醒机制等。

### 作用

加锁会导致阻塞，有阻塞就需要排队，实现排队必然需要有某种形式的队列来进行管理。

----

抢到资源的线程直接处理业务逻辑，抢占不到资源的线程的必然涉及一种排队等候机制，抢占资源失败的线程继续去等待(类似办理窗口都满了，暂时没有受理窗口的顾客只能去候客区排队等候)，仍然保留获取锁的可能且获取锁流程仍在继续(候客区的顾客也在等着叫号，轮到了再去受理窗口办理业务）。

既然说到了<span  style="color:red">排队等候机制</span>，那么就一定会有某种队列形成，这样的队列是什么数据结构呢?

如果共享资源被占用，<span style="color:red">就需要一定的阻塞等待唤醒机制来保证锁分配</span>。这个机制主要用的是CLH队列的变体实现的，将暂时获取不到锁的线程加入到队列中，这个队列就是**AQS**的抽象表现。它将请求共享资源的线程封装成队列的结点(Node) ，**通过CAS、自旋以及LockSuport.park()的方式，维护state变量的状态，使并发达到同步的效果**。                 

<img src="D:/69546/Documents/JavaEE/pics/JavaStrengthen/juc/AQS01.png" styyle="float:left">

### AQS体系

<img src="D:/69546/Documents/JavaEE/pics/JavaStrengthen/juc/AQS02.png" styyle="float:left">

**AQS自身**

AQS = state + CLH队列

- AQS的int变量：

  - AQS的同步状态State成员变量

    ```java
    /**
     * The synchronization state.
     */
    private volatile int state;
    ```

  - 举例：银行办理业务的受理窗口状态；零就是没人，自由状态可以办理；大于等于1，有人占用窗口，等着去

- `AQS`的`CLH`队列

  - `CLH`队列（三个大牛的名字组成），为一个双向队列
  - 举例：银行侯客区的等待顾客

- 小结：

  - 有阻塞就需要排队，实现排队必然需要队列
  - `state变量+CLH`双端Node队列

**`AQS`的内部类Node**

- Node的int变量
  - Node的等待状态`waitState`成员变量；`volatile int waitStatus `
  - 举例：等候区其它顾客(其它线程)的等待状态；队列中每个排队的个体就是一个Node.

**`AQS`是怎么排队的：**是用`LockSupport.pork()`来进行排队的

<img src="D:/69546/Documents/JavaEE/pics/JavaStrengthen/juc/AQS03.png" styyle="float:left">

### 源码解析

建议看一下Java并发编程实战的第14章。

#### `ReentrantLock`原理

`ReentrantLock`的lock，unlock调用的是它内部那个继承了`AQS`的内部类`Sync`的方法。

`ReentrantLock`可设置是公平锁还是非公平锁，那两个类也是`ReentrantLock`的内部类。

```java
/**
 * Sync object for non-fair locks
*/
static final class NonfairSync extends Sync {
    private static final long serialVersionUID = 7316153563782823691L;
    protected final boolean tryAcquire(int acquires) {
        return nonfairTryAcquire(acquires);
    }
}
```

公平锁和非公平锁的区别在哪里？

```java
// 公平锁
static final class FairSync extends Sync {
    private static final long serialVersionUID = -3000897897090466540L;
    /**
     * Fair version of tryAcquire.  Don't grant access unless
     * recursive call or no waiters or is first.
     */
    @ReservedStackAccess
    protected final boolean tryAcquire(int acquires) {
        final Thread current = Thread.currentThread();
        int c = getState();
        if (c == 0) {
            // 公平锁，先来后到，看阻塞队列中是否有
            if (!hasQueuedPredecessors() &&
                compareAndSetState(0, acquires)) {
                setExclusiveOwnerThread(current);
                return true;
            }
        }
        else if (current == getExclusiveOwnerThread()) {
            int nextc = c + acquires;
            if (nextc < 0)
                throw new Error("Maximum lock count exceeded");
            setState(nextc);
            return true;
        }
        return false;
    }
}
```

```java
static final class NonfairSync extends Sync {
    private static final long serialVersionUID = 7316153563782823691L;
    protected final boolean tryAcquire(int acquires) {
        return nonfairTryAcquire(acquires);
    }
}

// nonfairTryAcquire的代码
/**
  * Performs non-fair tryLock.  tryAcquire is implemented in
  * subclasses, but both need nonfair try for trylock method.
  */
@ReservedStackAccess
final boolean nonfairTryAcquire(int acquires) {
    final Thread current = Thread.currentThread();
    int c = getState();
    if (c == 0) {
        if (compareAndSetState(0, acquires)) {
            setExclusiveOwnerThread(current);
            return true;
        }
    }
    else if (current == getExclusiveOwnerThread()) {
        int nextc = c + acquires;
        if (nextc < 0) // overflow
            throw new Error("Maximum lock count exceeded");
        setState(nextc);
        return true;
    }
    return false;
}
```

#### `ReentrantLock`源码

一步一步看下去【基于jdk11，和jdk8有出入~~】

```java
public void test1() {
    ReentrantLock lock = new ReentrantLock(true);
    lock.lock();
}
```

点进lock里面一看,调用了内部类Sync的一个acquire方法。因为我们默认用的`ReentrantLock`的非公平锁。所以看非公平锁的`acquire`的源码

```java
public void lock() {
    sync.acquire(1);
}
```

点进`acqurie`方法一看，直接跳到了`AQS`类里。

```java
public final void acquire(int arg) {
    if (!tryAcquire(arg) &&
        acquireQueued(addWaiter(Node.EXCLUSIVE), arg))
        selfInterrupt();
}
```

去`tryAcquire`方法里看具体代码。这代码不对劲！模板方法！！！ 落地的实现交给子类来处理。

我们真实调用的是子类`NonfairSync`的方法，所以我们要找子类`NonfairSync`的`tryAcquire`方法

```java
protected boolean tryAcquire(int arg) {
    throw new UnsupportedOperationException();
}
```

看下子类`NonfairSync`的`tryAcquire`方法

```java
protected final boolean tryAcquire(int acquires) {
    return nonfairTryAcquire(acquires);
}
```

点进`nonfairTryAcquire`方法一看

```java
@ReservedStackAccess
final boolean nonfairTryAcquire(int acquires) {
    final Thread current = Thread.currentThread();
    int c = getState();
    // 如果当前的锁还没被其他线程拿到的话，去拿锁。
    if (c == 0) {
        // CAS操作了
        if (compareAndSetState(0, acquires)) {
            // 设置当前线程独占
            setExclusiveOwnerThread(current);
            return true;
        }
    }//如果请求锁的线程和持有锁的线程是同一个线程。这不就是可重入锁吗
    else if (current == getExclusiveOwnerThread()) {
        int nextc = c + acquires;
        if (nextc < 0) // overflow
            throw new Error("Maximum lock count exceeded");
        setState(nextc);
        return true;
    }// 未能拿到锁
    return false;
}
```

拿不到锁的话就，return false，会执行&&后面的语句

```java
/**
获取处于独占模式，忽略中断。通过调用至少一次tryAcquire，成功后返回来实现。否则，线程将排队，可能会反复阻塞和解除阻塞，调用tryAcquire直到成功。此方法可用于实现方法Lock.lock。
*/
public final void acquire(int arg) {
    // 假设无法获取到锁，那么
    // tryAcquire返回的是false， 取反后就是true。然后就执行addWaiter方法了~
    if (!tryAcquire(arg) &&
        acquireQueued(addWaiter(Node.EXCLUSIVE), arg))
        selfInterrupt();
}
```

`addWaiter`方法，放入队列等待？

```java
addWaiter(Node.EXCLUSIVE)

/** Marker to indicate a node is waiting in exclusive mode */
/** 指示节点以排他模式等待的标记 */
static final Node EXCLUSIVE = null;
```

```java
/**
* Creates and enqueues node for current thread and given mode.
*
* @param mode Node.EXCLUSIVE for exclusive, Node.SHARED for shared
* @return the new node
*/
private Node addWaiter(Node mode) {
    Node node = new Node(mode);
	// 节点入阻塞队列操作
    for (;;) {
        // 这个是伪节点
        Node oldTail = tail;
        if (oldTail != null) {
            node.setPrevRelaxed(oldTail);
            // 比较并设置头，和jdk1.8比起来好像减小了粒度
            if (compareAndSetTail(oldTail, node)) {
                oldTail.next = node;
                return node;
            }
        } else {
            // 没有元素就初始化阻塞队列
            initializeSyncQueue();
        }
    }
}
```

## Synchronized

> 学习内容概述

- synchronized介绍及其原理剖析【字节码层面】
- 非线程安全问题
- volatile主要作用
- volatile和synchronized的区别

> `主要内容总结：（预览）`

synchronized关键字的主要作用是保证同一时刻，只有一个线程可以执行某一个方法，或是某一代码块，synchronized可以修饰方法及代码块。JDK1.5还是1.6后synchronized的效率得到了极大的提升【JVM优化了这个关键字】

synchronized的特征：

- 可见性：synchronized可以，volatile也可以确保变量可见【可见性就是，被那两个修饰的变量，会强制从公共内存读取变量的值。】
- 原子性：被同步的代码块在同一时间只有一个线程在执行
- 禁止代码重排序：synchronized可以，volatile也可以

> `使用场景总结：`

- 想要变量的值被修改时，让其他线程能取到最新的值时，就要对变量使用volatile
- 多个线程操作一个实例对象时，想要确保线程安全，可使用synchronized关键字

### synchronized同步方法

#### 线程不安全例子

并发访问可能会造成数据的`“脏读”`【读不到期待的数据】

共享的数据可能会有线程安全问题，非共享的数据不会出现线程安全问题！

如：一个实例对象的成员变量被多个线程访问，这样可能会出现线程安全问题。 

```java
// 要多试几次，不容易出错误结果！
public class UnSafeDemo extends Thread {
    public static int ticket = 100;

    public static void main(String[] args) {
        UnSafeDemo unSafeDemo = new UnSafeDemo();
        Thread th1 = new Thread(unSafeDemo);
        Thread th2 = new Thread(unSafeDemo);
        Thread th3 = new Thread(unSafeDemo);
        Thread th4 = new Thread(unSafeDemo);
        th1.start();
        th2.start();
        th3.start();
        th4.start();
    }

    @Override
    public void run() {
        while (true) {
            if (ticket > 0) {
                try {
                    Thread.sleep(50);
                    System.out.println("还剩" + ticket-- + "张票");
                } catch (InterruptedException e) {
                }
            } else {
                return;
            }
        }
    }
}

// out put
还剩2张票
还剩-1张票
还剩1张票
还剩0张票
```

####  synchronized原理

synchronized锁的是当前对象，或字节码对象！

<span style="color:red">方法级别的锁，采用了flag标记ACC_SYNCHRONIZED。如果方法设置了ACC_SYNCHRONIZED标记则执行线程先持有同步锁，再执行方法，方法完成时再释放锁！</span>

```java
 public static synchronized void say();
    descriptor: ()V
    flags: (0x0029) ACC_PUBLIC, ACC_STATIC, ACC_SYNCHRONIZED
    Code:
      stack=0, locals=0, args_size=0
         0: return
      LineNumberTable:
        line 11: 0
```

<span style="color:red">同步代码块的锁。使用的时monitorenter和monitorexit指令进行同步处理</span>

```java
public static void hello();
    descriptor: ()V
    flags: (0x0009) ACC_PUBLIC, ACC_STATIC
    Code:
      stack=2, locals=2, args_size=0
         0: new           #2                  // class java/lang/Object
         3: dup
         4: invokespecial #1                  // Method java/lang/Object."<init>":()V
         7: dup
         8: astore_0
         9: monitorenter
        10: getstatic     #3                  // Field java/lang/System.out:Ljava/io/PrintStream;
        13: ldc           #4                  // String hello
        15: invokevirtual #5                  // Method java/io/PrintStream.println:(Ljava/lang/String;)V
        18: aload_0
        19: monitorexit
        20: goto          28
        23: astore_1
        24: aload_0
        25: monitorexit // 有两个exit是怕 第一个无法退出！
        26: aload_1
        27: athrow  // 这边抛出异常了
        28: return
      Exception table:
         from    to  target type
            10    20    23   any
            23    26    23   any
      .......
```

> PS:`多个线程多个锁`

多个对象多个锁，不同锁的线程互不影响噢！

####  带锁和不带锁的

不带synchronized方法的运行不受带synchronized方法的影响！【两者是异步的！不必一前一后！不会因为A的阻塞，导致B无法运行】

带锁的和不带锁的他们之间的运行是异步的。带锁的方法在运行，突然带锁的方法CPU时间到了，给不带锁的方法了！这个不带锁的可以正常运行！因为他的运行不需要拿锁！请看代码！

```java
public class LockAndUnLock {

    static class Demo1 extends Thread {
        private LockAndUnLock obj;

        public Demo1(LockAndUnLock obj) {
            this.obj = obj;
        }

        @Override
        public void run() {
            super.run();
            obj.lock();
        }
    }

    static class Demo2 extends Thread {
        private LockAndUnLock obj;

        public Demo2(LockAndUnLock obj) {
            this.obj = obj;
        }

        @Override
        public void run() {
            super.run();
            obj.unlock();
        }
    }

    private synchronized void lock() {
        try {
            System.out.println("I am lock");
            Thread.sleep(5000);
            System.out.println("I am lock！ I am over！");
        } catch (InterruptedException e) {
        }
    }

    private void unlock() {
        System.out.println("I am unlock");
    }

    public static void main(String[] args) {
        LockAndUnLock lock = new LockAndUnLock();
        // 传入的同一个对象 要锁的话 也是锁这个传入的对象
        Demo1 demo1 = new Demo1(lock);
        Demo2 demo2 = new Demo2(lock);
        demo1.start();
        demo2.start();
    }
}

// out put
I am lock
I am unlock
I am lock！ I am over！
并不干扰不加锁方法的运行！
```

都有synchronized关键字的话

```java
// unlock方法进行修改 加synchronized
private synchronized void unlock() {
    System.out.println("I am unlock");
}

// out put 有先后顺序！ 先执行lock内的所有语句 再执行unlock的！
I am lock
I am lock！ I am over！
I am unlock
```

####  小结

- A线程先持有obj对象的Lock锁，B线程可异步调用obj对象的非synchronized方法
- A线程先持有obj对象的Lock锁，B线程调用obj对象的synchronized方法，需要等待，等到A线程释放锁，B拿到锁了才可以运行！这就是同步！
- **方法声明处添加synchronized锁的不是方法而是当前类的对象！**
- 静态方法声明处加了锁的话，锁的是字节码对象！【简而言之，这个类的所有实例用的都是同一把锁！】

> 脏读

在读取变量时，此值被其他线程更改过了！

####  synchronized锁重入

锁重入？啥意思？

意思就是：有obj对象有A B C三个加锁修饰的方法。A内部调用了B方法，B内部调用了C方法，C B A这三个方法都可以正常执行。因为A拿到了锁，这个锁可以给B用【可重入】，B拿到了A的锁，这个锁又可以给C用！【编译原理，方法调用？参数传递？】

```java
public class Resynchronized {
    private synchronized void one(){
        System.out.println("one");
        two();
    }

    private synchronized void two(){
        System.out.println("two");
        three();
    }

    private synchronized void three(){
        System.out.println("three");
    }

    public static void main(String[] args) {
        new Thread(()->{new Resynchronized().one();}).start();
    }

}
```

#### 锁重入支持继承的环境

子类的锁声明方法可以调用父类锁声明的方法，调用过程中不会出现阻塞！

#### 出现异常锁自动释放

synchronized方法调用过程中出现异常，是会释放锁的

注意：sleep()方法和suspend()方法被调用后不释放锁【这句话我不太明白什么意思？】

```java
//  线程挂起的demo，不会释放锁！
public class ReleaseLock extends Thread {

    public synchronized void say() {
        while (true) {
            System.out.println("say~~~~~~~~");
        }
    }

    public synchronized void hello() {
        System.out.println("hello~~~~~~~~~");

    }

    public static void main(String[] args) throws InterruptedException {
        ReleaseLock releaseLock = new ReleaseLock();
        
        Thread thread1 = new Thread(() -> {
            releaseLock.say();
        });
        
        thread1.start();
        Thread.sleep(2000);
        thread1.suspend();

        Thread thread2 = new Thread(() -> {
            releaseLock.hello();
        });
        thread2.start();
    }
}
```

#### 重写方法不使用synchronized

子类重写父类的synchronized方法，但是重写后的方法不用synchronized修饰，则重写后的方法调用时不用获取锁。简而言之，子类的重写方法无synchronized关键字，就不是同步方法！

```java
/**
 * @author payphone
 * @date 2020/6/16 16:35
 * @Description 子类重写父类的synchronized方法
 */
public class OverrideDemo {
    public static void main(String[] args) throws InterruptedException {
        // 多个线程操作同一个对象，如果有锁，则会顺序执行！
        Son son = new Son();
        Thread th1 = new Thread(() -> {
            son.say();
        });
        Thread th2 = new Thread(() -> {
            son.say();

        });

        th1.start();
        Thread.sleep(500);
        th2.start();
    }
}

class Father {
    public synchronized void say() {
    }
}

class Son extends Father {
    @Override
    public void say() {
        try {
            System.out.println(Thread.currentThread().getName()+":son");
            Thread.sleep(5000);
            System.out.println(Thread.currentThread().getName()+":over");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

// out put
Thread-0:son
Thread-1:son
Thread-0:over
Thread-1:over
```

输出结果时0和1交替运行，并不是得0运行完毕1才可运行。不是同步得。==>重写得方法不带synchronized就不是同步方法！当然，如果内部调用了相同的同步方法，则那个被调用的同步方法还是需要同步执行的，请看如下代码！

```java
public class OverrideDemo {
    public static void main(String[] args) throws InterruptedException {
        // 多个线程操作同一个对象，如果有锁，则会顺序执行！
        Son son = new Son();
        Thread th1 = new Thread(() -> {
            son.say();
        });
        Thread th2 = new Thread(() -> {
            son.say();

        });

        th1.start();
        Thread.sleep(500);
        th2.start();
    }
}

class Father {
    public synchronized void say() {
        try {
            System.out.println("I am your father："+Thread.currentThread().getName());
            Thread.sleep(10000);
            System.out.println("You need wait me finish："+Thread.currentThread().getName());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

class Son extends Father {
    @Override
    public void say() {
        super.say();
        try {
            System.out.println(Thread.currentThread().getName()+":son");
            Thread.sleep(5000);
            System.out.println(Thread.currentThread().getName()+":over");
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

// out put 先只执行完Thread0的再执行完Thread-1的。
I am your father：Thread-0
You need wait me finish：Thread-0
I am your father：Thread-1
Thread-0:son
Thread-0:over
You need wait me finish：Thread-1
```

### synchronized同步语句块

#### synchronized方法的弊端

方法内的代码需要执行的时间越长，锁的时间就越长，其他同步方法等待的时间就越久！【缩减同步方法的大小，只对必要的操作进行加锁！】若，多个操作都需要加锁，加多次锁不如只加一次大范围的锁。频繁的加锁解锁也会导致性能低下。

**可用同步代码块来替代synchronized方法！**

==同步代码块中的方法同步执行，非同步代码块中的方法异步执行！==

```java
public class LongCode {
    private int count = 100;

    public static void main(String[] args) {
        LongCode longCode = new LongCode();
        Thread th1 = new Thread(() -> {
            longCode.increase();
        });
        Thread th2 = new Thread(() -> {
            longCode.decrease();
            ;
        });
        Thread th3 = new Thread(() -> {
            longCode.decrease();
        });
        th1.start();
        th2.start();
        th3.start();
    }

    private void increase() {
        print();
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
        }
        synchronized (this) {
            System.out.println(Thread.currentThread().getName() + ":" + count++);
        }
        print();
    }

    private void decrease() {
        print();
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
        }
        synchronized (this) {
            System.out.println(Thread.currentThread().getName() + ":" + count++);
        }
        print();

    }

    public void print() {
        System.out.println("*******"+Thread.currentThread().getName()+"*******");
        System.out.println("*******"+Thread.currentThread().getName()+"*******");
    }
}
```

> synchronized同步代码块的一些细节

- synchronized使用的对象监视器是同一个，即使用的锁是同一个！

- synchronized(this)中的this是当前对象！

- 也可以使用任意的对象作为(obj)，只要保证对象是一致的就可以了。
- 方法之间的锁不同的话，它们之间的调用是异步的。

> 何时需要同步?

当多线程调用的方法之间有严格的执行顺序，则需要进行同步操作。

操作需要一气呵成，不能被其他程序插入执行的时候，需要锁住他！

####  synchronized各方面的比较

> this与obj.class

```java
synchronized(this) 是把当前对象当锁来用
synchronized(Obj.class) 是把字节码对象当锁来用。

    this!= obj.class
        

public class ThisAndClass {

    public static void main(String[] args) {
        Object o1 = new Object();
        Object o2 = new Object();
        System.out.println(o1==o1.getClass()); // false
        System.out.println(o2==o2.getClass()); // false
        System.out.println(o1.getClass()==o2.getClass()); // true
        System.out.println(Object.class==o1.getClass()); // true
    }
}

```

> synchronized（this）

锁的是当前实例对象的方法。

> synchronized(obj.class)

锁的是Obj字节码对象的方法。与this 不是同一把锁。

`obj.class`可以被多个实例对象使用。即，这个类的多个实例对象，都用字节码对象这把锁，多个对象共用一把锁。

> 静态与非静态修饰

```java
public static synchronized void say(){} // 这个是把字节码对象当锁【有static关键字】
public synchronized void say(){} // 这个是把当前实例对象当锁【普通方法】
```

## volatile关键字

**用volatile声明的变量**

- 可确保其他线程可见
  - JVM分线程工作内存和OS内存，把数据从OS内存读到线程工作内存后再操作变量，操作完后的某个时间（这个要看字节码指令的顺序），再把变量写回OS内存！这样会导致数据无法及时同步【A线程把数据拷贝到了工作内存，B线程也拷贝了一份，并且修改了，同步到了OS内存，但是A线程读的还是工作内存中的值，没有及时获得新的值！】，即其他线程无法立即得知数据是否改变！
  - 用volatile声明的，操作完数据后，会立即将数据写入OS内存，其他线程在要读取的时候可以立即得知数据是否变动了！
- 禁止指令重排序。
  - 多流水线技术，为了提高程序的并发运行的效率，内部会对字节码指令的顺序进行重排序，让代码高效运行。具体内容请自行看计算机组成原理流水线部分。

- 不能保证操对变量的操作是原子性操作！如：
  - volatile int i = 0； i++
    - 并非原子性操作
    - 读取变量到工作内存
    - 执行加一操作
    - 将数据重新写回OS内存
    - 有三步，这种非一气呵成的操作，是可能出现并发安全问题的

### 数据不同步造成的死锁

我用IDEA Java11 测试的是会出现死循环。如果不出现死循环，在JVM中添加运行参数！

VM argument/option里添加-server

```java
public class DeadLock extends Thread{
    private boolean isRunning = true;

    public boolean isRunning(){
        return isRunning;
    }

    public void setRunning(boolean isRunning){
        this.isRunning = isRunning;
    }

    @Override
    public void run() {
        super.run();
        System.out.println("线程开始了！");
        //while里面不能有任何代码 不知道原因，反正不能有代码
        // System.out.print这个IO操作让主线程能获取最新的isRunning
        while (isRunning){
        }
        System.out.println("线程停止了！");
    }

    // 解释。有两个线程 main线程和Thread-0线程。main线程修改了Thread-0线程的isRunning的值。
    // 修改的是OS内存中的那个值。Thread-0读取的是自己工作内存中的值【非OS内存中的】，所以仍然
    // 为trun，继续保持死循环。
    public static void main(String[] args) throws InterruptedException {
        DeadLock deadLock = new DeadLock();
        deadLock.start();
        Thread.sleep(1000);
        deadLock.setRunning(false);
        System.out.println("main线程的最后一句输出");
    }
}
// out put
线程开始了！
main线程的最后一句输出
```

### 用volatile解决上述死锁

```java
public class DeadLock extends Thread{
    private volatile boolean  isRunning = true;

    public boolean isRunning(){
        return isRunning;
    }

    public void setRunning(boolean isRunning){
        this.isRunning = isRunning;
    }

    @Override
    public void run() {
        super.run();
        System.out.println("线程开始了！");
        while (isRunning){
        }
        System.out.println("线程停止了！");
    }

    public static void main(String[] args) throws InterruptedException {
        DeadLock deadLock = new DeadLock();
        deadLock.start();
        Thread.sleep(1000);
        deadLock.setRunning(false);
        System.out.println("main线程的最后一句输出");
    }
}
// out put
线程开始了！
main线程的最后一句输出
线程停止了！
```

### synchronized即可同步又可保证可见性

synchronized修饰的代码，里面的变量即可保证可见性，又可保证同步！

### volatile与重排序

volatile修饰的变量

- volatile前面的代码不可重排序到它后面
- volatile后面的代码不可重排序到它前面
- 用数据结构排序中的一句话来概括，将代码分成三部分前，volatile，后，这三部分是稳定的。

### 为什么单例要用volatile

volatile可以防止指令重排序。防止对象的初始化赋值 重排序，先给对象赋值，再初始化对象这种操作，导致对象不为空，非synchronized方法判断它不会空可以获取了，实际上对象的值还没初始化！拿到的数据不正确！volatile可以避免这种现象！

但是一般不会写饿汉式，，除非你真的在乎这一个对象的内存！

> PS：volatile修饰引用，引用指向的对象中的值改变了是不能立即察觉的！

## 锁优化

- 减小锁粒度：竞争不激烈时，尽量减少锁粒度。
- 增加锁粒度：锁竞争激烈，意思就是很多小锁，频繁的加锁解锁很消耗资源，可以考虑将锁粗化。

## CAS

`（Compare and Swap)`

凡是Atomic开头的都是用CAS来保证线程安全的

CAS是CPU级别的原语，一气呵成，不能被中断！

sync(Object)

`markword` 记录这个线程ID（偏向锁）只记录ID，也没申请锁，所以效率很高。

如果有线程争用：升级为 自旋锁

10次后，升级为重量级锁

`Atomicl` 用的lock，而lock用的基本就是自旋锁。是在用户态解决问题，不是在系统态，没有状态切换的开销。

占用CPU时间短的用自旋锁好（就算时间短，太多的线程自旋也不好）；而占用CPU时间长的（执行时间长的）用系统锁（OS锁）。

执行时间短（加锁代码），线程数少，用自旋

执行时间长，线程数多，用系统锁。

Integer 中的值一旦发送改变，就会产生一个新对象！所以也别用他当syn的锁对象

---

Compare And Swap (Compare And Exchange) / 自旋 / 自旋锁 / 无锁 

因为经常配合循环操作，直到完成为止，所以泛指一类操作

cas(v, a, b) ，变量v，期待值a, 修改值b。期待v的值和a的值一样，一样则把值修改为b，如果不一样就一直轮询对比，一样时才修改值

ABA问题，你的女朋友在离开你的这段儿时间经历了别的人，自旋就是你空转等待，一直等到她接纳你为止

ABA问题一般是没啥影响的，不用管。

解决办法（版本号 AtomicStampedReference），基础类型简单值不需要版本号【这个功能比较鸡肋】

## Unsafe

AtomicInteger:

```java
public final int incrementAndGet() {
    	// CAS的具体使用场景， for循环死等，占用cpu资源，但是一般来说这种场景不用等待太久，比加锁减锁耗时要小。
        for (;;) {
            int current = get();
            int next = current + 1;
            if (compareAndSet(current, next))
                return next;
        }
    }

public final boolean compareAndSet(int expect, int update) {
        return unsafe.compareAndSwapInt(this, valueOffset, expect, update);
    }
```

Unsafe:

```java
public final native boolean compareAndSwapInt(Object var1, long var2, int var4, int var5);
```

运用：

```java
package com.mashibing.jol;

import sun.misc.Unsafe;

import java.lang.reflect.Field;

public class T02_TestUnsafe {

    int i = 0;
    private static T02_TestUnsafe t = new T02_TestUnsafe();

    public static void main(String[] args) throws Exception {
        //Unsafe unsafe = Unsafe.getUnsafe();

        Field unsafeField = Unsafe.class.getDeclaredFields()[0];
        unsafeField.setAccessible(true);
        Unsafe unsafe = (Unsafe) unsafeField.get(null);

        Field f = T02_TestUnsafe.class.getDeclaredField("i");
        long offset = unsafe.objectFieldOffset(f);
        System.out.println(offset);

        boolean success = unsafe.compareAndSwapInt(t, offset, 0, 1);
        System.out.println(success);
        System.out.println(t.i);
        //unsafe.compareAndSwapInt()
    }
}
```

jdk8u: unsafe.cpp:

cmpxchg = compare and exchange

```c++
UNSAFE_ENTRY(jboolean, Unsafe_CompareAndSwapInt(JNIEnv *env, jobject unsafe, jobject obj, jlong offset, jint e, jint x))
  UnsafeWrapper("Unsafe_CompareAndSwapInt");
  oop p = JNIHandles::resolve(obj);
  jint* addr = (jint *) index_oop_from_field_offset_long(p, offset);
  return (jint)(Atomic::cmpxchg(x, addr, e)) == e;
UNSAFE_END
```

jdk8u: atomic_linux_x86.inline.hpp

is_MP = Multi Processor  

```c++
inline jint     Atomic::cmpxchg    (jint     exchange_value, volatile jint*     dest, jint     compare_value) {
  int mp = os::is_MP();
  __asm__ volatile (LOCK_IF_MP(%4) "cmpxchgl %1,(%3)"
                    : "=a" (exchange_value)
                    : "r" (exchange_value), "a" (compare_value), "r" (dest), "r" (mp)
                    : "cc", "memory");
  return exchange_value;
}
```

jdk8u: os.hpp is_MP()

```c++
  static inline bool is_MP() {
    // During bootstrap if _processor_count is not yet initialized
    // we claim to be MP as that is safest. If any platform has a
    // stub generator that might be triggered in this phase and for
    // which being declared MP when in fact not, is a problem - then
    // the bootstrap routine for the stub generator needs to check
    // the processor count directly and leave the bootstrap routine
    // in place until called after initialization has ocurred.
    return (_processor_count != 1) || AssumeMP;
  }
```

jdk8u: atomic_linux_x86.inline.hpp

```c++
#define LOCK_IF_MP(mp) "cmp $0, " #mp "; je 1f; lock; 1: "
```

最终实现：

cmpxchg = cas修改变量值

```assembly
lock cmpxchg 指令
```

硬件：

lock指令在执行后面指令的时候锁定一个北桥信号

（不采用锁总线的方式）

## markword

class字节码默认是采用8位对齐的。不够8位的话需要补齐。

> markword 8个字节；class pointer 4个字节【压缩后】；默认开启了指针压缩。没有指针压缩的话，就不用花费4个字节对齐了，还是16个字节，算引用的话，共20个字节

new Object() 占16个字节 引用Object o占4个字节【指针压缩从8-->4】共20个字节

## 工具：JOL 

JOL = Java Object Layout

```xml
<dependencies>
    <!-- https://mvnrepository.com/artifact/org.openjdk.jol/jol-core -->
    <dependency>
        <groupId>org.openjdk.jol</groupId>
        <artifactId>jol-core</artifactId>
        <version>0.9</version>
    </dependency>
</dependencies>
```

jdk8u: markOop.hpp

```java
// Bit-format of an object header (most significant first, big endian layout below):
//
//  32 bits:
//  --------
//             hash:25 ------------>| age:4    biased_lock:1 lock:2 (normal object)
//             JavaThread*:23 epoch:2 age:4    biased_lock:1 lock:2 (biased object)
//             size:32 ------------------------------------------>| (CMS free block)
//             PromotedObject*:29 ---------->| promo_bits:3 ----->| (CMS promoted object)
//
//  64 bits:
//  --------
//  unused:25 hash:31 -->| unused:1   age:4    biased_lock:1 lock:2 (normal object)
//  JavaThread*:54 epoch:2 unused:1   age:4    biased_lock:1 lock:2 (biased object)
//  PromotedObject*:61 --------------------->| promo_bits:3 ----->| (CMS promoted object)
//  size:64 ----------------------------------------------------->| (CMS free block)
//
//  unused:25 hash:31 -->| cms_free:1 age:4    biased_lock:1 lock:2 (COOPs && normal object)
//  JavaThread*:54 epoch:2 cms_free:1 age:4    biased_lock:1 lock:2 (COOPs && biased object)
//  narrowOop:32 unused:24 cms_free:1 unused:4 promo_bits:3 ----->| (COOPs && CMS promoted object)
//  unused:21 size:35 -->| cms_free:1 unused:7 ------------------>| (COOPs && CMS free block)
```

> Synchronized: 关于锁的信息是在对象的markword，对象头中。

## 锁升级过程

无锁 -> 偏向锁 -> 轻量级锁 （自旋锁，自适应自旋）-> 重量级锁

偏向锁和轻量级锁用到了对象的内存布局中的对象头信息。

HotSpot对象头分为两部分信息：

- 存储对象自身的运行时数据
    - 哈希码
    - GC分代年龄
    - 这部分数据在32位和64位vm中分别为32bit和64bit。<span style="color:green">官方称为`mark word`【实现偏向锁和轻量级锁的关键】</span>
- 存储指向方法区对象类型数据的指针

---

对象头信息与对象自身定义的数据无关，属额外的存储成本。考虑VM的空间效率，Mark Word被设计为一个非固定数据结构以便在极小的空间内存储尽量多的信息，它会根据对象的状态复用自己的存储空间。

32bit hotspot vm中对象未被锁定的状态下，Mark Word的32bit空间中的25bit用于存储对象哈希码（HashCode），4bit用于存储对象分代年龄，2bit用于存储锁标志位，1bit固定为0，在其他状态（轻量级锁定、重量级锁定、GC标记、可偏向）

| 存储内容                             | 标志位 | 状态               |
| ------------------------------------ | ------ | ------------------ |
| 对象哈希码、对象分代年龄             | 01     | 未锁定             |
| 指向锁记录的指针                     | 00     | 轻量级锁定         |
| 指向重量级锁的指针                   | 10     | 膨胀（重量级锁定） |
| 空、不需要记录信息                   | 11     | GC标记             |
| 偏向线程ID、偏向时间戳、对象分代年龄 | 01     | 可偏向             |

用CAS更新Mark Word的标记位。

### 无锁

>`典型的无锁算法有CAS算法`

- 大多数的机器都支持CAS操作（CAS是原子性操作，硬件保证的）

- `CAS操作需要三个操作数`，内存位置（V表示）、旧的预期值（A表示）、新值（B表示）；CAS指令执行时，当且仅当V（`对应内存位置中的值`）符合旧预期值A时，处理器用新值B更新V的值。

- CAS操作只有启动类加载器（Bootstrap ClassLoader）加载的Class才能访问它。不采用反射的话，只能通过Java API间接使用

    ```java
    // AtomicInteger调用了CAS操作的代码
    public final int incrementAndGet() {
        return U.getAndAddInt(this, VALUE, 1) + 1;
    }
    
    @HotSpotIntrinsicCandidate // hot spot vm热点代码候选（JIT）
    public final int getAndAddInt(Object o, long offset, int delta) {
        int v;
        do {
            v = getIntVolatile(o, offset);
        } while (!weakCompareAndSetInt(o, offset, v, v + delta));
        return v;
    }
    ```

CAS有个ABA问题。准备赋值检测时是A，中途被改为了B，后来有被改回了A。

J.U.C.的`AtomicStampedReference`带有标记，可解决这个问题。

### 偏向锁

偏向锁：如果资源被加锁了，但是实际上没有资源竞争的情况，则==采用偏向锁，并且不会对该资源的访问进行加锁，解决开销。如果后期存在资源竞争则进行加锁。==

> 偏向锁有存在的意义吗？

Java类库的不少代码都加了锁，但是多数情况下他们是在一个线程中运行，不存在线程安全问题。这样仍进行加锁解锁的话，会导致效率低下，因此偏向锁的存在是有意义的！

`被加锁却无资源竞争，将重量级锁弱化为偏向锁！【JDK编译优化，不是锁降级】`

### 轻量级锁

<span style="color:green">轻量级锁的实现应该是这两种：自旋锁与自适应锁</span>

#### 自旋锁

挂起线程和恢复线程的操作都需要转入内核态中完成，这些操作给系统的并发性能带来了很大的压力。如果共享数据的锁定状态只会持续很短的一段时间，不放弃处理器的执行时间，看持有锁是否很快就会释放，为了让线程等待，只需要让线程执行一个忙循环（自旋）`-->自旋锁`

不加锁，采用自旋的方式，不停的访问看是否为期待值，这个过程叫自旋。如果自旋超过一定的次数则升级为重量级锁（synchronized）。

自旋等待本身虽然避免了线程切换的开销，但它是要占用处理器时间的，因此，如果锁被占用的时间很短，自旋等待的效果就会非常好，反之，如果锁被占用的时间很长，那么自旋的线程只会白白消耗处理器资源，而不会做任何有用的工作，反而会带来性能上的浪费。

自旋等待的时间必须要有一定的限度，超过指定自旋次数仍未获得锁应使用传统的方式去挂起线程。

自旋默认次数为10，可通过-XX:PreBlockSpin来更改。

#### 自适应锁

在JDK 1.6中引入了自适应的自旋锁。自适应意味着自旋的时间不再固定了，而是由前一次在同一个锁上的自旋时间及锁的拥有者的状态来决定。如果在同一个锁对象上，自旋等待刚刚成功获得过锁，并且持有锁的线程正在运行中，那么虚拟机就会认为这次自旋也很有可能再次成功，进而它将允许自旋等待持续相对更长的时间，比如100个循环。另外，如果对于某个锁，自旋很少成功获得过，那在以后要获取这个锁时将可能省略掉自旋过程，以避免浪费处理器资源。有了自适应自旋，随着程序运行和性能监控信息的不断完善，虚拟机对程序锁的状况预测就会越来越准确，虚拟机就会变得越来越“聪明”了。

#### 轻量级锁

JDK 1.6之中加入的新型锁机制。



synchronized优化的过程和markword息息相关；==synchronized在字节码层面被翻译为了类似管程一类的东西 类似于OS的P V操作。==

==用markword中最低的三位代表锁状态 其中1位是偏向锁位 两位是普通锁位==

1. Object o = new Object()
   锁 = 0 01 无锁态 

2. o.hashCode()
   001 + hashcode

   ```java
   00000001 10101101 00110100 00110110
   01011001 00000000 00000000 00000000
   ```

   little endian big endian 

   00000000 00000000 00000000 01011001 00110110 00110100 10101101 00000000

3. 默认synchronized(o) 
   00 -> 轻量级锁
   默认情况 偏向锁有个时延，默认是4秒
   why? <u>因为JVM虚拟机自己有一些默认启动的线程，里面有好多sync代码，这些sync代码启动时就知道肯定会有竞争，如果使用偏向锁，就会造成偏向锁不断的进行锁撤销和锁升级的操作，效率较低。</u>

   ```shell
   -XX:BiasedLockingStartupDelay=0
   ```

   `轻量级锁是<u>执行在用户态的</u>，本质是一个while循环，看自己的值和xx的值是否一致，如果不一样，就一致while直到一样了，就停止自旋，如果自旋次数超过指定次数，锁就升级了！CAS`

4. 如果设定上述参数
   new Object () - > 101 偏向锁 ->线程ID为0 -> Anonymous BiasedLock 
   打开偏向锁，new出来的对象，默认就是一个可偏向匿名对象101

5. 如果有线程上锁
   上偏向锁，指的就是，把markword的线程ID改为自己线程ID的过程
   偏向锁不可重偏向 批量偏向 批量撤销

6. ==如果有线程竞争==
   撤销偏向锁，升级轻量级锁
   线程在自己的线程栈生成LockRecord ，用CAS操作将markword设置为指向自己这个线程的LR的指针，设置成功者得到锁

7. 如果竞争加剧
   竞争加剧：有线程超过10次自旋， -XX:PreBlockSpin， 或者自旋线程数超过CPU核数的一半， 1.6之后，加入自适应自旋 Adapative Self Spinning ， JVM自己控制
   升级重量级锁：-> 向操作系统申请资源，linux mutex , CPU从3级-0级系统调用，线程挂起，进入等待队列，等待操作系统的调度，然后再映射回用户空间。**JVM线程模型现在采用的多对多的线程模型；其实还是要看运行在上面操作系统上！**

(以上实验环境是JDK11，打开就是偏向锁，而JDK8默认对象头是无锁)

偏向锁默认是打开的，但是有一个时延，如果要观察到偏向锁，应该设定参数



没错，我就是厕所所长

加锁，指的是锁定对象

锁升级的过程

JDK较早的版本 OS的资源 互斥量 用户态 -> 内核态的转换 重量级 效率比较低

现代版本进行了优化

无锁 - 偏向锁 -轻量级锁（自旋锁）-重量级锁



偏向锁 - markword 上记录当前线程指针，下次同一个线程加锁的时候，不需要争用，只需要判断线程指针是否同一个，所以，偏向锁，偏向加锁的第一个线程 。hashCode备份在线程栈上 线程销毁，锁降级为无锁

有争用 - 锁升级为轻量级锁 - 每个线程有自己的LockRecord在自己的线程栈上，用CAS去争用markword的LR的指针，指针指向哪个线程的LR，哪个线程就拥有锁

自旋超过10次，升级为重量级锁 - 如果太多线程自旋 CPU消耗过大，不如升级为重量级锁，进入等待队列（不消耗CPU）-XX:PreBlockSpin



自旋锁在 JDK1.4.2 中引入，使用 -XX:+UseSpinning 来开启。JDK 6 中变为默认开启，并且引入了自适应的自旋锁（适应性自旋锁）。

自适应自旋锁意味着自旋的时间（次数）不再固定，而是由前一次在同一个锁上的自旋时间及锁的拥有者的状态来决定。如果在同一个锁对象上，自旋等待刚刚成功获得过锁，并且持有锁的线程正在运行中，那么虚拟机就会认为这次自旋也是很有可能再次成功，进而它将允许自旋等待持续相对更长的时间。如果对于某个锁，自旋很少成功获得过，那在以后尝试获取这个锁时将可能省略掉自旋过程，直接阻塞线程，避免浪费处理器资源。

偏向锁由于有锁撤销的过程revoke，会消耗系统资源，所以，在锁争用特别激烈的时候，用偏向锁未必效率高。还不如直接使用轻量级锁。

## synchronized vs Lock (CAS)

```
 在高争用 高耗时的环境下synchronized效率更高
 在低争用 低耗时的环境下CAS效率更高
 synchronized到重量级之后是等待队列（不消耗CPU）
 CAS（等待期间消耗CPU）
 
 一切以实测为准
```

> 参考资料

http://openjdk.java.net/groups/hotspot/docs/HotSpotGlossary.html

----

JVM是解释执行，翻译一句执行一句。

JIT，直接把热点代码编译成机器语言，不在解释，提高效率。

缓存对齐，访问速度会更快！NIO框架 netty采用的策略，高版本Java可用注解直接实现缓存行对齐。

# 三、JUC同步锁总结

## 概述

- `ReentrantLock`:可重入锁，利用Condition（每个Condition都有一个阻塞队列）精准唤醒。
- `CountDownLatch`：等待多少线程结束，线程结束后调用await后面的代码。
- `CyclicBarrier`：集齐七颗龙珠召唤神龙
- `Phaser`：未知
- `ReadWriteLock`：读写锁，提高读写效率。
- `Semaphore`：一是用于多个共享资源的互斥使用，二是用于并发线程数的控制（限流）
- `Exchanger`：交换两个线程的数据
- `LockSupport`

## ReentrantLock

**lock与synchronized一一对应的关系**

- `lock.newCondition();`
- `newCondition.await(); 替代wait`
- `newCondition.signal(); 替代notify`

**总而言之，lock替代了synchronized完成加锁解锁的操作**

**lock的`newCondition()`对象替代放弃锁权限，唤醒所有进程的操作**

**JUC实现多生产者，消费者。【生产容量为10】**

```java
/**
 * 最多持有10个资源
 */
public class Resource {
    private int count = 0;
    private Lock lock = new ReentrantLock();
    private Condition condition = lock.newCondition();

    public void increment() {
        lock.lock();
        // 有产品，可以唤醒消费者
        while (count >= 10) {
            try {
                // 放弃得到的锁，并把自身阻塞
                condition.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.printf("当前线程的名字%s，当前count=%d\n", Thread.currentThread().getName(),++count);
        // 唤醒所有进程
        condition.signalAll();
        lock.unlock();
    }

    public void decrement() {
        lock.lock();
        // 没有产品，无法消费了，唤醒生产者
        while (count <= 0) {
            try {
                // 放弃得到的锁，并把自身阻塞
                condition.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.printf("当前线程的名字%s，当前count=%d\n", Thread.currentThread().getName(),--count);
        // 唤醒所有进程
        condition.signalAll();
        lock.unlock();
    }
}


public class MainDemo {
    public static void main(String[] args) {
        Resource resource = new Resource();
        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.increment();
        }, "A").start();

        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.increment();
        }, "AA").start();

        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.decrement();
        }, "B").start();
        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.decrement();
        }, "BB").start();
    }
}
```



----

## CountDownLatch

计数。初始化数值为多少。然后根据条件进行countDown()

调用await方法，只要计数不是0，await这个栓就会锁着。直到计数为0，这个拴才会解锁。

等待多少线程结束，线程结束后做await后面的代码。

> **原理**：目的是为了简化开发！

 CountDownLatch主要有两个方法，当一个或多个线程调用await方法时，这些线程会阻塞。

 其它线程调用countDown方法会将计数器减1(调用countDown方法的线程不会阻塞)，

 当计数器的值变为0时，因await方法阻塞的线程会被唤醒，继续执行。

```java
/**
 * 
 * @Description: *让一些线程阻塞直到另一些线程完成一系列操作后才被唤醒。
 * 
 *               CountDownLatch主要有两个方法，当一个或多个线程调用await方法时，这些线程会阻塞。
 *               其它线程调用countDown方法会将计数器减1(调用countDown方法的线程不会阻塞)，
 *               当计数器的值变为0时，因await方法阻塞的线程会被唤醒，继续执行。
 * 
 *               解释：6个同学陆续离开教室后值班同学才可以关门。
 * 
 *               main主线程必须要等前面6个线程完成全部工作后，自己才能开干
 */
public class CountDownLatchDemo {
    public static void main(String[] args) throws InterruptedException {
        CountDownLatch countDownLatch = new CountDownLatch(6);
        for (int i = 1; i <= 6; i++) {
            new Thread(() -> {
                System.out.println(Thread.currentThread().getName() + "\t 号同学离开教室");
                // 这个方法，当计数为0时，会唤醒被wait阻塞的线程。
                countDownLatch.countDown();
            }, String.valueOf(i)).start();
        }
        //阻塞当前正在运行的线程
        countDownLatch.await();
        System.out.println(Thread.currentThread().getName() + "\t****** 班长关门走人，main线程是班长");
    }
}
```

## CyclicBarrier

回环屏障

满足条件就运行

```java
public class CyclicBarrier {
    public static void main(String[] args) {

        // await了20个线程后，执行run方法，这里用的lambda表达式！
        java.util.concurrent.CyclicBarrier barrier = new java.util.concurrent.CyclicBarrier(20, () -> System.out.println("满人"));

        for (int i = 0; i < 100; i++) {
            new Thread(() -> {
                // 输出20个start后，输出了满人
                System.out.println("start");
                try {
                    barrier.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } catch (BrokenBarrierException e) {
                    e.printStackTrace();
                }
            }).start();
        }
    }
}
```

## Phaser

> 分阶段执行

```java
public class T08_TestPhaser {
    static Random r = new Random();
    static MarriagePhaser phaser = new MarriagePhaser();

    static void milliSleep(int milli) {
        try {
            TimeUnit.MILLISECONDS.sleep(milli);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        phaser.bulkRegister(5);

        for(int i=0; i<5; i++) {
            final int nameIndex = i;
            new Thread(()->{

                Person p = new Person("person " + nameIndex);
                p.arrive();
                phaser.arriveAndAwaitAdvance();

                p.eat();
                phaser.arriveAndAwaitAdvance();

                p.leave();
                phaser.arriveAndAwaitAdvance();
            }).start();
        }

    }

    static class MarriagePhaser extends Phaser {
        @Override
        protected boolean onAdvance(int phase, int registeredParties) {

            switch (phase) {
                case 0:
                    System.out.println("所有人到齐了！");
                    return false;
                case 1:
                    System.out.println("所有人吃完了！");
                    return false;
                case 2:
                    System.out.println("所有人离开了！");
                    System.out.println("婚礼结束！");
                    return true;
                default:
                    return true;
            }
        }
    }

    static class Person {
        String name;

        public Person(String name) {
            this.name = name;
        }

        public void arrive() {
            milliSleep(r.nextInt(1000));
            System.out.printf("%s 到达现场！\n", name);
        }

        public void eat() {
            milliSleep(r.nextInt(1000));
            System.out.printf("%s 吃完!\n", name);
        }

        public void leave() {
            milliSleep(r.nextInt(1000));
            System.out.printf("%s 离开！\n", name);
        }

    }
}
```

## Semaphore

信号量,可用来限流,可以用来多个共享资源的互斥使用。

```java
 Semaphore s = new Semaphore(2, true);
```

AQS  AbstractQueueSynchronizer

在信号量上我们定义两种操作：

 \* acquire（获取） 当一个线程调用acquire操作时，它要么通过成功获取信号量（信号量减1），

 \*    要么一直等下去，直到有线程释放信号量，或超时。

 \* release（释放）实际上会将信号量的值加1，然后唤醒等待的线程。

 \* 信号量主要用于两个目的，一个是用于多个共享资源的互斥使用，另一个用于并发线程数的控制。

```java
package com.atguigu.thread;
import java.util.Random;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
/**
 * 
 * @Description: TODO(这里用一句话描述这个类的作用)  
 * 
 * 在信号量上我们定义两种操作：
 * acquire（获取） 当一个线程调用acquire操作时，它要么通过成功获取信号量（信号量减1），
 *             要么一直等下去，直到有线程释放信号量，或超时。
 * release（释放）实际上会将信号量的值加1，然后唤醒等待的线程。
 * 
 * 信号量主要用于两个目的，一个是用于多个共享资源的互斥使用，另一个用于并发线程数的控制。
 */
public class SemaphoreDemo
{
  public static void main(String[] args)
  {
     Semaphore semaphore = new Semaphore(3);//模拟3个停车位
     for (int i = 1; i 
     {
       new Thread(() -> {
          try 
          {
            semaphore.acquire();
            System.out.println(Thread.currentThread().getName()+"\t 抢到了车位");
            TimeUnit.SECONDS.sleep(new Random().nextInt(5));
            System.out.println(Thread.currentThread().getName()+"\t------- 离开");
          } catch (InterruptedException e) {
            e.printStackTrace();
          }finally {
            semaphore.release();
          }
       }, String.valueOf(i)).start();
     }
  }
}
```

## 读写锁

读：共享锁

写：排他锁

读频繁，写不频繁，读写锁好！

```java
ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
ReadLock readLock = lock.readLock();
WriteLock writeLock = lock.writeLock();
```

`ReentrantReadWriteLock`

>读-读 可以共存！

>读-写 不能共存！

>写-写 不能共存！

```java
package com.bbxx.callable;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * 独占锁（写锁） 一次只能被一个线程占有 
   共享锁（读锁） 多个线程可以同时占有 ReadWriteLock 
     读-读 可以共存！ 
     读-写 不能共存！ 
     写-写 不能共存！
 */
public class ReadWriteLockDemo {
  public static void main(String[] args) {
    MyCache myCache = new MyCache();
    // 写入
    for (int i = 1; i <= 5; i++) {
      final int temp = i;
      new Thread(() -> {
        myCache.put(temp + "", temp + "");
      }, String.valueOf(i)).start();
    }
    // 读取
    for (int i = 1; i <= 5; i++) {
      final int temp = i;
      new Thread(() -> {
        myCache.get(temp + "");
      }, String.valueOf(i)).start();
    }
  }
}

// 加锁的
class MyCacheLock {
  private volatile Map<String, Object> map = new HashMap<>();
  // 读写锁： 更加细粒度的控制
  private ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
  private Lock lock = new ReentrantLock();

  // 存，写入的时候，只希望同时只有一个线程写
  public void put(String key, Object value) {
    readWriteLock.writeLock().lock();
    try {
      System.out.println(Thread.currentThread().getName() + "写入" + key);
      map.put(key, value);
      System.out.println(Thread.currentThread().getName() + "写入OK");
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      readWriteLock.writeLock().unlock();
    }
  }

  // 取，读，所有人都可以读！
  public void get(String key) {
    // readLock是为为了防止 写数据
    readWriteLock.readLock().lock();
    try {
      System.out.println(Thread.currentThread().getName() + "读取" + key);
      Object o = map.get(key);
      System.out.println(Thread.currentThread().getName() + "读取OK");
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      readWriteLock.readLock().unlock();
    }
  }
}

/**
 * 自定义缓存
 */
class MyCache {
  private volatile Map<String, Object> map = new HashMap<>();

  // 存，写
  public void put(String key, Object value) {
    System.out.println(Thread.currentThread().getName() + "写入" + key);
    map.put(key, value);
    System.out.println(Thread.currentThread().getName() + "写入OK");
  }

  // 取，读
  public void get(String key) {
    System.out.println(Thread.currentThread().getName() + "读取" + key);
    Object o = map.get(key);
    System.out.println(Thread.currentThread().getName() + "读取OK");
  }
}
```

## 交换

```java
Exchanger<String> objectExchanger = new Exchanger<>();
不同线程进行交换数据【两个线程之间的数据交换】
    public class Exchange {
        static Exchanger<String> exchanger = new Exchanger<>();

        public static void main(String[] args) {
            new Thread(()->{
                String s = "T1";
                try {
                    s = exchanger.exchange(s);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println(Thread.currentThread().getName() + " " + s);
            }, "t1").start();


            new Thread(()->{
                String s = "T2";
                try {
                    s = exchanger.exchange(s);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println(Thread.currentThread().getName() + " " + s);

            }, "t2").start();
        }
    }

```





----

#  四、不安全集合

## 4.1  `ArrayList`

> **ArrayList的扩容策略是：扩容为原来的1.5倍**

```java
private void grow(int minCapacity) {
    // overflow-conscious code
    int oldCapacity = elementData.length;
    // 扩容为原来的1.5倍
    int newCapacity = oldCapacity + (oldCapacity >> 1);
    if (newCapacity - minCapacity < 0)
        newCapacity = minCapacity;
    if (newCapacity - MAX_ARRAY_SIZE > 0)
        newCapacity = hugeCapacity(minCapacity);
    // minCapacity is usually close to size, so this is a win:
    elementData = Arrays.copyOf(elementData, newCapacity);
}
```

java.util.ConcurrentModificationException

ArrayList在迭代的时候如果同时对其进行修改就会

抛出java.util.ConcurrentModificationException异常

并发修改异常

```java
public class ListTest {
    public static void main(String[] args) {
        // 并发下 ArrayList 不安全的吗，Synchronized；
        /**
         * 解决方案； 1、List<String> list = new Vector<>(); 2、List<String> list =
         * Collections.synchronizedList(new ArrayList<> ()); 3、List<String> list = new
         * CopyOnWriteArrayList<>()；
         */
        // CopyOnWrite 写入时复制 COW 计算机程序设计领域的一种优化策略；
        // 就是我要写入的时候，copy一份副本，在副本中写入，然后把副本设置为本体！读和写的不是同一个容器，写后新容器替换了旧容器！
        // CopyOnWriteArrayList 比 Vector Nb 在哪里？
        List<String> list = new CopyOnWriteArrayList<>();
        for (int i = 1; i <= 10; i++) {
            new Thread(() -> {
                list.add(UUID.randomUUID().toString().substring(0, 5));
                System.out.println(list);
            }, String.valueOf(i)).start();
        }
    }
}

// 看源码可以知道CopyOnWriteArrayList写入 设置新数组时是加了锁的。适合读频繁，写入不频繁的操作。
public boolean add(E e) {
    final ReentrantLock lock = this.lock;
    lock.lock();
    try {
        Object[] elements = getArray();
        int len = elements.length;
        Object[] newElements = Arrays.copyOf(elements, len + 1);
        newElements[len] = e;
        setArray(newElements);
        return true;
    } finally {
        lock.unlock();
    }
}
```

## 4.2 `HashSet`

Set set = new HashSet();//线程不安全

Set set = new CopyOnWriteArraySet();//线程安全

HashSet底层数据结构是什么？

HashMap ?

但HashSet的add是放一个值，而HashMap是放K、V键值对

public HashSet() { map = new HashMap();} HashMap();}

private static final Object *PRESENT* = new Object();

public boolean add(E e) { return map.put(e, *PRESENT*)==null;} ;}

## 4.3 HashMap

```java
Map map = new HashMap();//线程不安全
Map map = new ConcurrentHashMap();//线程安全
```

# 五、线程八锁问题

1 标准访问，先打印短信还是邮件

2 停4秒在短信方法内，先打印短信还是邮件

3 普通的hello方法，是先打短信还是hello

4 现在有两部手机，先打印短信还是邮件

5 两个静态同步方法，1部手机，先打印短信还是邮件

6 两个静态同步方法，2部手机，先打印短信还是邮件

7 1个静态同步方法，1个普通同步方法，1部手机，先打印短信还是邮件

8 1个静态同步方法，1个普通同步方法，2部手机，先打印短信还是邮件

运行答案：

1、短信 

2、短信

3、Hello

4、邮件

5、短信6、短信 5、短信6、短信

7、邮件

8、邮件

----

A 一个对象里面如果有多个synchronized方法，某一个时刻内，只要一个线程去调用其中的一个synchronized方法了，

其它的线程都只能等待，换句话说，某一个时刻内，只能有唯一一个线程去访问这些synchronized方法

锁的是当前对象this，被锁定后，其它的线程都不能进入到当前对象的其它的synchronized方法

加个普通方法后发现和同步锁无关

换成两个对象后，不是同一把锁了，情况立刻变化。

synchronized实现同步的基础：Java中的每一个对象都可以作为锁。

具体表现为以下3种形式。

对于普通同步方法，锁是当前实例对象。

对于静态同步方法，锁是当前类的Class对象。

对于同步方法块，锁是Synchonized括号里配置的对象

当一个线程试图访问同步代码块时，它首先必须得到锁，退出或抛出异常时必须释放锁。

也就是说如果一个实例对象的非静态同步方法获取锁后，该实例对象的其他非静态同步方法必须等待获取锁的方法释放锁后才能获取锁，

可是别的实例对象的非静态同步方法因为跟该实例对象的非静态同步方法用的是不同的锁，

所以毋须等待该实例对象已获取锁的非静态同步方法释放锁就可以获取他们自己的锁。

所有的静态同步方法用的也是同一把锁——类对象本身，

这两把锁是两个不同的对象，所以静态同步方法与非静态同步方法之间是不会有竞态条件的。

但是一旦一个静态同步方法获取锁后，其他的静态同步方法都必须等待该方法释放锁后才能获取锁，

而不管是同一个实例对象的静态同步方法之间，

还是不同的实例对象的静态同步方法之间，只要它们同一个类的实例对象！

```java
package com.atguigu.thread;

import java.util.concurrent.TimeUnit;

class Phone {
    public synchronized void sendSMS() throws Exception {
        System.out.println("------sendSMS");
    }

    public synchronized void sendEmail() throws Exception {
        System.out.println("------sendEmail");
    }

    public void getHello() {
        System.out.println("------getHello");
    }
}

/**
 * 
 * @Description: 8锁
 * 
             1 标准访问，先打印短信还是邮件 
 			2 停4秒在短信方法内，先打印短信还是邮件 
 			3 新增普通的hello方法，是先打短信还是hello 
 			4 现在有两部手机，先打印短信还是邮件 
 			5 两个静态同步方法，1部手机，先打印短信还是邮件 
 			6 两个静态同步方法，2部手机，先打印短信还是邮件 
 			7 1个静态同步方法,1个普通同步方法，1部手机，先打印短信还是邮件 
 			8 1个静态同步方法,1个普通同步方法，2部手机，先打印短信还是邮件
             ---------------------------------
 */
public class Lock_8 {
    public static void main(String[] args) throws Exception {
        Phone phone = new Phone();
        Phone phone2 = new Phone();
        new Thread(() -> {
            try {
                phone.sendSMS();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, "AA").start();
        Thread.sleep(100);
        new Thread(() -> {
            try {
                phone.sendEmail();
                // phone.getHello();
                // phone2.sendEmail();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, "BB").start();
    }
}
```



# 六、强软弱虚

## 强引用

没有引用指向该对象了，就会被垃圾回收了！

重写finalize（）会出现OOM问题。

## 软引用

```java
SoftReference<byte[]> m = new SoftReference<>(new byte[1024 * 1024 * 10]);
m --> Softxxx ~~~> byte
    byte是软引用
    m是强引用
```

软引用主要用于缓存！当内存不够时，会把软引用gc了，腾出空间给其他。如果内存够，则软就存活，用作缓存！

## 弱引用

```java
WeakReference<Object> weakReference = new WeakReference<>(new Object());
new Object()是弱引用
weakReference.get()是get得到的
```

垃圾回收一看到就回收。弱引用是起到一次性的作用哦！ThreadLocal用到了弱引用，确保不要时，把它置为null，gc一定会进行回收。

## 虚引用

get()不到！也随时被回收！有什么用？

管理堆外内存。堆外内存会和jvm内存中的对象关联，当jvm中的对象被回收时，把该对象写入虚引用队列中，虚引用回收时，也回收掉对应的堆外内存



我们现在来看一下自定义了（override）finalize()的对象（或是某个父类override finalize()）是怎样被GC回收的，首先需要注意的是，含有override finalize()的对象A创建要经历以下3个步骤：

- 创建对象A实例
- 创建java.lang.ref.Finalizer对象实例F1，F1指向A和一个reference queue
  (引用关系，F1—>A，F1—>ReferenceQueue，ReferenceQueue的作用先卖个关子)
- 使java.lang.ref.Finalizer的类对象引用F1
  (这样可以保持F1永远不会被回收，除非解除Finalizer的类对象对F1的引用)

经过上述三个步骤，我们建立了这样的一个引用关系：

java.lang.ref.Finalizer–>F1–>A，F1–>ReferenceQueue。GC过程如下所示：

[![JVM finalize实现原理与由此引发的血案](https://simg.open-open.com/show/1e05719ead843db7c45e095d36bcc9b7.png)](https://simg.open-open.com/show/1e05719ead843db7c45e095d36bcc9b7.png)

有override finalize()对象的minor gc

如 上图所示，在发生minor gc时，即便一个对象A不被任何其他对象引用，只要它含有override finalize()，就会最终被java.lang.ref.Finalizer类的一个对象F1引用，等等，如果新生代的对象都含有override finalize()，那岂不是无法GC？没错，这就是finalize()的第一个风险所在，对于刚才说的情况，minor gc会把所有活跃对象以及被java.lang.ref.Finalizer类对象引用的（实际）垃圾对象拷贝到下一个survivor区域，如果拷贝溢 出，就将溢出的数据晋升到老年代，极端情况下，老年代的容量会被迅速填满，于是让人头痛的full gc就离我们不远了。

那 么含有override finalize()的对象什么时候被GC呢？例如对象A，当第一次minor gc中发现一个对象只被java.lang.ref.Finalizer类对象引用时，GC线程会把指向对象A的Finalizer对象F1塞入F1所引 用的ReferenceQueue中，java.lang.ref.Finalizer类对象中包含了一个运行级别很低的deamon线程 finalizer来异步地调用这些对象的finalize()方法，调用完之后，java.lang.ref.Finalizer类对象会清除自己对 F1的引用。这样GC线程就可以在下一次minor gc时将对象A回收掉。

也就是说一次minor gc中实际至少包含两个操作：

- 将活跃对象拷贝到survivor区域中
- 以Finalizer类对象为根，遍历所有Finalizer对象，将只被Finalizer对象引用的对象（对应的Finalizer对象）塞入Finalizer的ReferenceQueue中

可见Finalizer对象的多少也会直接影响minor gc的快慢。

包含有自定义finalizer方法的对象回收过程总结下来，有以下三个风险：

- 如果随便一个finalize()抛出一个异常，finallize线程会终止，很快地会由于f queue的不断增长导致OOM
- finalizer线程运行级别很低，有可能出现finalize速度跟不上对象创建速度，最终可能还是会OOM，实际应用中一般会有富裕的CPU时间，所以这种OOM情况可能不太常出现
- 含有override finalize()的对象至少要经历两次GC才能被回收，严重拖慢GC速度，运气不好的话直接晋升到老年代，可能会造成频繁的full gc，进而影响这个系统的性能和吞吐率。

以上的三点还没有考虑minor gc时为了分辨哪些对象只被java.lang.ref.Finalizer类对象引用的开销，讲完了finalize()原理，我们回头看看最初的那句话：JVM能够保证一个对象在回收以前一定会调用一次它的finalize()方法。

含 有override finalize()的对象在会收前必然会进入F QUEUE，但是JVM本身无法保证一个对象什么时候被回收，因为GC的触发条件是需要GC，所以JVM方法不保证finalize()的调用点，如果对 象一直不被回收，就一直不调用，而调用了finalize()，也不代表对象就被回收了，只有到了下一次GC时该对象才能真正被回收。另外一个关键点是一 次，在调用过一次对象A的finalize()之后，就解除了Finalizer类对象和对象F1之间的引用关系，如果在finalize()中又将对象 本身重新赋给另外一个引用（对象拯救），那这个对象在真正被GC前是不会再次调用finalize()的。

总结一下finalize()的两个个问题：

- 没有析构函数那样明确的语义，调用时间由JVM确定，一个对象的生命周期中只会调用一次
- 拉长了对象生命周期，拖慢GC速度，增加了OOM风险

回 到最初的问题，对于那些需要释放资源的操作，我们应该怎么办？effective java告诉我们，最好的做法是提供close()方法，并且告知上层应用在不需要该对象时一掉要调用这类接口，可以简单的理解这类接口充当了析构函数。 当然，在某些特定场景下，finalize()还是非常有用的，例如实现一个native对象的伙伴对象，这种伙伴对象提供一个类似close()接口可 能不太方便，或者语义上不够友好，可以在finalize()中去做native对象的析构。不过还是那句话，fianlize()永远不是必须的，千万 不要把它当做析构函数，对于一个对性能有相当要求的应用或服务，从一开始就杜绝使用finalize()是最好的选择。

#### 总结

override finalize()的主要风险在于Finalizer的Deamon线程运行的是否够快，它本身是个级别较低的线程，若应用程序中CPU资源吃紧，很可 能出现Finalizer线程速度赶不上新对象产生的速度，如果出现这种情况，那程序很快会朝着“GC搞死你”的方向发展。当然，如果能确保CPU的性能 足够好，以及应用程序的逻辑足够简单，是不用担心这个问题的。例如那个再现问题的小程序，在我自己i7的笔记本上跑，就没有任何GC问题，CPU占用率从 未超过25%（硬件上的东西不太懂，为什么差距会这么多？），出现问题的是在我的办公机上，CPU使用率维持在90%左右。

当 然，互联网应用，谁能保障自己的服务器在高峰期不会资源吃紧？无论如何，我们都需要慎重使用override finalize()。至于JDBC Connector/J中应不应该override finalize()，出于保险考虑，我认为是应该的，但若是公司内部服务，例如网易DDB实现的JDBC DBI（分布式JDBC），Connection完全没必要做这层考虑，如果应用程序忘了调close()，测试环境会很快发现问题，及时更改即可。

---

# 七、生产者消费者

**线程之间通信的典型问题。**

==**解决思路：判断，干活，通知**==

> ==常见面试==：单例模式、排序、生产者和消费者、死锁

## 2.1 sale tickets

为防止虚假唤醒，应该使用while判断是否wait.

if只判断一次流水线状态，while每次都判断？

synchronized锁的对象是方法的调用者。

> **synchronized锁的是当前的对象。一次只需要这个对象的一个synchronized方法进入**，可以把被sy修饰的方法当作原子性的方法。无法其他方法插入执行。

```java
// 资源类
public class SalePerson {
    public int number = 100;

    public synchronized void sale() {
        if (number > 0)
            System.out.println("当前线程为" + Thread.currentThread().getName() + "还剩" + (--number) + "张票");
    }

}

public class MainDemo {
    public static void main(String[] args) {
        SalePerson sp = new SalePerson();
        new Thread( ()->{ while(true) { sp.sale(); } }, "A").start();
        new Thread( ()->{ while(true) { sp.sale(); } }, "AA").start();
        new Thread( ()->{ while(true) { sp.sale(); } }, "AAA").start();
        new Thread( ()->{ while(true) { sp.sale(); } }, "AAAA").start();
        new Thread( ()->{ while(true) { sp.sale(); } }, "AAAAA").start();
        new Thread( ()->{ while(true) { sp.sale(); } }, "AAAAAA").start();
    }
}
```

> **在需要保证原子性操作的代码中加锁 解锁**

```java
// 资源类 JUC写法入门
public class JUCPerson {
    private int number = 100;

    private Lock lock = new ReentrantLock();

    public void sale() {
        try {
            lock.lock();
            if (number > 0)
                number--;
        } finally {
            System.out.printf(Thread.currentThread().getName() + ":还剩%d张票\n", number);
            lock.unlock();
        }
    }
}

public class JUCDemo {
    public static void main(String[] args) {
        JUCPerson jucPerson = new JUCPerson();
        new Thread( ()->{ while(true){ jucPerson.sale(); } } ,"A").start();
        new Thread( ()->{ while(true){ jucPerson.sale(); } } ,"AA").start();
        new Thread( ()->{ while(true){ jucPerson.sale(); } } ,"AAA").start();
        new Thread( ()->{ while(true){ jucPerson.sale(); } } ,"AAAA").start();
    }
}
```



## 2.2 多生产者/消费者

> **wait方法会释放当前持有的锁。**多线程并发条件判断==一定要用while！！！==

```java
/**
 * 只准持有一个资源
 */
public class Resource {
    private int count = 0;

    public synchronized void increment() {
        while (count != 0) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        count++;
        System.out.printf("当前线程为%s；count数值为%d\n", Thread.currentThread().getName(), count);
        this.notifyAll();
    }

    public synchronized void decrement() {
        while (count <= 0) {
            try {
                this.wait();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        count--;
        System.out.printf("当前线程为%s；count数值为%d\n", Thread.currentThread().getName(), count);
        this.notifyAll();
    }
}

public class MainDemo {
    public static void main(String[] args) {
        Resource resource = new Resource();
        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.increment();
        }, "A").start();

        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.increment();
        }, "AA").start();

        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.decrement();
        }, "B").start();
        new Thread(() -> {
            for (int i = 0; i < 66; i++)
                resource.decrement();
        }, "BB").start();
    }
}
```

## 2.3  思考，为什么不用notify?

生产者P1 P2  消费者 C1 C2。缓冲区最大为1.
C1 C2 P1在阻塞队列。 缓冲区为0。
P2拿到锁  生产并唤醒P1。**（此时缓冲区为1 非阻塞对象有P1，P2）**
P1拿了到锁 ，打算生产，发现缓冲区满了，于是阻塞自己，并放弃锁。**（此时缓冲区为1 非阻塞对象有P2）**
此时的就绪队列只有P2，P2拿到锁发现缓冲区为1，也阻塞了自己，造成死锁。**（此时缓冲区为C1 C2 P1 P2全部被阻塞了）**

用notify编写无死锁的程序难度大，推荐用notifyAll。

> **如何达成 C1 C2 P1在阻塞队列且缓冲区为0？**

初始缓冲区为0.
C1 想拿 没有  C1阻塞
C2 想拿 没有  C2阻塞
P1 生产 唤醒 C1  缓冲区为1
P2 拿到锁，生产 发现无法生产  P2阻塞。（此时P1，C1未阻塞，缓冲区为1）
P1 拿到锁，生产 发现无法生产  P1阻塞（C1未阻塞，缓冲区为1）
C1 想拿，有，缓冲区为0，并释放P2，此时只有P2未阻塞，且缓冲区为0。