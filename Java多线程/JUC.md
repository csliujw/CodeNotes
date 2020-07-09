# 一、概述

并发编程的本质：**充分利用CPU的资源**

## 1.1 线程的几个状态

JDK中定义了六个；

新生；运行；阻塞；等待；超时等待；终止==**（WAITING是不见不散，TIMED_WAITING是逾期不候）**==

## 1.2 wait和sleep的区别

**1、来自不同的类**

wait==>Object

sleep==>Thread

**2、关于锁的释放**

wait会释放锁，sleep是不会释放锁。

**3、使用范围不同**

wait必须在同步代码块中

sleep可以在任何地方使用

## 1.3 Lock锁和synchronized

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



## 1.4 一篇优秀的博客

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

值得提的一点是，synchronized是一个非公平的锁，如果竞争激烈的话，可能导致某些线程一直得不到执行。
------------------------------------------------
版权声明：本文为CSDN博主「Crazy丶Mark」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_39907763/article/details/79301813

# 二、生产者消费者

**线程之间通信的典型问题。**

==**解决思路：判断，干活，通知**==

> **==常见面试==：单例模式、排序、生产者和消费者、死锁**

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

# 三、JUC入门之lock

## 3.1 Lock的基本使用

> **lock与synchronized一一对应的关系**

- `lock.newCondition();`
- `newCondition.await(); 替代wait`
- `newCondition.signal(); 替代notify`

**总而言之，lock替代了synchronized完成加锁解锁的操作**

**lock的`newCondition()`对象替代放弃锁权限，唤醒所有进程的操作**

> **JUC实现多生产者，消费者。【生产容量为10】**

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

# 四、不安全集合

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

# 五、`CallableDemo`

## 5.1 获得多线程的几种方式

> 

（1）继承thread类（2）runnable接口

如果只回答这两个你连被问到juc的机会都没有 

正确答案如下：

传统的是继承thread类和实现runnable接口，java5以后又有实现 是继承thread类和实现runnable接口，java5以后又有实现callable接口和java的线程池获得

## 5.2 与Runnable对比

```java
 创建新类MyThread实现runnable接口
class MyThread implements Runnable{

     @Override

     public void run() {

     }

}

新类MyThread2实现callable接口
class MyThread2 implements Callable{

     @Override

     public Integer call() throws Exception {

     	return 200;
     } 

}
```

> **面试题:callable接口与runnable接口的区别？**

 答：（1）是否有返回值

​    （2）是否抛异常

​    （3）落地方法不一样，一个是run，一个是call

## 5.3 Callable使用

new Thread需要传入一个Runnable接口的实现类，但是Callable是一个单独的接口，没有继承接口Runnable。怎么办？找一个类，同时继承了这两个接口即可！

`FutureTask`接口继承了Runnable接口。内部有个构造方法需要传入Callable接口的实现类

```java
public class Demo {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        FutureTask ft = new FutureTask(new MyThread());
        new Thread(ft, "A").start();
        System.out.println(ft.get());

    }
}

class MyThread implements Callable<String> {
    @Override
    public String call() throws Exception {
        System.out.println("I am coming! call!");
        return "11";
    }

}
```

未来的任务，用它就干一件事，异步调用

main方法就像一个冰糖葫芦，一个个方法由main串起来。

但解决不了一个问题：正常调用挂起堵塞问题

在主线程中需要执行比较耗时的操作时，但又不想阻塞主线程时，可以把这些作业交给Future对象在后台完成，

当主线程将来需要时，就可以通过Future对象获得后台作业的计算结果或者执行状态。

一般FutureTask多用于耗时的计算，主线程可以在完成自己的任务后，再去获取结果。

仅在计算完成时才能检索结果；如果计算尚未完成，则阻塞 get 方法。一旦计算完成，

就不能再重新开始或取消计算。get方法而获取结果只有在计算完成时获取，否则会一直阻塞直到任务转入完成状态，

然后会返回结果或者抛出异常。 

只计算一次

get方法放到最后

# 六、JUC强大的辅助类

## 6.1 `CountDownLatch`减少计数

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



## 6.2 `CyclicBarrier`循环栅栏

 \* CyclicBarrier

 \* 的字面意思是可循环（Cyclic）使用的屏障（Barrier）。它要做的事情是，

 \* 让一组线程到达一个屏障（也可以叫同步点）时被阻塞，

 \* 直到最后一个线程到达屏障时，屏障才会开门，所有

 \* 被屏障拦截的线程才会继续干活。

 \* 线程进入屏障通过CyclicBarrier的await()方法。

```java
/**
 * CyclicBarrier
 * 的字面意思是可循环（Cyclic）使用的屏障（Barrier）。它要做的事情是，
 * 让一组线程到达一个屏障（也可以叫同步点）时被阻塞，
 * 直到最后一个线程到达屏障时，屏障才会开门，所有
 * 被屏障拦截的线程才会继续干活。
 * 线程进入屏障通过CyclicBarrier的await()方法。
 * 
 * 集齐7颗龙珠就可以召唤神龙
 */
public class CyclicBarrierDemo{
  private static final int NUMBER = 7;

  public static void main(String[] args) {
    // CyclicBarrier(int parties, Runnable barrierAction)
    CyclicBarrier cyclicBarrier = new CyclicBarrier(NUMBER, () -> System.out.println("召唤神龙"));
    for (int i = 1; i <= 7; i++)
      new Thread(() -> {
        try {
          System.out.println(Thread.currentThread().getName() + "\t 星龙珠被收集 ");
          cyclicBarrier.await();
        } catch (InterruptedException | BrokenBarrierException e) {
          e.printStackTrace();
        }
      }, String.valueOf(i)).start();
  }
}
```



## 6.3 `Semaphore`信号灯

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

# 七、读写锁`ReentrantReadWriteLock`

>**读-读 可以共存！** 

>**读-写 不能共存！** 

>**写-写 不能共存！**

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

# 八、阻塞队列

`BlockingQueue`

