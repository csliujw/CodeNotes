# JVM简介

JVM = <span style="color:green">**Java vritua Machine**</span>

jdk中包含了jvm和“屏蔽操作系统差异的组件”

- jvm各个操作系统之上是一致的
- “屏蔽操作系统差异的组件：在各个PC上各不相同（回忆下载jdk，不同系统需要下载不同版本的jdk；**jdk不同，但是jdk提供的操作方式是一致的,屏蔽了操作系统之间的差异！JVM牛！**）

<img src="./jvm_image/1568186168727.png" style="float:left">

# 零零散散

## 对象内存布局

HotSpot虚拟机中，对象在堆内存中的存储布局可划分为三部分：

- 对象头（Header）
- 实例数据（Instance Data）
- 对齐填充（Padding）

### 对象头

包含两类信息

第一类：存储对象自身的运行时数据

- 如HashCode、GC分代年龄、锁状态标志、线程持有的锁、偏向线程ID、偏向时间戳等。这部分数据在长度为32位和64位（未开启压缩指针）中分别位32个比特和64个比特，官方称之为“Mark Word”
- 在32位的HotSpot虚拟机中，如对象未被同步锁锁定的状态下，mark word的32比特存储空间中的25比特存储HashCode，4bit存储分代年龄（最大15岁），2bit存储锁标志位，1bit固定为0.
- 锁的标志位：
  - 01  未锁定
  - 00  轻量级锁定
  - 10  膨胀（重量级锁定）
  - 11  GC标记
  - 01  可偏向 

## JVM堆内存相关

Java堆的内存可能是规整的，也可能不是规整的。规整的堆，一般用指针碰撞式的分配方式（就是移动指针，分配合适的内存区域），不规整的就是用空闲列表法（熟悉吧！OS！）。

选择何种分配方式由Java堆是否规整决定。而Java堆是否规整又由所采用的垃圾收集器是否带有空间压缩整理（Compact）的能力决定。因此，当使用Serial、ParNew等带压缩整理过程的收集器时，系统采用的分配算法是指针碰撞，既简单又高效；而当使用CMS这种基于清除（Sweep）算法的收集器时，理论上[插图]就只能采用较为复杂的空闲列表来分配内存。

除如何划分可用空间之外，还有另外一个需要考虑的问题：对象创建在虚拟机中是非常频繁的行为，即使仅仅修改一个指针所指向的位置，在并发情况下也并不是线程安全的，可能出现正在给对象A分配内存，指针还没来得及修改，对象B又同时使用了原来的指针来分配内存的情况。解决这个问题有两种可选方案：一种是对分配内存空间的动作进行同步处理——实际上虚拟机是采用CAS配上失败重试的方式保证更新操作的原子性；另外一种是把内存分配的动作按照线程划分在不同的空间之中进行，即每个线程在Java堆中预先分配一小块内存，称为本地线程分配缓冲（Thread Local Allocation Buffer，TLAB），哪个线程要分配内存，就在哪个线程的本地缓冲区中分配，只有本地缓冲区用完了，分配新的缓存区时才需要同步锁定。虚拟机是否使用TLAB，可以通过-XX：+/-UseTLAB参数来设定。

# 垃圾回收算法

## 非JVM垃圾回收算法

引用计数算法。Python用了，COM技术用了。

- 原理：在对象中添加一个引用计数器，每当有一个地方引用，计数器+1，引用失效，计数器-1。计数为0时，对象不可能再被使用，可进行回收~~
- 优点：实现简单 + 高效；但是JVM不是这样做的。
- 缺点：如果存在循环引用，则不是很好判断，但是解决办法是有的。可用图论的算法判断是否有环，然后破环。之前看到过，大概是这个意思，具体咋做的，我忘了~

## JVM用的可达性分析算法

主流的商用程序语言（Java、C#）的内存管理用的都是可达性分析算法判断的对象是否存活。

- 原理：从根结点“GC Roots”开始，根据引用关系向下搜索，搜索走过的路径称为“引用链”。如果从“GC Roots”到某一点不可达，则说明该对象不可能再被使用。

- 没有说优点、缺点、差评！！！

  下图的GC Roots无法到达obj5、6、7 所以5 6 7是可以被回收的~

  <img src="./jvm_image/jvm_gc.jpg" style="float:left">

想深入的话，可能需要去看看专门讲垃圾回收算法的书籍。

## 方法区的回收

方法区的GC性价比低，一般是不能回收到多少东西的。且方法区的回收条件好严格的。

**类的卸载条件相当苛刻：**

- 该类所有的实例都被回收了，即Java堆中不存在该类及其任何派生子类的实例~
- 加载该类的类加载器已经被回收，这个条件除非是经过精心设计的可替换类加载器的场景，如OSGI、JSP的重加载，否则通常是很难达成的！！<span style="color:green">JSP的重加载！！好想知道啊！！！</span>
- 该类对应的java.lang.Class对象没有在任何地方被引用，无法在任何地方通过反射访问该类的方法。

<span style="color:green">**查看类加载卸载的JVM参数**</span>

```shell
-XX:+TraceClass-Loading 
-XX:+TraceClassUnLoading
-XX:+TraceClassLoading 可以在Product版的虚拟机中使用~
```

在大量使用反射、动态代理、CGLib等字节码框架，动态生成JSP以及OSGI这类频繁自定义类加载器的场景中，通常需要Java虚拟机具备类型卸载的能力，保证不会对方法区造成过大的内存压力。常量池从方法区移动到了堆减轻了方法区的压力.

## 正菜：GC算法

推荐阅读 Richard Jones的《垃圾回收算法手册》2~4章的内容

### 分代收集理论！

分代收集理论也有缺陷，最新出现（或实验中）的几款垃圾收集器都展现出了面向全区域收集设计的思想，或可以支持全区域不分代的收集的工作模式~~

#### 分代假说

- 弱分代假说：绝大多数对象都是朝生夕灭~
- 强分代假说：熬过越多次垃圾收集过程的对象越难消亡~
- 跨代引用假说：跨代引用相对于同代引用来说仅占极少数~ 所以如果老生代用了新生代的对象，我们是不考虑的，跨代引用是及少数，不必为了这么一点去浪费时间。
- JVM中的好像就是这样的~

#### 收集的定义

- 新生代收集：Minor GC / Young GC  只是新生代的垃圾收集
- 老年代收集：Major GC / Old GC  只是老年代的垃圾收集
- 混合收集：Mixed GC 收集整个新生代以及部分老年代的垃圾收集
- 整堆收集：Full GC，收集整个Java堆和方法区的垃圾收集

### 算法开始啦

#### 标记-清除算法

算法分为标记和清除两部分。

先标记出要回收的对象，标记完成后统一回收掉被标记的对象。

这里面好多细节没有提到，还是去看下zzm推荐的那本书吧~

- 缺点：
  - 执行效率不稳定
  - 内存空间碎片化（学学OS，内存整理，或者内存分区域，活着的对象移动到一遍，另一边内存直接清零）

#### 标记-复制算法

内存按容量划分成大小相等的两块，每次只用其中一块，要进行垃圾回收时，把活着的对象移动到未使用的那一部分区域，然后把当前区域全部清零。

- 优点：不用考虑内存碎片化的问题，实现简单、高效
- 缺点：浪费空间~（实际上不用对半开，因为多数对象都是会被回收的）

多数商用JVM回收新生代都采用的这种方式。有个量化的诠释：新生代中的对象有98%熬不过第一轮收集。

新生代的划分：

Eden 80%

Survivor from 10%

Survivor to 10%

只有90%的会被使用，剩下10%做担保的~

#### 标记-整理算法

标记复制算法不适用于老年代，因为老年代中的对象，多数都会一直存货下去~

**标记整理算法的原理：**标记过程仍然与“标记-清除”算法一样，但后续步骤不是直接对可回收对象进行清理，而是让所有存活的对象都向内存空间一端移动，然后直接清理掉边界以外的内存。

<span style="color:green">标记-清除算法与标记-整理算法的本质差异在于前者是一种非移动式的回收算法，而后者是移动式的。boxi是否移动回收后的存活对象是一项优缺点并存的风险决策：</span>

- 在老年代 移动存活对象，并更新所有引用这些对象的地方将会是一种极为负重的操作。而这种移动操作必须全程暂停用户应用程序才能进行【新生代不要更新引用吗，真的是！等一下！新生代只有极少数存活下来了！更新的量很小！！！】
- 不移动又容易碎片化
- 折中！先标记清除，碎片化到忍不了了在整理！（<span style="color:green">OS内存管理？？</span>）

#### other

`Parallel Scavenge收集器`：关注吞吐量，基于标记-整理

`CMS收集器`：关注延迟，基于标记清除-算法

### HotSpot算法细节实现

自己看书，好难，我概况不了。

## 经典垃圾收集器

## 低延迟垃圾收集器

## 选择合适的垃圾收集器

# 虚拟机执行子系统

主要记类加载器

## 类的生命周期 虚拟机类加载机制

<span style="color:green">**生命周期： 类的加载->连接->初始化->使用->卸载**</span>

-  **类的加载**

  - 查找并加载类的二进制数据（class文件）

  - 把硬盘上的class文件加载到jvm内存中

- **连接** ：确定类与类之间的关系  ； student.setAddress( address ); 

  - 验证：`*.class` 正确性校验，回忆下类加载器

  - 准备

    static静态变量分配内存，并赋初始化默认值【这是准备阶段做的事】

    static int num =  10 ;  在准备阶段，会把num=0，之后（初始化阶段）再将0修改为10

    在准备阶段，JVM中只有类，没有对象。

    初始化顺序： static ->非static ->构造方法

    ```java
    public class Student{
        
    	static int age; //在准备阶段，将age = 0 ; 在初始化阶段会重新赋值的~~
    	String name;  
    }
    ```
    
  - 解析：把类中符号引用，转为直接引用
  
  前期阶段，还不知道类的具体内存地址，只能使用“com.yanqun.pojo.Student ”来替代Student类，“com.yanqun.pojo.Student ”就称为符号引用；
  
  在解析阶段，JVM就可以将 “com.yanqun.pojo.Student ”映射成实际的内存地址，会后就用 内存地址来代替Student，这种使用 内存地址来使用 类的方法 称为直接引用。
  
 - **初始化**：给static变量 赋予正确的值

    - static int num =  10 ;  在连接的准备阶段，会把num=0，之后（初始化阶段）再将0修改为10

 - **使用：** 对象的初始化、对象的垃圾回收、对象的销毁

 - **卸载：**class的卸载条件非常严格！

<span style="color:green">**JVM结束生命周期的时机：**</span>

- **正常结束**
- **异常结束/错误**  
- **System.exit()**
- **操作系统异常**

## 双亲委派

前置：类的加载

```java
package com.yanqun.parents;
class MyClass{
    
    static int num1 = 100 ;

    static MyClass myClass = new MyClass();
    public MyClass(){
        num1 = 200 ;
        num2 = 200 ;
    }
    static int num2 = 100 ;
    public static MyClass getMyClass(){
        return myClass ;
    }

    @Override
    public String toString() {
        return this.num1 + "\t" + this.num2 ;
    }
}

public class MyClassLoader {
    public static void main(String[] args) {
        MyClass myc =  MyClass.getMyClass() ;
        System.out.println(myc);

    }
}
```

分析

```java
    static int num1 = 100 ;     【 0 】-> 【100】->【200】

    static MyClass myClass = new MyClass();【null】 ->【引用地址0x112231】
    public MyClass(){
        num1 = 200 ;
        num2 = 200 ;
    }
    static int num2 = 100 ;  【0】->【200】->【100】

连接：static静态变量并赋默认值

初始化：给static变量 赋予正确的值
```

总结：在类中 给静态变量的初始化值问题，一定要注意顺序问题（静态变量 和构造方法的顺序问题）

双亲委派： JVM自带的加载器（在JVM的内部所包含，C++）、用户自定义的加载器（独立于JVM之外的加载器,Java）

-  JVM自带的加载器

   - 根加载器,Bootstrap   : 加载 jre\lib\rt.jar （包含了平时编写代码时 大部分jdk api）；指定加载某一个jar（ -Xbootclasspath=a.jar）
   - 扩展类加载器，Extension：C:\Java\jdk1.8.0_101\jre\lib\ext\\\*.jar ;指定加载某一个jar(-Djava.ext.dirs= ....)
   - AppClassLoader/SystemClassLoader，系统加载器（应用加载器）：加载classpath；指定加载（-Djava.class.path= 类/jar）

-  用户自定义的加载器

   - 都是抽象类java.lang.ClassLoader的子类


<img src="./jvm_image/1571541215432.png">

双亲委派：当一个加载器要加载类的时候，自己先不加载，而是逐层向上交由双亲去加载；当双亲中的某一个加载器 加载成功后，再向下返回成功。如果所有的双亲和自己都无法加载，则报异常。

```java
package com.yanqun.parents;
//classpath: .; ..lib，其中“.”代表当前（自己写的类）
class MyClass2{
}

public class TestParentsClassLoader {


    public static void main(String[] args) throws Exception {
       Class myClass1 =  Class.forName("java.lang.Math") ;
        ClassLoader classLoader1 = myClass1.getClassLoader();
        System.out.println(classLoader1);
        /* JDK中的官方说明：
            Some implementations may use null to represent the bootstrap class loader
         */
       Class myClass2 =  Class.forName("com.yanqun.parents.MyClass2") ;
        ClassLoader classLoader2 = myClass2.getClassLoader();
        System.out.println(classLoader2);
    }
}
/**
* null
* sun.misc.Launcher$AppClassLoader@18b4aac2
* 注意：
* /
```

小结：如果类是 rt.jar中的，则该类是被 bootstrap（根加载器）加载；如果是classpath中的类（自己编写的类），则该类是被AppClassLoader加载。

定义类加载：最终实际加载类的 加载器  

初始化类加载类：直接面对加载任务的类

```java
package com.yanqun.parents;

import java.net.URL;
import java.util.Enumeration;

class MyCL{
}
public class JVMParentsCL {
    public static void main(String[] args) throws Exception {
        Class<?> myCL = Class.forName("com.yanqun.parents.MyCL");
        ClassLoader classLoader = myCL.getClassLoader();
        System.out.println(classLoader);
        System.out.println("---");
        ClassLoader systemClassLoader = ClassLoader.getSystemClassLoader();
        System.out.println(systemClassLoader);

        ClassLoader parent1 = systemClassLoader.getParent();
        System.out.println(parent1);
        ClassLoader parent2 = parent1.getParent();
        System.out.println(parent2);

        System.out.println("----");

        ClassLoader appClassLoader = ClassLoader.getSystemClassLoader();
        Enumeration<URL> resources = appClassLoader.getResources("com/yanqun/parents/MyCL.class");// a/b/c.txt
        while(resources.hasMoreElements()){
            URL url = resources.nextElement();
            System.out.println(url);
        }
    }
}
/*
 * sun.misc.Launcher$AppClassLoader@18baac2
 * ---
 * sun.misc.Launcher$AppClassLoader@18baac2
 * sun.misc.Launcher$ExtClassLoader@4554617c
 * null
 * ----
 * file:/D:/github/JavaCore/%xsfsfxasfas/MyJVM/out/production/MyJVM/com/yanqun/parents/MyCL.class
*/
```

值得考究的API：

- `getResource(String name) return URL`
- `getResourceAsStream(String...) return InputStream`

### 自定义类的加载器

二进制名binary names:

```java
"java.lang.String"
"javax.swing.JSpinner$DefaultEditor"
"java.security.KeyStore$Builder$FileBuilder$1"
"java.net.URLClassLoader$3$1"
```

$代表内部类：

$数字：第几个匿名内部类

```
The class loader for an array class, as returned by {@link* Class#getClassLoader()} is the same as the class loader for its element* type; if the element type is a primitive type, then the array class has no* class loader.
```

- 数组的加载器类型  和数组元素的加载器类型 是相同

- 原声类型的数组 是没有类加载器的  

如果加载的结果是null：  可能是此类没有加载器(int[]) ， 也可能是 加载类型是“根加载器”

```
<p> However, some classes may not originate from a file; they may originate* from other sources, such as the network, or they could be constructed by an* application.  The method {@link #defineClass(String, byte[], int, int)* <tt>defineClass</tt>} converts an array of bytes into an instance of class* <tt>Class</tt>. Instances of this newly defined class can be created using* {@link Class#newInstance <tt>Class.newInstance</tt>}.
```

xxx.class文件可能是在本地存在，也可能是来自于网络 或者在运行时动态产生(jsp)

```java
<p> The network class loader subclass must define the methods {@link
 * #findClass <tt>findClass</tt>} and <tt>loadClassData</tt> to load a class
 * from the network.  Once it has downloaded the bytes that make up the class,
 * it should use the method {@link #defineClass <tt>defineClass</tt>} to
 * create a class instance.  A sample implementation is:
 *
 * <blockquote><pre>
 *     class NetworkClassLoader extends ClassLoader {
 *         String host;
 *         int port;
 *
 *         public Class findClass(String name) {
 *             byte[] b = loadClassData(name);
 *             return defineClass(name, b, 0, b.length);
 *         }
 *
 *         private byte[] loadClassData(String name) {
 *             // load the class data from the connection
 *             &nbsp;.&nbsp;.&nbsp;.
 *         }
 *     }
```

如果class文件来自原Network，则加载器中必须重写findClas()和loadClassData().

自定义类加载器的实现

重写findClas()和loadClassData()

```java
package com.yanqun.parents;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

//public class MyException extends Exception{...}
public class MyClassLoaderImpl  extends ClassLoader{
        //优先使用的类加载器是：getSystemClassLoader()
        public MyClassLoaderImpl(){
            // 无参 默认调用的系统类加载器
            super();
        }

        public MyClassLoaderImpl(ClassLoader parent){//扩展类加载器
            super(parent);
        }
        //com.yq.xx.class
        public Class findClass(String name) {
            System.out.println(name);
              byte[] b = loadClassData(name);
              return defineClass(name, b, 0, b.length);
          }

          //“com/yq/xxx.class” ->  byte[]  把字符串变成字节数组就行了
          private byte[] loadClassData(String name)  {

              name =  dotToSplit("out.production.MyJVM."+name)+".class" ;
              byte[] result = null ;
              FileInputStream inputStream = null ;
              ByteArrayOutputStream output = null ;
              try {
                 inputStream = new FileInputStream( new File(  name)  );
                //inputStream -> byte[]
                 output = new ByteArrayOutputStream();

                byte[] buf = new byte[2];
                int len = -1;
                while ((len = inputStream.read(buf)) != -1) {
                    output.write(buf, 0, len);
                }
                result = output.toByteArray();
            }catch (Exception e){
                    e.printStackTrace(); ;
            }finally {
                  try {
                      if(inputStream != null )inputStream.close();
                      if(output != null ) output.close();
                  }catch (Exception e){
                      e.printStackTrace();
                  }
            }
            return result ;
          }

    public static void main(String[] args) throws Exception{
        //自定义加载器的对象
        //默认在双亲委派时，会根据正规流程：系统 —> 扩展 -> 根
        MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl();
		//MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl();
        //要干预的话 直接指定某个 具体的的委派
        Class<?> aClass = null;
        aClass  = myClassLoader.loadClass("com.yanqun.parents.MyDefineCL");
        System.out.println(aClass.getClassLoader());
        MyDefineCL myDefineCL =  (MyDefineCL)(aClass.newInstance() );
        myDefineCL.say();
    }

    public static String dotToSplit(String clssName){
        return clssName.replace(".","/") ;
    }
}

class MyDefineCL{
    public void say(){
        System.out.println("Say...");
    }
}
```

**实现流程：**

- public class MyClassLoaderImpl  extends ClassLoader

- findClass(String name){...} ：直接复制文档中的NetworkClassLoader中的即可

- loadClassData(String name){...} ：name所代表的文件内容->byte[] 

**细节：**

- loadClassData(String name)： 是文件形式的字符串a/b/c.class，并且开头out.production..

- findClass(String name):是全类名的形式  a.b.c.class，并且开头 是： 包名.类名.class

**操作思路：**

要先将 .class文件从classpath中删除，之后才可能用到 自定义类加载器；否在classpath中的.class会被 APPClassLoader加载

```java
package com.yanqun.parents;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

//public class MyException extends Exception{...}
public class MyClassLoaderImpl  extends ClassLoader{
    private String path ; //null
        //优先使用的类加载器是：getSystemClassLoader()
        public MyClassLoaderImpl(){
            super();
        }

        public MyClassLoaderImpl(ClassLoader parent){//扩展类加载器
            super(parent);
        }
        //com.yq.xx.class
        public Class findClass(String name) {
            System.out.println("findClass...");
              byte[] b = loadClassData(name);
              return defineClass(name, b, 0, b.length);
          }

          //“com/yq/xxx.class” ->  byte[]
          private byte[] loadClassData(String name)  {
              System.out.println("加载loadClassData...");
              if(path != null){//name: com.yanqun.parents.MyDefineCL
                  name = path+ name.substring(name.lastIndexOf(".")+1)+".class" ;
              }else{
                  //classpath ->APPClassLoader
                  name =  dotToSplit("out.production.MyJVM."+name)+".class" ;
              }

              byte[] result = null ;
              FileInputStream inputStream = null ;
              ByteArrayOutputStream output = null ;
              try {
                inputStream = new FileInputStream(new File(name));
                //inputStream -> byte[]
                output = new ByteArrayOutputStream();

                byte[] buf = new byte[2];
                int len = -1;
                while ((len = inputStream.read(buf)) != -1) {
                    output.write(buf, 0, len);
                }
                result = output.toByteArray();
            }catch (Exception e){
                    e.printStackTrace(); ;
            }finally {
                  try {
                      if(inputStream != null )inputStream.close();
                      if(output != null ) output.close();
                  }catch (Exception e){
                      e.printStackTrace();
                  }
            }
            return result ;
          }

    public static void main(String[] args) throws Exception {
        System.out.println("main...");
        //自定义加载器的对象
        //默认在双亲委派时，会根据正规流程：系统—>扩展->根
        MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl();
        myClassLoader.path = "d:/" ;

        //MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl();
        //直接指定某个 具体的的委派
        Class<?> aClass = null;
        aClass = myClassLoader.loadClass("com.yanqun.parents.MyDefineCL");
        System.out.println(aClass.getClassLoader());
		// MyDefineCL myDefineCL = (MyDefineCL)(aClass.newInstance()) ;
    }

    public static String dotToSplit(String clssName){  
        return clssName.replace(".","/") ;  
    }
}


class MyDefineCL{
    public void say(){
        System.out.println("Say...");
    }
}
```

代码流程：

```
loadClass() ->findClass()->loadClassData()
```

一般而言，启动类加载loadClass()；

**实现自定义加载器，只需要：**

- **继承ClassLoader**

- **重写的 findClass()**

情况一：用APPClassLoader

classpath中的MyDefineCL.class文件：

1163157884
1163157884

d盘中的MyDefineCL.class文件：

356573597

说明，类加载器 只会把同一个类 加载一次； 同一个class文件  加载后的位置

结论：

自定义加载器 加载.class文件的流程：

先委托APPClassLoader加载，APPClassLoader会在classpath中寻找是否存在，如果存在 则直接加载；如果不存在，才有可能交给 自定义加载器加载。

```java
package com.yanqun.parents;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

//public class MyException extends Exception{...}
public class MyClassLoaderImpl  extends ClassLoader{
    private String path ; //null
        //优先使用的类加载器是：getSystemClassLoader()
        public MyClassLoaderImpl(){
            super();
        }

        public MyClassLoaderImpl(ClassLoader parent){//扩展类加载器
            super(parent);
        }
        //com.yq.xx.class
        public Class findClass(String name) {
//            System.out.println("findClass...");
              byte[] b = loadClassData(name);
              return defineClass(name, b, 0, b.length);
          }

          //“com/yq/xxx.class” ->  byte[]
          private byte[] loadClassData(String name)  {
			// System.out.println("加载loadClassData...");
              if(path != null){//name: com.yanqun.parents.MyDefineCL
				// System.out.println("去D盘加载;;");
                  name = path+ name.substring(name.lastIndexOf(".")+1)+".class";
              }

              byte[] result = null ;
              FileInputStream inputStream = null ;
              ByteArrayOutputStream output = null ;
              try {
                 inputStream = new FileInputStream( new File(  name)  );
                //inputStream -> byte[]
                 output = new ByteArrayOutputStream();

                byte[] buf = new byte[2];
                int len = -1;
                while ((len = inputStream.read(buf)) != -1) {
                    output.write(buf, 0, len);
                }
                result = output.toByteArray();
            }catch (Exception e){
                    e.printStackTrace(); ;
            }finally {
                  try {
                      if(inputStream != null )inputStream.close();
                      if(output != null ) output.close();
                  }catch (Exception e){
                      e.printStackTrace();
                  }
            }
            return result ;
          }

    public static void main(String[] args) throws Exception {
        // System.out.println("main...");
        // 自定义加载器的对象
        // MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl();
        // 默认在双亲委派时，会根据正规流程：系统—>扩展->根
        // myClassLoader.path = "d:/" ;
        // Class<?> aClass = null
        // aClass = myClassLoader.loadClass("com.yanqun.parents.MyDefineCL");
        // System.out.println(aClass.hashCode());
        //默认在双亲委派时，会根据正规流程：系统—>扩展->根
        MyClassLoaderImpl myClassLoader2 = new MyClassLoaderImpl();
        Class<?> aClass2 = myClassLoader2.loadClass("com.yanqun.parents.MyDefineCL");
        System.out.println(aClass2.hashCode());
        // System.out.println(aClass.getClassLoader());
        // MyDefineCL myDefineCL = (MyDefineCL)( aClass.newInstance());
    }

    public static String dotToSplit(String clssName){  
        return clssName.replace(".","/"); 
    }
}

class MyDefineCL{
    public void say(){
        System.out.println("Say...");
    }
}
```

通过以下源码可知，在双亲委派体系中，“下面”的加载器 是通过parent引用 “上面”的加载器。即在双亲委派体系中，各个加载器之间不是继承关系。

```java
public abstract class ClassLoader {

    private static native void registerNatives();
    static {
        registerNatives();
    }

    // The parent class loader for delegation
    // Note: VM hardcoded the offset of this field, thus all new fields
    // must be added *after* it.
    private final ClassLoader parent;
```

ClassLoader源码解读

```java
    protected Class<?> loadClass(String name, boolean resolve)
        throws ClassNotFoundException{
        synchronized (getClassLoadingLock(name)) {
            // First, check if the class has already been loaded
            Class<?> c = findLoadedClass(name);
            if (c == null) {
                long t0 = System.nanoTime();
                try {
                   //如果“父类”不为空，则委托“父类”加载
                    if (parent != null) {
                        c = parent.loadClass(name, false);
                    } else {
                        //如果“父类”为空，说明是双亲委派的顶层了，就调用顶层的加载器（BootstrapClassLoader）
                        c = findBootstrapClassOrNull(name);
                    }
                } catch (ClassNotFoundException e) {
                    // ClassNotFoundException thrown if class not found
                    // from the non-null parent class loader
                }
				//如果“父类”加载失败，则只能自己加载（自定义加载器中的findClass()方法）
                if (c == null) {
                    // If still not found, then invoke findClass in order
                    // to find the class.
                    long t1 = System.nanoTime();
                    c = findClass(name);

                    // this is the defining class loader; record the stats
                    sun.misc.PerfCounter.getParentDelegationTime().addTime(t1 - t0);
                    sun.misc.PerfCounter.getFindClassTime().addElapsedTimeFrom(t1);
                    sun.misc.PerfCounter.getFindClasses().increment();
                }
            }
            if (resolve) {
                resolveClass(c);
            }
            return c;
        }
    }
```

**双亲委派机制优势：** 可以防止用户自定义的类 和 rt.jar中的类重名，而造成的混乱

自定义一个java.lang.Math(和jdk中rt.jar中的类重名)

```java
package java.lang;

public class Math {
    public static void main(String[] args) {
        System.out.println("hello Math...");
    }
}

```

**运行结果：**

<img src="./jvm_image/1571816579201.png">

**原因：**根据双亲委派， 越上层的加载器越优先执行。最顶层的加载器是 根加载器，根加载器就会加载rt.jar中的类。因此rt.jar中的Math会被优先加载。 即程序最终加载的是不是我们自己写的Math，而是jdk/rt.jar中 内置的Math;而内置的Math根本没有提供main()方法，因此报 无法找到main()。

实验：将相关联的类A.class和B.class分别用 不同的类加载器加载

**A和B是继承关系**

```java
public class B{
    public B(){
        System.out.println("B被加载了，加载器是:"+                  this.getClass().getClassLoader());
        //对象使用之前，必然先把此对象对应的类加载
    }
}

public class A extends  B{
    public A(){
        super();
        System.out.println("A被加载了，加载器是："+
                           this.getClass().getClassLoader());
        //对象使用之前，必然先把此对象对应的类加载
    }
}

//AppClassLoader.class : TestMyClassLoader2
//自定义加载器: A.class/B.class
public class TestMyClassLoader2 {
    public static void main(String[] args) throws Exception{
        MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl() ;
        //自定义加载路径
        myClassLoader.path = "d:/";
        Class<?> aClass = myClassLoader.loadClass("com.yanqun.parents.A");
        //newInstance()会调用 该类的构造方法(new 构造方法())
        Object aObject = aClass.newInstance();
        System.out.println(aObject);
    }
}
```

**A和B不是继承关系**

```java
public class Y {
    public Y(){
        System.out.println("Y被加载了，加载器是： "+ this.getClass().getClassLoader());//对象使用之前，必然先把此对象对应的类加载
    }
}
public class X {
    public X(){
        new Y() ;//加载Y（系统加载器）
        System.out.println("X被加载了，加载器是： "+ this.getClass().getClassLoader());//对象使用之前，必然先把此对象对应的类加载
    }
}

//AppClassLoader.class : TestMyClassLoader2
//自定义加载器: A.class/B.class
public class TestMyClassLoader3 {
    public static void main(String[] args) throws Exception{
        MyClassLoaderImpl myClassLoader = new MyClassLoaderImpl() ;
        //自定义加载路径
        myClassLoader.path = "d:/" ;
        //程序第一次加载时（X），使用的是  自定义加载器
        Class<?> aClass = null;
        aClass = myClassLoader.loadClass("com.yanqun.parents.X");

		//newInstance()会调用 该类的构造方法(new 构造方法())
        Object aObject = aClass.newInstance();
        System.out.println(aObject);
    }
}
```



```markdown
存在继承关系

A.class:  classpath
B.class:   classpath
原因
同一个AppClassLoader 会同时加载A.class和B.class

-------

A.class:   d:\
B.class:   classpath
抛出异常 IllegalAccess
原因：
A.class：自定义加载器加载
B.class：被AppClassLoader加载
因此，加载A.class和B.class的不是同一个加载器

---

A.class:    classpath
B.class:    d:\	
NoClassDefFoundError
原因:
A.class: 被AppClassLoader加载  
B.class: 自定义加载器加载
因此，加载A.class和B.class的不是同一个加载器

--

A.class	d:\
B.class d:\
TestMyClassLoader2 can not access a member of class com.yanqun.parents.A with modifiers "public"
A.class/B.class: 自定义加载器加载
原因是 main()方法所在类在 工程中（APPClassLoader），而A和B不在工程中（自定义加载器）。
造成这些异常的核心原因： 命名空间（不是由同一个类加载器所加载）

----

没有继承关系
X.class:  D:		自定义加载器
Y.class:  classpath	系统加载器

Y被加载了，加载器是： sun.misc.Launcher$AppClassLoader@18b4aac2
X被加载了，加载器是： com.yanqun.parents.MyClassLoaderImpl@74a14482

---

X.class:  classpath  系统加载器
Y.class:  D:	    自定义加载器
java.lang.NoClassDefFoundError: com/yanqun/parents/Y

---

```

<img src="./jvm_image/1571824096839.png">

如果存在继承关系： 继承的双方（父类、子类）都必须是同一个加载器，否则出错；

如果不存在继承关系： 子类加载器可以访问父类加载器加载的类（自定义加载器，可以访问到 系统加载器加载的Y类）；反之不行（父类加载器 不能访问子类加载器）

核心： 双亲委派

如果都在同一个加载器 ，则不存在加载问题； 如果不是同一个，就需要双亲委派。

如果想实现各个加载器之间的自定义依赖，可以使用ogsi规范

<img src="./jvm_image/1572660850713.png">

OSGi：

- 网状结构的加载结构

- 屏蔽掉硬件的异构性。例如，可以将项目部署在网络上，可以在A节点上 远程操作B节点。在操作上，可以对硬件无感。也可以在A节点上 对B节点上的项目进行运维、部署，并且立项情况下  在维护的期间，不需要暂时、重启。【热部署？】

Java可以实现热部署，单不是很方便，（不是Java擅长的。）但是用OSGI模型的话，它天然支持。

> **RPC远程调用**

### 类的卸载

- 系统自带（系统加载器、扩展加载器、根加载器）：这些加载器加载的类  是不会被卸载。

- 用户自定义的加载器，会被GC卸载GC

# 高效并发

## JMM

### JMM划分

**JVM内存模型（Java Memoery Model，简称JMM），用于定义（所有线程的共享变量， 不能是局部变量）变量的访问规则，屏蔽各种硬件和OS的内存访问差异，以实现让Java程序在各种平台下都达到一致的内存访问效果**

**PS：**主流程序语言（C / CPP等）直接使用物理硬件和OS的内存模型，由于不同平台上内存模型的差异，可能会导致程序在一套平台上并发完全正常，在另一套平台上并发访问经常出错。为了获取更好的执行效能，Java内存模型并没有限制执行引擎使用处理器的特定寄存器或缓存来和主内存进行交互，也没有限制即时编译器是否要进行调整代码执行顺序这类优化措施。

<span style="color:green">**JMM内存划分：主内存区、工作内存区**</span>

**PS：**Java内存区域和JMM不是同一层次的内存划分，两者基本上没有任何关系。如果非要对应起来，那么对应【】中的内容。

- 主内存区 ：真实存放变量【主内存主要对应Java堆中的对象实例数据部分】
- 工作内存区：主内存中变量的副本，供各个线程所使用【工作内存对应于虚拟机栈中的部分区域】
- <u>*再基础一些，主内存直接对应物理硬件内存，为了更好的效率，可能会让工作内存优先存储于寄存器和高速缓存中*。</u>

**注意：**

- 各个线程只能访问自己私有的工作内存（不能访问其他线程的工作内存，也不能访问主内存）
- 不同线程之间，可以通过主内存 简介的访问其他线程的工作内存

<span style="color:red">**都是双向箭头**</span>

```mermaid
graph LR
Java线程1==>工作内存1==>Save和Load操作==>主内存
Java线程2==>工作内存2==>Save和Load操作
Java线程3==>工作内存3==>Save和Load操作
```

### 内存间交互操作

主内存与工作内存之间具体的交互协议。变量如何再工作内存和主内存之间相互交互。

<span style="color:green">**记忆型内容，无法测试**</span>

完整的研究：不同线程之间交互数据时 经历的步骤：

- **Lock：**将主内存中的变量，表示为一条线程的独占状态
- **Unlock：**解决线程的独占状态，把处于锁定状态的变量释放出来。
- **Read**：将主内存中的变量，读取到工作内存中【主内存---->工作内存】
- **Load：**将read操作从主内存中得到的变量存放到工作内存的变量副本中。【**把主内存读到的数据，存入工作内存，read读数据，load存数据**】
- **Use：**把工作内存中的变量副本，传递给线程去使用【得到数据是为了use】
- **Assign：**把线程正在使用的变量，传递给工作内存中的变量副本中
- **Store：**将工作内存中变量副本的值，传递到主内存中【工作内存----->主内存】
- **Write：**将变量副本作为一个主内存中的变量进行存储【**把工作内存读到的数据，存入主内存，Store拿到要存的数据，Write存数据**】

<img src="./jvm_image/1568776570724.png" style="float:left">

### 针对long和double型变量的特殊规则

JVM要求以上的8个动作必须是原子性的；JVM但是对于64位的数据类型（long double）有些非原子性协议。说明什么问题：在执行以上8个操作时，可能会出现 只读取（写入等）了半个long/double数据，因此出现错误。如何避免？ 

- 商用JVM已经充分考虑了此问题，无需我们操作  
- 可以通过**volatile**避免此类问题（读取半个数据的问题）   volatile double num ;

我们只需要关注Java内存模型的定义即可，不用考虑上述8个动作的并发问题，Java将上述的操作简化成了read、write、lock、unlock四种，与上述的8种操作是等价的~~

<span style="color:green">**等效判断原则-----先行发生原则，用来确定一个操作在并发环境下是否安全**</span>

### volatile 

#### 概念

JVM提供的一个轻量级的同步机制（可以说是最轻量级的同步机制）

当一个变量被声明为volatile时，线程在写入变量时不会包值缓存在寄存器或者其他地方，而是会把值刷新会主存。

volatile可以防止指令重排序

<span style="color:green">**请详细阅读汇编代码和对应的解释！**</span>

-----

volatile只是保证你读取的时候一定是从主内存中拿数据.

但是在操作的时候,不保证这个值是与主内存中同步更新的.

且Java的一些操纵如:+ - * /不是原子性的,所以可能会出现并发问题.

请看下方的部分汇编代码

**附上汇编代码**

```java
javap -c 字节码指令

public static void increase():
	Code:
		Stack=2, Locals=0, Args_size=0
        0:	getstatic  #13;  //Field race:I
		3:  iconst_1
		4:  iadd
		5:  putstatic #13;  //Field race:I
		8:  return
    LineNumberTable:
		line 14: 0
        line 15: 8
```

当getstatic指令把race的值渠道操作栈顶时，volatile关键字保证了race的值是正确的，但是在执行iconst_1、iadd这些指令的时候，其他线程可能已经把race的值改变了，而操作栈顶的值就变成了过期的数据，所以putstatic指令执行后就可能把较小的race值同步回主内存中。

使用字节码分析并发问题并不严谨,因为字节码指令也不一定是原子性的.但是这里用字节码足以说明问题了!

#### 特性

- 此变量对所有线程的可见性。一条线程修改了这个变量的值，新值对于其他线程来说是可以立即得知的。【它给的原因是Java里面的运算符操作并非原子性的，而volatile不能保证操作的原子性】
- 可防止指令重排序

```java
/**
 * volatile非原子性的测试
 */
public class VolatileTest {
    public static volatile int race = 0;
    private static final int THREAD_COUNT = 20;

    public static void increase() {
        race++;
    }

    public static void main(String[] args) {
        Thread[] threads = new Thread[THREAD_COUNT];
        for (int i = 0; i < THREAD_COUNT; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 100000; j++) {
                    increase();
                    System.out.println(race);
                }
            });
            threads[i].start();
        }
       //current thread will to yield当前线程让步，当前线程为main线程
        while (Thread.activeCount() > 1)
            Thread.yield();
    }
}
```

#### 适用场景

- 防止JVM对long/double等64位的非原子性协议进行的误操作（读取半个数据）

- 可以使变量对所有的线程立即可见（某一个线程如果修改了 工作内存中的变量副本，那么加上volatile 之后，该变量就会立刻同步到其他线程的工作内存中）【立即可见还是不是很清楚。大概就是，工作内存修改完变量后会**立刻**同步到主内存中。】

- 禁止指令的“重排序”优化

#### 经典代码

> <span style="color:green">**单例模式DCL**</span>

重排序：比如`Object obj = new Object()`假设这个操作分为三个原子性操作，则排序的对象就是这三个原子性操作。重排序的目的是为了提高执行效率，优化.【CPU的流水线操作？】

```java
int a = 10 ; //1    int a ; a = 10 ; 先把操作拆成原子性的，再对原子性操作进行排序
int b ;//2
b = 20 ;//3
int c = a * b ;//4
```

重排序“不会影响**单线程**的执行结果”，因此以上程序在经过重排序后，可能的执行结果：1,2,3,4 ；2,3,1,4

```java
//2 3 1 4
int b ;
b = 20 ;
int a  =10 ;
int c = a * b ;
```

但是以下代码可能会有问题【】

```java
/**
 * 经典DCL 为什么要volatile？
 * 防止指令重排序
 * 可以看下 即时编译的汇编代码，如何获得，我不知道
 */
public class DCL {
    // 没有用volatile修饰
    private static DCL instance;

    public static DCL getInstance() {
        // 这个null是为了提高效率，如果实例化了，就不走锁这步了。
        if (instance == null) { 
            // class字节码锁，字节码对象一般唯一
            synchronized (DCL.class) {
                if (instance == null) {
                    instance = new DCL(); // 非原子性操作！
                }
            }
        }
        return instance;
    }
}
```

以上代码可能会出现问题，原因 instance = new Singleton() 不是一个原子性操作，会在执行时**拆分成以下动作**：

- JVM会分配内存地址、内存空间
- 使用构造方法实例化对象
- instance = 第1步分配好的内存地址

根据重排序的知识，可知，以上3个动作在真正执行时可能1、2、3，也可能是1、3、2

**如果在多线程环境下，使用1、3、2可能出现问题**：

假设线程A刚刚执行完以下步骤（即刚执行 1、3，但还没有执行2，先分配内存空间，它就不是null了，然后再进行实例化。）

1正常0x123 ,  ...

3 instance=0x123

此时，线程B进入单例程序的if，直接会得到Instance对象（注意，此instance是刚才线程A并没有new的对象）,就去使用该对象，例如instance.xxx() 则必然报错。解决方案，就是 禁止此程序使用1 3 2 的重排序顺序。解决办法：

```java
// 加volatile 禁止对象初始化的时候 进行重排序。
// 那么问题来了，volatile是对Java的一步操作 禁止重排序吗？
private volatile static Singleton instance = null ;
```

**volatile是通过“内存屏障”防止重排序问题：**

- 在volatile写操作前，插入StoreStore屏障
- 在volatile写操作后，插入StoreLoad屏障
- 在volatile读操作后，插入LoadLoad屏障
- 在volatile读操作后，插入LoadStore屏障

***volatile***是否能保证原子性、保证线程安全？不能！

要想保证原子性/线程安全，可以使用原子包java.util.cocurrent.aotmic中的类，该类能够保证原子性的核心，是因为提供了compareAndSet()方法，该方法提供了 cas算法（无锁算法）。

### 原子性、可见性与有序性

#### 原子性（Atomicity）

由Java内存模型来直接保证的原子性变量操作包括read、load、assign、use、store和write这六个。大致可以认为基本数据类型的访问、读、写都是具备原子性的（例外就是long和double的非原子性协定）。

Java提供了更高层的字节码指令monitorenter和monitorexit保证更大范围的原子性操作。（synchronized的底层就是monitorxxx）

#### 可见性（Visibility）

- **volatile**
- **synchronized**
- **final**

都可保证可见性

一个线程修改了共享变量的值是，其他线程能够立即得知这个修改。JMM是通过在变量修改后将值同步回主内存。volatile保证了新值能立即同步过去，而普通变量无法保证。（是因为volatile禁止了重排序吗？）

#### 有序性（Ordering）

本线程内观察，所有操作都是有序的；如果在一个线程中观察另一个线程，所有操作都是无序的。

### 先行发生原则

不知道写些什么。

## JVM运行时的内存区域

将JVM在运行时的内存，划分为了5个部分，如图所示。

<img src="./jvm_image/1568883650128.png" style="float:left">

### 程序计数器

程序计数器：行号指示器，指向当前线程所执行的字节码指令的地址。像不像编译原理中的一个东西。

Test.java -> Test.class

```java
int num = 1;   //1
int num2 = 2 ; //2
if(num1>num2){//3
...//4-10
}else //11
{
    ...
}
while(...)
{
    
}
```

简单的可以理解为：class文件中的行号

> **注意：**

- 一般情况下，程序计数器 是行号；但如果正在执行的方法是native方法，则程序计数器的值 undefined。
- 程序计数器 **是唯一一个 不会 产生 “内存溢出”的区域。**

goto的本质就是改变的 程序计数器的值（java中没有goto，goto在java中的保留字）

### 虚拟机栈

定义：描述 方法执行的内存模型

- 方法在执行的同时，会在虚拟机栈中创建一个栈帧
- 栈帧中包含：方法的局部变量表，操作数据栈、动态链接、方法出口信息等



当方法太多时，就可能发生 栈溢出异常StackOverflowError，或者内存溢出异常OutOfMemoryError

```java
public static void main(String[] args) {
    main(new String[]{"abc","abc"});
}
```

### 本地方法栈

原理和结构与虚拟机栈一致，不同点： 虚拟机栈中存放的 jdk或我们自己编写的方法，而本地方法栈调用的 操作系统底层的方法。【感觉在看编译原理】

先在操作数栈中放10，在把10传递到局部变量表中。

<img src="./jvm_image/1568887498037.png" style="float:left">

### 堆

从回收内存的角度看，由于现代垃圾收集器大部分都是基于分代收集理论设计的，所以Java堆中经常会出现“新生代”“老年代”“永久代”“Eden空间”“From Survivor空间”“To Survivor空间”等名词.

<img src="./jvm_image/1568940940376.png" style="float:left">

- 存放对象实例（数组、对象）

- 堆是jvm区域中最大的一块，在jvm启动时就已经创建完毕

- GC主要管理的区域

- 堆本身是线程共享，**但在堆内部可以划分出多个线程私有的缓冲区**【线程缓冲区】

- 堆允许物理空间不连续，只要逻辑连续即可

- 堆可以分 新生代、老生代 。大小比例，新生代：老生代= 1:2  

- 新生代中 包含eden、s0、s1 = 8:1:1  

- 新生代的使用率一般在90%。 在使用时，只能使用 一个eden和一块s区间(s0或s1)

- 新生代：存放 1.生命周期比较短的对象  2.小的对象；反之，存放在老生代中。对象的大小，可以通过参数设置 -XX：PretenureSizeThredshold 。一般而言，大对象一般是 集合、数组、字符串。生命周期： -XX:MaxTenuringThredshold

  新生代、老生代中年龄：MinorGC回收新生代中的对象。**如果Eden区中的对象在一次回收后仍然存活，就会被转移到 s区中；之后，如果MinorGC再次回收，已经在s区中的对象仍然存活，则年龄+1。如果年龄增长一定的数字，则对象会被转移到 老生代中。**简言之：在新生代中的对象，每经过一次MinorGC，有三种可能：1从eden --> s区   2.（已经在s区中）年龄+1  3.转移到老生代中

  <img src="./jvm_image/1568946567405.png">

  

  新生代在使用时，只能同时使用一个s区：**底层采用的是复制算法，为了避免碎片产生**。只同时使用一个s区。如s0在用着，回收后s0的对象不是连续存放了，于是把存活的复制到s1中，让他们重新连续起来。空间连续的话，寻址方便。

  老生代： 1.生命周期比较长的对象  2.大的对象； 使用的回收器 MajorGC\FullGC

  新生代特点： 

  - 大部分对象都存在于新生代
- 新生代的回收频率高、效率高
  
  老生代特点：

  - 空间大、
- 增长速度慢
  - 频率低
  
  意义：可以根据项目中 对象大小的数量，设置新生代或老生代的空间容量，从提高GC的性能。


如果对象太多，也可能导致内存异常。

虚拟机参数：JDK版本不同，虚拟参数也有所出入。具体情况看Oracle官方文档。

-Xms128m ：JVM启动时的大小

 -Xmn32m：新生代大小

 -Xmx128：总大小

jvm总大小= 新生代 + 老生代  JDK1.8一起有个叫永久代【存放元数据，比如说类的一些加载信息】的，现在已经不存在了。

堆内存溢出的示例：java.lang.OutOfMemoryError: Java heap space

```java
package com.yanqun;

import java.util.ArrayList;
import java.util.List;

public class TestHeap {
    public static void main(String[] args) {

        List list = new ArrayList() ;
        while(true){
            list.add(  new int[1024*1024]) ;
        }

    }
}

```

### 方法区

永久代从堆转到了方法区

存放：类的元数据（描述类的信息）、常量池、方法信息（方法数据、方法代码）

gc：类的元数据（描述类的信息）、常量池

方法区中数据如果太多，也会抛异常OutOfMemory异常



图：方法区与其他区域的调用关系【<不是很明白】

<img src="./jvm_image/1568949355998.png">

常量池：存放编译期间产生的 字面量("abc")、**符号引用**。【字符串常量池只是其中的一部分】

<span style="color:green">**注意： 导致内存溢出的异常OutOfMemoryError，除了虚拟机中的4个区域以外，还可能是直接内存。在NIO技术中会使用到直接内存。直接内存，直接操作OS的内存**</span>

那么直接内存如何回收？

## 类的使用方式

类的初始化：JVM只会在**“首次主动使用”**一个类/接口时，才会初始化它们，初始化类哦！不是初始化对象！这里说的好像是类加载器中的那个类~~【查查资料！】 。

### 主动使用

- new 构造类的使用  这个例子不恰当啊！

```java
package init;

public class Test1 {

    static{
        System.out.println("Test1...");
    }

    public static void main(String[] args) {
        new Test1();//首次主动使用
        new Test1();
    }
}
```

结果：Test1...

- 访问类/接口的 静态成员（属性、方法）

```java
package init;
class A{
    static int i  = 10;

    static{
        System.out.println("A...");
    }

     static void method(){
        System.out.println("A method...");
    }
}

public class Test2 {
    public static void main(String[] args) {

//        A.i = 1 ;
//        A.i = 1 ;
//        System.out.println(A.i);
        A.method();
    }
}
```

注：main()本身也是一个静态方法，也此main()的所在类 也会在执行被初始化

<span style="color:green">**特殊情况：**</span>

- 如果成员变量既是static，又是final ，即常量，则不会被初始化
- 上一种情况中，如果常量的值 是一个随机值，则会被初始化 (为了安全)

- 使用Class.forName("init.B")执行反射时使用的类（B类）

- 初始化一个子类时，该子类的父类也会被初始化

```java
public class Son extends Father {
    public static void main(String[] args) {
        new Son();
    }
}
```

- 动态语言在执行所涉及的类 也会被初始化（动态代理）

### 被动使用

除了主动以外，其他都是被动使用。

```java
package init;
class BD{
    
    static {
        System.out.println("BD...");
    }
}

public class BeiDong {
    
    public static void main(String[] args) {
        BD[] bds = new BD[3];
    }
}

```

以上代码，不属于主动使用类，因此不会被初始化。

### 主动使用中的静态成员问题

```java
class B{
    /**
     final static 称为常量
     常量产生的时机：
     	时间：编译期间
     	地点：（调用这个常量的方法 所在类【Test2】）常量池
     **/
    final static int i = 10;
    
    static{
        System.out.println("A....")
    }
}
public class Test2{
    // Test2中调用的A的常量，所以A中产生的常量保存在Test2的常量池中
    public static void main(String[] args){
        System.out.println(A.i);
    }
}
```

```java
class Father{
    public static int i = 10;
    static{
        system.out.println("father");
    }
}
class Son extends Father{
    static{
        System.out.println("son...");
    }
}

public class Test{
    public static void main(String[] args){
        System.out.println(Son.i)
    	// 将会输出 father...
        // 10
        // 因为 i是Father里的
    }
}
```

# 字节码知识

## 助记符

<span style="color:green">**Java的bin目录下常用的命令**</span>

- java 运行字节码
- javac 把` *.java`编译成` *.class`
- javap 把字节码文件反编译

反编译： cd到class目录中， javap -c class文件名



Test.java  ->                    javap -c Test.java

javap反编译的是class文件

应该：xx.java -> xx.class ->javap



aload_0: 装载了一个引用类型

 Invokespecial:  init,  private  , super.method() :  \<init\>存放的是初始化代码的位置

getstatic ：获取静态成员

bipush ： 整数范围 -128  -- 127之内  (8位带符号的整数),放到栈顶

sipush:    >127   (16个带符号的整数),放到栈顶

注意：无论是定义int或short 等，只要在 -128 --127以内 都是bipush，否则是sipush.

注意：特殊：-1  -- 5不是bipush

​		 iconst_m1（-1）  iconst_0   iconst_1 ....  iconst_5



ldc  : int  float String 常量 ,放到栈顶

ldc2_w :long  double常量,放到栈顶

# JVM四大引用

## JVM四种引用级别

- **Reference：**
  - 直接new的 强引用
  - SoftReference 软引用  内存不足时，回收软引用
  - WeakReference 弱引用  只要发生了GC就会回收
  - PhantomReference 虚引用
- **虚拟机内存大小设置：**
  - -Xmx128m 最大128m
  - -Xms64m  启动时64m
  - 示例：`-Xmx128m  -Xms64m`

如果一个对象存在着指向它的引用，那么这个对象就不会被GC回收？  -- 局限性 【循环引用！JVM用的可达性分析！】

Object obj = new Object() ; --强引用

<span style="color:green">**根据引用的强弱关系： 强引用>软引用>弱引用>虚引用；强软弱虚 四大引用**</span>

### 强引用

Object obj = new Object() ;

**约定：**引用 obj，引用对象new Object()

强引用对象什么失效？

- 生命周期结束（作用域失效）

```java
public void method(){
	Object obj = new Object() ;
}
//当方法执行完毕后，强引用指向的 引用而对象new Object()就会等待被GC回收
```

- 引用被置为null，引用对象可以被GC回收

```java
obj = null ;
//此时，没有任何引用指向new Object() 因此，new Object() 就会等待被GC回收
```

- 除了以上两个情况以外，其他任何时候GC都不会回收强引用对象。宁愿内存溢出，也不会随便回收强引用对象！

### 软引用

<a href="bilibili.com/video/BV1d54y1z792?p=13">p13视频补充（DT课堂）</a>

**可以做缓存**

根据JVM内存情况： 如果内存充足，GC不会随便的回收软引用对象；如果JVM内存不足，则GC就会主动的回收软引用对象。

各种引用的出处：

强引用:new

软引用 弱引用 虚引用 （最终引用）：都是Reference的子类

软引用：java.lang.ref.SoftReference

Reference中有一个get()方法，用于返回 所引用的对象【**<span style="color:green">软引用包裹一下对象。</span>**】

```java
SoftReference<SoftObject> softRef = new SoftReference<>(new SoftObject() );
```

softRef.get()  -->返回引用所指向的SoftObject对象本身

```java
package ref;

import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.List;

//软引用对象
class SoftObject{}
public class SoftReferenceDemo {
    public static void main(String[] args) throws Exception {
        //softRef  -->SoftObject  设计模式中的：装饰模式
        SoftReference<SoftObject> softRef = new SoftReference<>(new SoftObject() );
        List<byte[]> list = new ArrayList<>();

        //开启一个线程，监听 是否有软引用已经被回收
        new Thread(  ()->{
        while(true) {
            //软引用对象
            if (softRef.get() == null){
                System.out.println("软引用对象已被回收..");
                System.exit(0);
            }
        }
       }  ,"线程A" ) .start();     //lambda

        //不断的往集合中 存放数据，模拟内存不足
        while(true){
//          Thread.sleep(10);
            if(softRef.get() != null)
                list.add(new byte[1024*1024]) ;//每次向list中增加1m内容
        }
    }
}
```

### 弱引用

回收的时机：只要GC执行，就会将弱引用对象进行回收。

`java.lang.ref.WeakReference<T>`

```java
package ref;

import java.lang.ref.WeakReference;

public class WeakReferenceDemo {
    public static void main(String[] args) throws Exception {

        WeakReference<Object> weakRef = new WeakReference<>(new Object());
        //weakRef->Object
        System.out.println( weakRef.get()==null?"已被回收":"没被回收");
        System.gc();//建议GC执行一次回收（存在概率）
        Thread.sleep(100);
        System.out.println( weakRef.get()==null?"已被回收":"没被回收");
    }
}
```

### 虚引用（幻影引用或者幽灵引用）

java.lang.ref.PhantomReference\<T\>

是否使用虚引用，和引用对象本身 没有任何关系； 无法通过虚引用来获取对象本身.

引用get() -> 引用对象

虚引用get() -> null

虚引用不会单独使用，一般会和 引用队列（java.lang.ref.ReferenceQueue）一起使用。

价值： 当gc回收一个对象，如果gc发现 此对象还有一个虚引用，就会将虚引用放入到 引用队列中，之后（当虚引用出队之后）再去回收该对象。因此，我们可以使用 虚引用+引用对象 实现：在对象被gc之前，进行一些额外的其他操作。

 GC ->如果有虚引用->虚引用入队->虚引用出队-> 回收对象 【<span style="color:green">**在被回收前，可以进行一些操作**</span>】

```java
package ref;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

class MyObject {
}

public class PhantomReferenceDemo {

    public static void main(String[] args) throws Exception {
        MyObject obj = new MyObject();
        //引用队列
        ReferenceQueue queue = new ReferenceQueue();

        //虚引用+引用队列
        PhantomReference<MyObject> phantomRef = new PhantomReference<>(obj, queue);

        //让gc执行一次回收操作
        obj = null;
        System.gc();
        Thread.sleep(30);
        System.out.println("GC执行...");

        //GC-> 虚引用->入队->出队->     obj
        System.out.println(queue.poll());
    }
}
```

<span style="color:green">**特殊情况：如果虚引用对象重写了finalize()，那么JVM会延迟 虚引用的入队时间。**</span>

```java
package ref;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

class MyObject3 {
    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        System.out.println("即将被回收之前...");
    }
}

public class PhantomReferenceDemo2 {

    public static void main(String[] args) throws Exception {
        MyObject3 obj = new MyObject3();
        // 引用队列
        ReferenceQueue queue = new ReferenceQueue();

        // 虚引用+引用队列
        PhantomReference<MyObject3> phantomRef = new PhantomReference<>(obj, queue);

        // 让gc执行一次回收操作
        obj = null;

        System.gc();
		// Thread.sleep(30);
        System.out.println("GC执行...");

        // GC-> 虚引用->入队->出队->     obj
        System.out.println(queue.poll());//虚引用并没有入队

        System.gc();
		// Thread.sleep(30);
        System.out.println("GC执行...");

        // GC-> 虚引用->入队->出队->     obj
        System.out.println(queue.poll());//虚引用延迟到了第二次gc时入队

        System.gc();
		// Thread.sleep(30);
        System.out.println("GC执行...");

        // GC-> 虚引用->入队->出队->     obj
        System.out.println(queue.poll());

        System.gc();
        Thread.sleep(30);
        System.out.println("GC执行...");

        // GC-> 虚引用->入队->出队->     obj
        System.out.println(queue.poll());
    }
}
```

final class Finalizer extends FinalReference：最终引用

构造方法()  -> 析构函数()，在java中存在Finalizer  可以帮我们自动的回收一些不需要的对象，因此不需要写析构函数。   

jvm能够直接操作的是：非直接内存 

直接内存：native （操作系统中的内存，而不是jvm内存）

jvm不能操作 直接内存（非jvm操作的内容）时，而恰好 此区域的内容 又忘了关闭，此时Finalizer就会将这些内存进行回收。

### 使用软引用实现缓存的淘汰策略

常见的淘汰策略见OS缓存那一块：

- FIFO：先进先出
- LIFO：后进先出
- LRU：近期最少用★★★★★
- OPT：最佳置换算法，无法实现，理论上的而已，只用做和其他算法做比较。

java ->缓存( 90% ->60%)  ->   db(iphone)

LRU

一般的淘汰策略：

根据容量/缓存个数 + LRU 进行淘汰。

在java中 还可以用引用实现 淘汰策略。

MyObject obj = new MyObject();//强引用，不会被GC回收

map.put( id   ,   obj ) ;

Map.put(id,  软引用(obj) )；//当jvm内存不足时，会主动回收。

```java
package ref;

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Map;

class MyObject10{}

public class SoftDemo {

    //map: key:id  ,value:对象的软引用  （拿对象： 对象的软引用 .get() ）
    Map<String, SoftReference<MyObject10>> caches = new HashMap<>();
    //java -> caches -> db
    //set: db->caches
    //get: java->cache

    void setCaches(String id,MyObject10 obj){
        caches.put( id,   new SoftReference<MyObject10>(obj) );

    }

    MyObject10 getCache(String id){
        SoftReference<MyObject10> softRef = caches.get(id) ;
       return  softRef == null ?  null : softRef.get()  ;
    }
    //优势：当jvm内存不足时，gc会自动回收软引用。因此本程序 无需考虑 OOM问题。
}
```

-----

# JVM调优

## JVM监测工具

一般排查问题是要用命令行的。

jps: 查看Java进程 （java命令）

- jps 查看java进程的名字和对应id值 

jstat:只能查看当前时刻的内存情况；可以查看到 新生代、老年代中的内存使用情况

- `jstat -gcutil java进程id`

jmap：查看堆内存的占用情况；也可以执行dump操作

- `jmap -heap java进程id`

jconsole: 图形的监控界面

​	例如：如果通过jconsole中的"执行gc"按钮发现 GC回收的内存太少，就说明当前进程是存在问题的（至少是可以被优化的）

jvisualvm:  监视 - 堆Dump -查找最大对象，从中可以发现 当前进程中是哪个对象 占据了最大的内存，从而对这个对象进行分析。

`jmap -dump:file=d:\abc java进程id`

通过VM参数实现： 当内存溢出时，自动将溢出时刻的内存dump下来。

```shell
-Xmx100m
-Xms100m
-XX:+HeapDumpOnOutOfMemoryError `On OutOfMemoryError 当内存溢出的时候把内存dump下来`
```

## GC调优

Java开发者为什么不把所有的参数调到最优？非得让我们手工去调？【这话说的】

取舍。

调优实际是是一种取舍，以xx换xx的策略。因此在调优之前，必须明确方向：低延迟？高吞吐量呢？

有两种情况需要考虑：

-  在已知条件相同的前提下， 牺牲低延迟 来换取 高吞吐量，或者反之。

- 随着软件硬件技术的发展，可能 二者都升高。

GC的发展：

JVM自身在GC上进行了很多次的改进升级：

JVM默认的GC:  CMS GC（在jdk9以后被逐步废弃） -> G1 GC(jdk9) -> Z GC(jdk11)

- Serial GC:

  串行GC，是一种在单核环境下的串行回收器。当GC回收的时刻，其他线程必须等待。一般不会使用。

- Parallel GC：

  在Serial 的基础上，使用 了多线程技术。 提高吞吐量。

- CMS GC

  CMS使用了多线程技术，使用“标记-清除”算法，可以极大提升效率 （尤其在低延迟上有很大的提升）。繁琐，参数太多，对开发者的经验要求太高。

- G1 GC

  jdk9开始使用的默认GC。特点：将堆内存划分为很多大小相等region，并会对这些区域的使用状态进行标记。以便GC在回收时，能够快速的定位出哪些region是空闲的，哪些是有垃圾对象，从而提升GC的效率。G1使用的算法是“标记-整理”算法。

- ZGC

  jdk11开始提供全新的GC。回收TB级别的垃圾 在毫秒范围。

**常用的垃圾回收算法：（JVM书）**

如果从生命周期角度划分，GC也可以划分成：Minor GC，和Full GC

- Minor GC：回收新生代中的对象
- Full  GC：回收整个堆空间（新生代、老年代）

案例：

如果通过监测工具发现： Minor GC和Full GC都在频繁的回收，如何优化？

Minor GC为什么会频繁执行？因为 新生代中的对象太多了  Minor GC->短生命周期的对象太多了->造成逃逸到老年代中的对象越多->  新生代多+老年代多->Full GC 【新生代多，幸存的就多（概率上），老年代对象就多！Full GC就触发的多！】

Minor GC:可以尝试调大 新生代的最大空间

再调大 新生代晋升到老年代的阈值，从而降低  短生命周期的对象 从新生代转移到老年代的概率。

# 编译优化

结合编译原理中的一些东西进行学习~

# GC和GC Tuning

作者：马士兵教育 http://mashibing.com

### GC的基础知识

#### 1.什么是垃圾

> C语言申请内存：malloc free
>
> C++： new delete
>
> c/C++ 手动回收内存
>
> Java: new ？
>
> 自动内存回收，编程上简单，系统不容易出错，手动释放内存，容易出两种类型的问题：
>
> 1. 忘记回收
> 2. 多次回收

没有任何引用指向的一个对象或者多个对象（循环引用）

#### 2.如何定位垃圾

1. 引用计数（ReferenceCount）
2. 根可达算法(RootSearching)

#### 3.常见的垃圾回收算法

1. 标记清除(mark sweep) - 位置不连续 产生碎片 效率偏低（两遍扫描）
2. 拷贝算法 (copying) - 没有碎片，浪费空间
3. 标记压缩(mark compact) - 没有碎片，效率偏低（两遍扫描，指针需要调整）

#### 4.JVM内存分代模型（用于分代垃圾回收算法）

1. 部分垃圾回收器使用的模型

   > 除Epsilon ZGC Shenandoah之外的GC都是使用逻辑分代模型
   >
   > G1是逻辑分代，物理不分代
   >
   > 除此之外不仅逻辑分代，而且物理分代

2. 新生代 + 老年代 + 永久代（1.7）Perm Generation/ 元数据区(1.8) Metaspace

   1. 永久代 元数据 - Class
   2. 永久代必须指定大小限制 ，元数据可以设置，也可以不设置，无上限（受限于物理内存）
   3. 字符串常量 1.7 - 永久代，1.8 - 堆
   4. MethodArea逻辑概念 - 永久代、元数据

3. 新生代 = Eden + 2个suvivor区 

   1. YGC回收之后，大多数的对象会被回收，活着的进入s0
   2. 再次YGC，活着的对象eden + s0 -> s1
   3. 再次YGC，eden + s1 -> s0
   4. 年龄足够 -> 老年代 （15 CMS 6）
   5. s区装不下 -> 老年代

4. 老年代

   1. 顽固分子
   2. 老年代满了FGC Full GC

5. GC Tuning (Generation)

   1. 尽量减少FGC
   2. MinorGC = YGC
   3. MajorGC = FGC

6. 对象分配过程图
   ![](D:\开学先看这些\对象分配过程详解.png)

7. 动态年龄：（不重要）
   https://www.jianshu.com/p/989d3b06a49d

8. 分配担保：（不重要）
   YGC期间 survivor区空间不够了 空间担保直接进入老年代
   参考：https://cloud.tencent.com/developer/article/1082730

#### 5.常见的垃圾回收器

![常用垃圾回收器](D:\开学先看这些\常用垃圾回收器.png)

1. 垃圾回收器的发展路线，是随着内存越来越大的过程而演进							
   从分代算法演化到不分代算法
   Serial算法 几十兆
   Parallel算法 几个G
   CMS 几十个G  - 承上启下，开始并发回收 -
   .- 三色标记 - 
2. JDK诞生 Serial追随 提高效率，诞生了PS，为了配合CMS，诞生了PN，CMS是1.4版本后期引入，CMS是里程碑式的GC，它开启了并发回收的过程，但是CMS毛病较多，因此目前任何一个JDK版本默认是CMS
   并发垃圾回收是因为无法忍受STW
3. Serial 年轻代 串行回收
4. PS 年轻代 并行回收
5. ParNew 年轻代 配合CMS的并行回收
6. SerialOld 
7. ParallelOld
8. ConcurrentMarkSweep 老年代 并发的， 垃圾回收和应用程序同时运行，降低STW的时间(200ms)
   CMS问题比较多，所以现在没有一个版本默认是CMS，只能手工指定
   CMS既然是MarkSweep，就一定会有碎片化的问题，碎片到达一定程度，CMS的老年代分配对象分配不下的时候，使用SerialOld 进行老年代回收
   想象一下：
   PS + PO -> 加内存 换垃圾回收器 -> PN + CMS + SerialOld（几个小时 - 几天的STW）
   几十个G的内存，单线程回收 -> G1 + FGC 几十个G -> 上T内存的服务器 ZGC
   算法：三色标记 + Incremental Update
9. G1(200ms - 10ms)
   算法：三色标记 + SATB
10. ZGC (10ms - 1ms) PK C++
      算法：ColoredPointers + LoadBarrier
11. Shenandoah
    算法：ColoredPointers + WriteBarrier
12. Eplison
13. PS 和 PN区别的延伸阅读：
    ▪[https://docs.oracle.com/en/java/javase/13/gctuning/ergonomics.html#GUID-3D0BB91E-9BFF-4EBB-B523-14493A860E73](https://docs.oracle.com/en/java/javase/13/gctuning/ergonomics.html)
14. 垃圾收集器跟内存大小的关系
    1. Serial 几十兆
    2. PS 上百兆 - 几个G
    3. CMS - 20G
    4. G1 - 上百G
    5. ZGC - 4T - 16T（JDK13）

1.8默认的垃圾回收：PS + ParallelOld

### 常见垃圾回收器组合参数设定：(1.8)

* -XX:+UseSerialGC = Serial New (DefNew) + Serial Old
  * 小型程序。默认情况下不会是这种选项，HotSpot会根据计算及配置和JDK版本自动选择收集器
* -XX:+UseParNewGC = ParNew + SerialOld
  * 这个组合已经很少用（在某些版本中已经废弃）
  * https://stackoverflow.com/questions/34962257/why-remove-support-for-parnewserialold-anddefnewcms-in-the-future
* -XX:+UseConc<font color=red>(urrent)</font>MarkSweepGC = ParNew + CMS + Serial Old
* -XX:+UseParallelGC = Parallel Scavenge + Parallel Old (1.8默认) 【PS + SerialOld】
* -XX:+UseParallelOldGC = Parallel Scavenge + Parallel Old
* -XX:+UseG1GC = G1
* Linux中没找到默认GC的查看方法，而windows中会打印UseParallelGC 
  * java +XX:+PrintCommandLineFlags -version
  * 通过GC的日志来分辨

* Linux下1.8版本默认的垃圾回收器到底是什么？

  * 1.8.0_181 默认（看不出来）Copy MarkCompact
  * 1.8.0_222 默认 PS + PO

### JVM调优第一步，了解JVM常用命令行参数

* JVM的命令行参数参考：https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html

* HotSpot参数分类

  > 标准： - 开头，所有的HotSpot都支持
  >
  > 非标准：-X 开头，特定版本HotSpot支持特定命令
  >
  > 不稳定：-XX 开头，下个版本可能取消

  java -version

  java -X

  

  java -XX:+PrintFlagsWithComments //只有debug版本能用

  

  试验用程序：

  ```java
  import java.util.List;
  import java.util.LinkedList;
  
  public class HelloGC {
    public static void main(String[] args) {
      System.out.println("HelloGC!");
      List list = new LinkedList();
      for(;;) {
        byte[] b = new byte[1024*1024];
        list.add(b);
    }
    }
  }
  ```

  

  1. 区分概念：内存泄漏memory leak，内存溢出out of memory
  2. java -XX:+PrintCommandLineFlags HelloGC
  3. java -Xmn10M -Xms40M -Xmx60M -XX:+PrintCommandLineFlags -XX:+PrintGC  HelloGC
     PrintGCDetails PrintGCTimeStamps PrintGCCauses
  4. java -XX:+UseConcMarkSweepGC -XX:+PrintCommandLineFlags HelloGC
  5. java -XX:+PrintFlagsInitial 默认参数值
  6. java -XX:+PrintFlagsFinal 最终参数值
  7. java -XX:+PrintFlagsFinal | grep xxx 找到对应的参数
  8. java -XX:+PrintFlagsFinal -version |grep GC
  9. java -XX:+PrintFlagsFinal -version | wc -l 
     共728个参数

### PS GC日志详解

每种垃圾回收器的日志格式是不同的！

PS日志格式

![GC日志详解](D:\开学先看这些\GC日志详解.png)

heap dump部分：

```java
eden space 5632K, 94% used [0x00000000ff980000,0x00000000ffeb3e28,0x00000000fff00000)
                            后面的内存地址指的是，起始地址，使用空间结束地址，整体空间结束地址
```

![GCHeapDump](D:\开学先看这些\GCHeapDump.png)

total = eden + 1个survivor

### 调优前的基础概念：

1. 吞吐量：用户代码时间 /（用户代码执行时间 + 垃圾回收时间）
2. 响应时间：STW越短，响应时间越好

所谓调优，首先确定，追求啥？吞吐量优先，还是响应时间优先？还是在满足一定的响应时间的情况下，要求达到多大的吞吐量...

问题：

科学计算，吞吐量。数据挖掘，thrput。吞吐量优先的一般：（PS + PO）

响应时间：网站 GUI API （1.8 G1）

### 什么是调优？

1. 根据需求进行JVM规划和预调优
2. 优化运行JVM运行环境（慢，卡顿）
3. 解决JVM运行过程中出现的各种问题(OOM)

### 调优，从规划开始

* 调优，从业务场景开始，没有业务场景的调优都是耍流氓

* 无监控（压力测试，能看到结果），不调优

* 步骤：

  1. 熟悉业务场景（没有最好的垃圾回收器，只有最合适的垃圾回收器）
     1. 响应时间、停顿时间 [CMS G1 ZGC] （需要给用户作响应）
     2. 吞吐量 = 用户时间 /( 用户时间 + GC时间) [PS]
  2. 选择回收器组合
  3. 计算内存需求（经验值 1.5G 16G）
  4. 选定CPU（越高越好）
  5. 设定年代大小、升级年龄
  6. 设定日志参数
     1. -Xloggc:/opt/xxx/logs/xxx-xxx-gc-%t.log -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=5 -XX:GCLogFileSize=20M -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintGCCause
     2. 或者每天产生一个日志文件
  7. 观察日志情况

* 案例1：垂直电商，最高每日百万订单，处理订单系统需要什么样的服务器配置？

  > 这个问题比较业余，因为很多不同的服务器配置都能支撑(1.5G 16G)
  >
  > 1小时360000集中时间段， 100个订单/秒，（找一小时内的高峰期，1000订单/秒）
  >
  > 经验值，
  >
  > 非要计算：一个订单产生需要多少内存？512K * 1000 500M内存
  >
  > 专业一点儿问法：要求响应时间100ms
  >
  > 压测！

* 案例2：12306遭遇春节大规模抢票应该如何支撑？

  > 12306应该是中国并发量最大的秒杀网站：
  >
  > 号称并发量100W最高
  >
  > CDN -> LVS -> NGINX -> 业务系统 -> 每台机器1W并发（10K问题） 100台机器
  >
  > 普通电商订单 -> 下单 ->订单系统（IO）减库存 ->等待用户付款
  >
  > 12306的一种可能的模型： 下单 -> 减库存 和 订单(redis kafka) 同时异步进行 ->等付款
  >
  > 减库存最后还会把压力压到一台服务器
  >
  > 可以做分布式本地库存 + 单独服务器做库存均衡
  >
  > 大流量的处理方法：分而治之

* 怎么得到一个事务会消耗多少内存？

  > 1. 弄台机器，看能承受多少TPS？是不是达到目标？扩容或调优，让它达到
  >
  > 2. 用压测来确定

### 优化环境

1. 有一个50万PV的资料类网站（从磁盘提取文档到内存）原服务器32位，1.5G
   的堆，用户反馈网站比较缓慢，因此公司决定升级，新的服务器为64位，16G
   的堆内存，结果用户反馈卡顿十分严重，反而比以前效率更低了
   1. 为什么原网站慢?
      很多用户浏览数据，很多数据load到内存，内存不足，频繁GC，STW长，响应时间变慢
   2. 为什么会更卡顿？
      内存越大，FGC时间越长
   3. 咋办？
      PS -> PN + CMS 或者 G1
2. 系统CPU经常100%，如何调优？(面试高频)
   CPU100%那么一定有线程在占用系统资源，
   1. 找出哪个进程cpu高（top）
   2. 该进程中的哪个线程cpu高（top -Hp）
   3. 导出该线程的堆栈 (jstack)
   4. 查找哪个方法（栈帧）消耗时间 (jstack)
   5. 工作线程占比高 | 垃圾回收线程占比高
3. 系统内存飙高，如何查找问题？（面试高频）
   1. 导出堆内存 (jmap)
   2. 分析 (jhat jvisualvm mat jprofiler ... )
4. 如何监控JVM
   1. jstat jvisualvm jprofiler arthas top...

### 解决JVM运行中的问题

#### 一个案例理解常用工具

1. 测试代码：

   ```java
   package com.mashibing.jvm.gc;
   
   import java.math.BigDecimal;
   import java.util.ArrayList;
   import java.util.Date;
   import java.util.List;
   import java.util.concurrent.ScheduledThreadPoolExecutor;
   import java.util.concurrent.ThreadPoolExecutor;
   import java.util.concurrent.TimeUnit;
   
   /**
    * 从数据库中读取信用数据，套用模型，并把结果进行记录和传输
    */
   
   public class T15_FullGC_Problem01 {
   
       private static class CardInfo {
           BigDecimal price = new BigDecimal(0.0);
           String name = "张三";
           int age = 5;
           Date birthdate = new Date();
   
           public void m() {}
       }
   
       private static ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(50,
               new ThreadPoolExecutor.DiscardOldestPolicy());
   
       public static void main(String[] args) throws Exception {
           executor.setMaximumPoolSize(50);
   
           for (;;){
               modelFit();
               Thread.sleep(100);
           }
       }
   
       private static void modelFit(){
           List<CardInfo> taskList = getAllCardInfo();
           taskList.forEach(info -> {
               // do something
               executor.scheduleWithFixedDelay(() -> {
                   //do sth with info
                   info.m();
   
               }, 2, 3, TimeUnit.SECONDS);
           });
       }
   
       private static List<CardInfo> getAllCardInfo(){
           List<CardInfo> taskList = new ArrayList<>();
   
           for (int i = 0; i < 100; i++) {
               CardInfo ci = new CardInfo();
               taskList.add(ci);
           }
   
           return taskList;
       }
   }
   
   ```

2. java -Xms200M -Xmx200M -XX:+PrintGC com.mashibing.jvm.gc.T15_FullGC_Problem01

3. 一般是运维团队首先受到报警信息（CPU Memory）

4. top命令观察到问题：内存不断增长 CPU占用率居高不下

5. top -Hp 观察进程中的线程，哪个线程CPU和内存占比高

6. jps定位具体java进程
   jstack 定位线程状况，重点关注：WAITING BLOCKED
   eg.
   waiting on <0x0000000088ca3310> (a java.lang.Object)
   假如有一个进程中100个线程，很多线程都在waiting on <xx> ，一定要找到是哪个线程持有这把锁
   怎么找？搜索jstack dump的信息，找<xx> ，看哪个线程持有这把锁RUNNABLE
   作业：1：写一个死锁程序，用jstack观察 2 ：写一个程序，一个线程持有锁不释放，其他线程等待

7. 为什么阿里规范里规定，线程的名称（尤其是线程池）都要写有意义的名称
   怎么样自定义线程池里的线程名称？（自定义ThreadFactory）

8. jinfo pid 

9. jstat -gc 动态观察gc情况 / 阅读GC日志发现频繁GC / arthas观察 / jconsole/jvisualVM/ Jprofiler（最好用）
   jstat -gc 4655 500 : 每个500个毫秒打印GC的情况
   如果面试官问你是怎么定位OOM问题的？如果你回答用图形界面（错误）
   1：已经上线的系统不用图形界面用什么？（cmdline arthas）
   2：图形界面到底用在什么地方？测试！测试的时候进行监控！（压测观察）

10. jmap - histo 4655 | head -20，查找有多少对象产生

11. jmap -dump:format=b,file=xxx pid ：

    线上系统，内存特别大，jmap执行期间会对进程产生很大影响，甚至卡顿（电商不适合）
    1：设定了参数HeapDump，OOM的时候会自动产生堆转储文件（不是很专业，因为多有监控，内存增长就会报警）
    2：<font color='red'>很多服务器备份（高可用），停掉这台服务器对其他服务器不影响</font>
    3：在线定位(一般小点儿公司用不到)

    4：在测试环境中压测（产生类似内存增长问题，在堆还不是很大的时候进行转储）

12. java -Xms20M -Xmx20M -XX:+UseParallelGC -XX:+HeapDumpOnOutOfMemoryError com.mashibing.jvm.gc.T15_FullGC_Problem01

13. 使用MAT / jhat /jvisualvm 进行dump文件分析
     https://www.cnblogs.com/baihuitestsoftware/articles/6406271.html 
    jhat -J-mx512M xxx.dump
    http://192.168.17.11:7000
    拉到最后：找到对应链接
    可以使用OQL查找特定问题对象

14. 找到代码的问题

#### jconsole远程连接

1. 程序启动加入参数：

   > ```shell
   > java -Djava.rmi.server.hostname=192.168.17.11 -Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=11111 -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false XXX
   > ```

2. 如果遭遇 Local host name unknown：XXX的错误，修改/etc/hosts文件，把XXX加入进去

   > ```java
   > 192.168.17.11 basic localhost localhost.localdomain localhost4 localhost4.localdomain4
   > ::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
   > ```

3. 关闭linux防火墙（实战中应该打开对应端口）

   > ```shell
   > service iptables stop
   > chkconfig iptables off #永久关闭
   > ```

4. windows上打开 jconsole远程连接 192.168.17.11:11111

#### jvisualvm远程连接

 https://www.cnblogs.com/liugh/p/7620336.html （简单做法）

#### jprofiler (收费)

#### arthas在线排查工具

* 为什么需要在线排查？
  在生产上我们经常会碰到一些不好排查的问题，例如线程安全问题，用最简单的threaddump或者heapdump不好查到问题原因。为了排查这些问题，有时我们会临时加一些日志，比如在一些关键的函数里打印出入参，然后重新打包发布，如果打了日志还是没找到问题，继续加日志，重新打包发布。对于上线流程复杂而且审核比较严的公司，从改代码到上线需要层层的流转，会大大影响问题排查的进度。 
* jvm观察jvm信息
* thread定位线程问题
* dashboard 观察系统情况
* heapdump + jhat分析
* jad反编译
  动态代理生成类的问题定位
  第三方的类（观察代码）
  版本问题（确定自己最新提交的版本是不是被使用）
* redefine 热替换
  目前有些限制条件：只能改方法实现（方法已经运行完成），不能改方法名， 不能改属性
  m() -> mm()
* sc  - search class
* watch  - watch method
* 没有包含的功能：jmap

### GC算法的基础概念

* Card Table
  由于做YGC时，需要扫描整个OLD区，效率非常低，所以JVM设计了CardTable， 如果一个OLD区CardTable中有对象指向Y区，就将它设为Dirty，下次扫描时，只需要扫描Dirty Card
  在结构上，Card Table用BitMap来实现

### CMS

#### CMS的问题

1. Memory Fragmentation

   > -XX:+UseCMSCompactAtFullCollection
   > -XX:CMSFullGCsBeforeCompaction 默认为0 指的是经过多少次FGC才进行压缩

2. Floating Garbage

   > Concurrent Mode Failure
   > 产生：if the concurrent collector is unable to finish reclaiming the unreachable objects before the tenured generation fills up, or if an allocation cannot be satisfiedwith the available free space blocks in the tenured generation, then theapplication is paused and the collection is completed with all the applicationthreads stopped
   >
   > 解决方案：降低触发CMS的阈值
   >
   > PromotionFailed
   >
   > 解决方案类似，保持老年代有足够的空间
   >
   > –XX:CMSInitiatingOccupancyFraction 92% 可以降低这个值，让CMS保持老年代足够的空间

#### CMS日志分析

执行命令：java -Xms20M -Xmx20M -XX:+PrintGCDetails -XX:+UseConcMarkSweepGC com.mashibing.jvm.gc.T15_FullGC_Problem01

[GC (Allocation Failure) [ParNew: 6144K->640K(6144K), 0.0265885 secs] 6585K->2770K(19840K), 0.0268035 secs] [Times: user=0.02 sys=0.00, real=0.02 secs] 

> ParNew：年轻代收集器
>
> 6144->640：收集前后的对比
>
> （6144）：整个年轻代容量
>
> 6585 -> 2770：整个堆的情况
>
> （19840）：整个堆大小



```java
[GC (CMS Initial Mark) [1 CMS-initial-mark: 8511K(13696K)] 9866K(19840K), 0.0040321 secs] [Times: user=0.01 sys=0.00, real=0.00 secs] 
	//8511 (13696) : 老年代使用（最大）
	//9866 (19840) : 整个堆使用（最大）
[CMS-concurrent-mark-start]
[CMS-concurrent-mark: 0.018/0.018 secs] [Times: user=0.01 sys=0.00, real=0.02 secs] 
	//这里的时间意义不大，因为是并发执行
[CMS-concurrent-preclean-start]
[CMS-concurrent-preclean: 0.000/0.000 secs] [Times: user=0.00 sys=0.00, real=0.00 secs] 
	//标记Card为Dirty，也称为Card Marking
[GC (CMS Final Remark) [YG occupancy: 1597 K (6144 K)][Rescan (parallel) , 0.0008396 secs][weak refs processing, 0.0000138 secs][class unloading, 0.0005404 secs][scrub symbol table, 0.0006169 secs][scrub string table, 0.0004903 secs][1 CMS-remark: 8511K(13696K)] 10108K(19840K), 0.0039567 secs] [Times: user=0.00 sys=0.00, real=0.00 secs] 
	//STW阶段，YG occupancy:年轻代占用及容量
	//[Rescan (parallel)：STW下的存活对象标记
	//weak refs processing: 弱引用处理
	//class unloading: 卸载用不到的class
	//scrub symbol(string) table: 
		//cleaning up symbol and string tables which hold class-level metadata and 
		//internalized string respectively
	//CMS-remark: 8511K(13696K): 阶段过后的老年代占用及容量
	//10108K(19840K): 阶段过后的堆占用及容量

[CMS-concurrent-sweep-start]
[CMS-concurrent-sweep: 0.005/0.005 secs] [Times: user=0.00 sys=0.00, real=0.01 secs] 
	//标记已经完成，进行并发清理
[CMS-concurrent-reset-start]
[CMS-concurrent-reset: 0.000/0.000 secs] [Times: user=0.00 sys=0.00, real=0.00 secs]
	//重置内部结构，为下次GC做准备
```



### G1

1. ▪https://www.oracle.com/technical-resources/articles/java/g1gc.html

#### G1日志详解

```java
[GC pause (G1 Evacuation Pause) (young) (initial-mark), 0.0015790 secs]
//young -> 年轻代 Evacuation-> 复制存活对象 
//initial-mark 混合回收的阶段，这里是YGC混合老年代回收
   [Parallel Time: 1.5 ms, GC Workers: 1] //一个GC线程
      [GC Worker Start (ms):  92635.7]
      [Ext Root Scanning (ms):  1.1]
      [Update RS (ms):  0.0]
         [Processed Buffers:  1]
      [Scan RS (ms):  0.0]
      [Code Root Scanning (ms):  0.0]
      [Object Copy (ms):  0.1]
      [Termination (ms):  0.0]
         [Termination Attempts:  1]
      [GC Worker Other (ms):  0.0]
      [GC Worker Total (ms):  1.2]
      [GC Worker End (ms):  92636.9]
   [Code Root Fixup: 0.0 ms]
   [Code Root Purge: 0.0 ms]
   [Clear CT: 0.0 ms]
   [Other: 0.1 ms]
      [Choose CSet: 0.0 ms]
      [Ref Proc: 0.0 ms]
      [Ref Enq: 0.0 ms]
      [Redirty Cards: 0.0 ms]
      [Humongous Register: 0.0 ms]
      [Humongous Reclaim: 0.0 ms]
      [Free CSet: 0.0 ms]
   [Eden: 0.0B(1024.0K)->0.0B(1024.0K) Survivors: 0.0B->0.0B Heap: 18.8M(20.0M)->18.8M(20.0M)]
 [Times: user=0.00 sys=0.00, real=0.00 secs] 
//以下是混合回收其他阶段
[GC concurrent-root-region-scan-start]
[GC concurrent-root-region-scan-end, 0.0000078 secs]
[GC concurrent-mark-start]
//无法evacuation，进行FGC
[Full GC (Allocation Failure)  18M->18M(20M), 0.0719656 secs]
   [Eden: 0.0B(1024.0K)->0.0B(1024.0K) Survivors: 0.0B->0.0B Heap: 18.8M(20.0M)->18.8M(20.0M)], [Metaspace: 38
76K->3876K(1056768K)] [Times: user=0.07 sys=0.00, real=0.07 secs]

```



### 案例汇总

OOM产生的原因多种多样，有些程序未必产生OOM，不断FGC(CPU飙高，但内存回收特别少) （上面案例）

1. 硬件升级系统反而卡顿的问题（见上）

2. 线程池不当运用产生OOM问题（见上）
   不断的往List里加对象（实在太LOW）

3. smile jira问题
   实际系统不断重启
   解决问题 加内存 + 更换垃圾回收器 G1
   真正问题在哪儿？不知道

4. tomcat http-header-size过大问题（Hector）

5. lambda表达式导致方法区溢出问题(MethodArea / Perm Metaspace)
   LambdaGC.java     -XX:MaxMetaspaceSize=9M -XX:+PrintGCDetails

   ```java
   "C:\Program Files\Java\jdk1.8.0_181\bin\java.exe" -XX:MaxMetaspaceSize=9M -XX:+PrintGCDetails "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2019.1\lib\idea_rt.jar=49316:C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2019.1\bin" -Dfile.encoding=UTF-8 -classpath "C:\Program Files\Java\jdk1.8.0_181\jre\lib\charsets.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\deploy.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\access-bridge-64.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\cldrdata.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\dnsns.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\jaccess.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\jfxrt.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\localedata.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\nashorn.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\sunec.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\sunjce_provider.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\sunmscapi.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\sunpkcs11.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\ext\zipfs.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\javaws.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\jce.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\jfr.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\jfxswt.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\jsse.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\management-agent.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\plugin.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\resources.jar;C:\Program Files\Java\jdk1.8.0_181\jre\lib\rt.jar;C:\work\ijprojects\JVM\out\production\JVM;C:\work\ijprojects\ObjectSize\out\artifacts\ObjectSize_jar\ObjectSize.jar" com.mashibing.jvm.gc.LambdaGC
   [GC (Metadata GC Threshold) [PSYoungGen: 11341K->1880K(38400K)] 11341K->1888K(125952K), 0.0022190 secs] [Times: user=0.00 sys=0.00, real=0.00 secs] 
   [Full GC (Metadata GC Threshold) [PSYoungGen: 1880K->0K(38400K)] [ParOldGen: 8K->1777K(35328K)] 1888K->1777K(73728K), [Metaspace: 8164K->8164K(1056768K)], 0.0100681 secs] [Times: user=0.02 sys=0.00, real=0.01 secs] 
   [GC (Last ditch collection) [PSYoungGen: 0K->0K(38400K)] 1777K->1777K(73728K), 0.0005698 secs] [Times: user=0.00 sys=0.00, real=0.00 secs] 
   [Full GC (Last ditch collection) [PSYoungGen: 0K->0K(38400K)] [ParOldGen: 1777K->1629K(67584K)] 1777K->1629K(105984K), [Metaspace: 8164K->8156K(1056768K)], 0.0124299 secs] [Times: user=0.06 sys=0.00, real=0.01 secs] 
   java.lang.reflect.InvocationTargetException
   	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
   	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
   	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
   	at java.lang.reflect.Method.invoke(Method.java:498)
   	at sun.instrument.InstrumentationImpl.loadClassAndStartAgent(InstrumentationImpl.java:388)
   	at sun.instrument.InstrumentationImpl.loadClassAndCallAgentmain(InstrumentationImpl.java:411)
   Caused by: java.lang.OutOfMemoryError: Compressed class space
   	at sun.misc.Unsafe.defineClass(Native Method)
   	at sun.reflect.ClassDefiner.defineClass(ClassDefiner.java:63)
   	at sun.reflect.MethodAccessorGenerator$1.run(MethodAccessorGenerator.java:399)
   	at sun.reflect.MethodAccessorGenerator$1.run(MethodAccessorGenerator.java:394)
   	at java.security.AccessController.doPrivileged(Native Method)
   	at sun.reflect.MethodAccessorGenerator.generate(MethodAccessorGenerator.java:393)
   	at sun.reflect.MethodAccessorGenerator.generateSerializationConstructor(MethodAccessorGenerator.java:112)
   	at sun.reflect.ReflectionFactory.generateConstructor(ReflectionFactory.java:398)
   	at sun.reflect.ReflectionFactory.newConstructorForSerialization(ReflectionFactory.java:360)
   	at java.io.ObjectStreamClass.getSerializableConstructor(ObjectStreamClass.java:1574)
   	at java.io.ObjectStreamClass.access$1500(ObjectStreamClass.java:79)
   	at java.io.ObjectStreamClass$3.run(ObjectStreamClass.java:519)
   	at java.io.ObjectStreamClass$3.run(ObjectStreamClass.java:494)
   	at java.security.AccessController.doPrivileged(Native Method)
   	at java.io.ObjectStreamClass.<init>(ObjectStreamClass.java:494)
   	at java.io.ObjectStreamClass.lookup(ObjectStreamClass.java:391)
   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1134)
   	at java.io.ObjectOutputStream.defaultWriteFields(ObjectOutputStream.java:1548)
   	at java.io.ObjectOutputStream.writeSerialData(ObjectOutputStream.java:1509)
   	at java.io.ObjectOutputStream.writeOrdinaryObject(ObjectOutputStream.java:1432)
   	at java.io.ObjectOutputStream.writeObject0(ObjectOutputStream.java:1178)
   	at java.io.ObjectOutputStream.writeObject(ObjectOutputStream.java:348)
   	at javax.management.remote.rmi.RMIConnectorServer.encodeJRMPStub(RMIConnectorServer.java:727)
   	at javax.management.remote.rmi.RMIConnectorServer.encodeStub(RMIConnectorServer.java:719)
   	at javax.management.remote.rmi.RMIConnectorServer.encodeStubInAddress(RMIConnectorServer.java:690)
   	at javax.management.remote.rmi.RMIConnectorServer.start(RMIConnectorServer.java:439)
   	at sun.management.jmxremote.ConnectorBootstrap.startLocalConnectorServer(ConnectorBootstrap.java:550)
   	at sun.management.Agent.startLocalManagementAgent(Agent.java:137)
   
   ```

6. 直接内存溢出问题（少见）
   《深入理解Java虚拟机》P59，使用Unsafe分配直接内存，或者使用NIO的问题

7. 栈溢出问题
   -Xss设定太小

8. 比较一下这两段程序的异同，分析哪一个是更优的写法：

   ```java 
   Object o = null;
   for(int i=0; i<100; i++) {
       o = new Object();
       //业务处理
   }
   ```

   ```java
   for(int i=0; i<100; i++) {
       Object o = new Object();
   }
   ```

9. 重写finalize引发频繁GC
   小米云，HBase同步系统，系统通过nginx访问超时报警，最后排查，C++程序员重写finalize引发频繁GC问题
   为什么C++程序员会重写finalize？（new delete）
   finalize耗时比较长（200ms）

10. 如果有一个系统，内存一直消耗不超过10%，但是观察GC日志，发现FGC总是频繁产生，会是什么引起的？
    System.gc() (这个比较Low)

11. Distuptor有个可以设置链的长度，如果过大，然后对象大，消费完不主动释放，会溢出 (来自 死物风情)

12. 用jvm都会溢出，mycat用崩过，1.6.5某个临时版本解析sql子查询算法有问题，9个exists的联合sql就导致生成几百万的对象（来自 死物风情）

13. new 大量线程，会产生 native thread OOM，（low）应该用线程池，
    解决方案：减少堆空间（太TMlow了）,预留更多内存产生native thread
    JVM内存占物理内存比例 50% - 80%

14. 近期学生案例SQLLite的类库，批处理的时候会把所有的结果加载内存，有的人一下子更新几十万条数据，结果就产生了内存溢出，定位上用的是排除法，去掉这个模块就没问题，加上该模块就会出问题

15. java在线解压以及压缩文件造成的内存溢出

16. java使用opencv造成的卡顿与缓慢

17. 最容易引起崩溃的报表系统

18. 分库分表所引起的系统崩溃


### GC常用参数

* -Xmn -Xms -Xmx -Xss
  年轻代 最小堆 最大堆 栈空间
* -XX:+UseTLAB
  使用TLAB，默认打开
* -XX:+PrintTLAB
  打印TLAB的使用情况
* -XX:TLABSize
  设置TLAB大小
* -XX:+DisableExplictGC
  System.gc()不管用 ，FGC
* -XX:+PrintGC
* -XX:+PrintGCDetails
* -XX:+PrintHeapAtGC
* -XX:+PrintGCTimeStamps
* -XX:+PrintGCApplicationConcurrentTime (低)
  打印应用程序时间
* -XX:+PrintGCApplicationStoppedTime （低）
  打印暂停时长
* -XX:+PrintReferenceGC （重要性低）
  记录回收了多少种不同引用类型的引用
* -verbose:class
  类加载详细过程
* -XX:+PrintVMOptions
* -XX:+PrintFlagsFinal  -XX:+PrintFlagsInitial
  必须会用
* -Xloggc:opt/log/gc.log
* -XX:MaxTenuringThreshold
  升代年龄，最大值15
* 锁自旋次数 -XX:PreBlockSpin 热点代码检测参数-XX:CompileThreshold 逃逸分析 标量替换 ... 
  这些不建议设置

### Parallel常用参数

* -XX:SurvivorRatio
* -XX:PreTenureSizeThreshold
  大对象到底多大
* -XX:MaxTenuringThreshold
* -XX:+ParallelGCThreads
  并行收集器的线程数，同样适用于CMS，一般设为和CPU核数相同
* -XX:+UseAdaptiveSizePolicy
  自动选择各区大小比例

### CMS常用参数

* -XX:+UseConcMarkSweepGC
* -XX:ParallelCMSThreads
  CMS线程数量
* -XX:CMSInitiatingOccupancyFraction
  使用多少比例的老年代后开始CMS收集，默认是68%(近似值)，如果频繁发生SerialOld卡顿，应该调小，（频繁CMS回收）
* -XX:+UseCMSCompactAtFullCollection
  在FGC时进行压缩
* -XX:CMSFullGCsBeforeCompaction
  多少次FGC之后进行压缩
* -XX:+CMSClassUnloadingEnabled
* -XX:CMSInitiatingPermOccupancyFraction
  达到什么比例时进行Perm回收
* GCTimeRatio
  设置GC时间占用程序运行时间的百分比
* -XX:MaxGCPauseMillis
  停顿时间，是一个建议时间，GC会尝试用各种手段达到这个时间，比如减小年轻代

### G1常用参数

* -XX:+UseG1GC
* -XX:MaxGCPauseMillis
  建议值，G1会尝试调整Young区的块数来达到这个值
* -XX:GCPauseIntervalMillis
  ？GC的间隔时间
* -XX:+G1HeapRegionSize
  分区大小，建议逐渐增大该值，1 2 4 8 16 32。
  随着size增加，垃圾的存活时间更长，GC间隔更长，但每次GC的时间也会更长
  ZGC做了改进（动态区块大小）
* G1NewSizePercent
  新生代最小比例，默认为5%
* G1MaxNewSizePercent
  新生代最大比例，默认为60%
* GCTimeRatio
  GC时间建议比例，G1会根据这个值调整堆空间
* ConcGCThreads
  线程数量
* InitiatingHeapOccupancyPercent
  启动G1的堆空间占用比例



#### 作业

1. -XX:MaxTenuringThreshold控制的是什么？
   A: 对象升入老年代的年龄
     	B: 老年代触发FGC时的内存垃圾比例

2. 生产环境中，倾向于将最大堆内存和最小堆内存设置为：（为什么？）
   A: 相同 B：不同

3. JDK1.8默认的垃圾回收器是：
   A: ParNew + CMS
     	B: G1
     	C: PS + ParallelOld
     	D: 以上都不是

4. 什么是响应时间优先？

5. 什么是吞吐量优先？

6. ParNew和PS的区别是什么？

7. ParNew和ParallelOld的区别是什么？（年代不同，算法不同）

8. 长时间计算的场景应该选择：A：停顿时间 B: 吞吐量

9. 大规模电商网站应该选择：A：停顿时间 B: 吞吐量

10. HotSpot的垃圾收集器最常用有哪些？

11. 常见的HotSpot垃圾收集器组合有哪些？

12. JDK1.7 1.8 1.9的默认垃圾回收器是什么？如何查看？

13. 所谓调优，到底是在调什么？

14. 如果采用PS + ParrallelOld组合，怎么做才能让系统基本不产生FGC

15. 如果采用ParNew + CMS组合，怎样做才能够让系统基本不产生FGC

     1.加大JVM内存

     2.加大Young的比例

     3.提高Y-O的年龄

     4.提高S区比例

     5.避免代码内存泄漏

16. G1是否分代？G1垃圾回收器会产生FGC吗？

17. 如果G1产生FGC，你应该做什么？

        1. 扩内存
        2. 提高CPU性能（回收的快，业务逻辑产生对象的速度固定，垃圾回收越快，内存空间越大）
        3. 降低MixedGC触发的阈值，让MixedGC提早发生（默认是45%）

 18. 问：生产环境中能够随随便便的dump吗？
     小堆影响不大，大堆会有服务暂停或卡顿（加live可以缓解），dump前会有FGC

 19. 问：常见的OOM问题有哪些？
     栈 堆 MethodArea 直接内存

 20. **如果JVM进程静悄悄退出怎么办？**

       1. JVM自身OOM导致
          1. heap dump on oom，这种最容易解决
       2. JVM自身故障
          1. -XX:ErrorFile=/var/log/hs_err_pid<pid>.log 超级复杂的文件 包括：crash线程信息 safepoint信息 锁信息 native code cache , 编译事件, gc相关记录 jvm内存映射 等等
       3. 被Linux OOM killer杀死
          1. 日志位于/var/log/messages
          2. egrep -i 'killed process' /var/log/messages
       4. 硬件或内核问题
          1. dmesg | grep java
       5. **找我！**

 21. **如何排查直接内存**？

       1. NMT打开 -- -XX:NativeMemoryTracking=detail
       2. perf工具
       3. gperftools

22. 有哪些常用的日志分析工具？

    1. gceasy

23. CPU暴增如何排查？

    1. top -Hp  jstack 
    2. arthas - dashboard thread thread XXXX
    3. 两种情况：1：业务线程 2：GC线程 - GC日志

24. 死锁如何排查？

    1. jstack 观察线程情况
    2. arthas -  thread -b

### 参考资料

1. [https://blogs.oracle.com/](https://blogs.oracle.com/jonthecollector/our-collectors)[jonthecollector](https://blogs.oracle.com/jonthecollector/our-collectors)[/our-collectors](https://blogs.oracle.com/jonthecollector/our-collectors)
2. https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html
3. http://java.sun.com/javase/technologies/hotspot/vmoptions.jsp
4. JVM调优参考文档：https://docs.oracle.com/en/java/javase/13/gctuning/introduction-garbage-collection-tuning.html#GUID-8A443184-7E07-4B71-9777-4F12947C8184 
5. https://www.cnblogs.com/nxlhero/p/11660854.html 在线排查工具
6. https://www.jianshu.com/p/507f7e0cc3a3 arthas常用命令
7. Arthas手册：
   1. 启动arthas java -jar arthas-boot.jar
   2. 绑定java进程
   3. dashboard命令观察系统整体情况
   4. help 查看帮助
   5. help xx 查看具体命令帮助
8. jmap命令参考： https://www.jianshu.com/p/507f7e0cc3a3 
   1. jmap -heap pid
   2. jmap -histo pid
   3. jmap -clstats pid
9. https://blog.csdn.net/chenssy/article/details/78271744 分析hotspot error file



