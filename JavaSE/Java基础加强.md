# 一、单元测试

## 1.1 单元测试的有点

保证的程序代码的正确性【语法上了逻辑上】。

## 1.2 单元测试的使用【junit】

- @Test
- @Before 无论Test是否出现异常，都会执行
- @After 无论Test是否出现异常，都会执行

```java
public class JunitDemo {
    private OutputStream outputStream;

    @Before
    public void init() throws FileNotFoundException {
        System.out.println("IO 流初始化完毕了");
        outputStream = new FileOutputStream("junit.txt");
    }

    @Test
    /**
     * 单元测试判断数据的正确性
     * 一般用Assert里面的方法
     */
    public void fn1(){
        // 断言不为null  不是null则成功
        Assert.assertNotNull(outputStream);
    }

    @After
    public void destory() throws IOException {
        System.out.println("IO 流关闭了");
        outputStream.close();
    }
}
```

# 二、反射

## 2.1 反射概述

反射可以把类的各个组成部分封装为其他对象。

## 2.2 反射的基本操作

### 2.2.1 获取成员变量

- `File[] getFileds()` // **获得所有公有字段，包括继承的**
- `Filed getFiled(String name)` // 获取指定name的
- `Filed[] getDeclaredFileds() `// 获取该类自己声明的，包括私有
- `Filed[] getDeclaredFileds(String name)` // 获取指定名称的

#### 2.2.2 获取构造方法

- `Constructor<?>[] getConstructors()` // 获得所有公有构造器

- `Constructor<?> getConstructor(Class<?>...parameterTypes)` //获得指定参数的公有构造器
- `Constructor<?>[]getDeclaredConstructors()`// 获得所有私有构造器
- `Constructor<T>[]getDeclaredConstructors()`//  得指定参数的构造器【包括public~~~private 】

#### 2.2.3 获取成员方法

- `Method[] getMethods()` // **获得所有public修饰的方法，包括继承的**

- `Method getMethod(String name, Class<?>... parameterTypes)` // 获得指定名称和参数类型的public修饰的方法
- `Method[] getDeclaredMethods()` //获得所有的私有方法
- `Method getDeclaredMethod(String name, Class<?>... parameterTypes)` // 获得指定名称和参数类型的方法

#### 2.2.4 获取类名

- `String getName()` // 获得类全名`com.bbxx.junits.Son`

# 三、注解

**注解也叫元数据**。是一种代码级别的说明，JDK1.5引入的特性，与类，接口，枚举是在同一层次。**可声明在包，类，字段，方法，局部变量，方法参数等的前面**，对这些元素进行说明。

**作用分类：**

1.代码分析，通过代码里标识的元数据对代码进行分析【结合反射技术】

2.编译检查，通过代码里标识的元数据让编译器能够实现机泵的编译检查【Override】

3.编写文档，通过代码里标识的元数据生成文档【生成文档doc文档】

## 3.1 内置注解

- `Override`：检测被标注的是否继承自父类
- `Deprecated`：表示方法过时
- `SuppressWarnings`：压制警告
  - 一般传递参数all

## 3.2 自定义注解

> **元注解`public @interface annotationName{}`**

反编译发现，本质就是一个接口。

```java
import java.lang.annotation.Annotation;

public interface Annotation extends Annotation {
}
```

### 3.2.1 注解的属性

#### 3.2.1.1 属性的返回值

- 基本数据类型
- String
- 枚举
- 注解
- 以上类型的数组

#### 3.2.1.2 赋值问题

- 设置默认值`String sex() default "1";`
- 使用注解，数组类型的赋值 `str={xx,xx,xx}`，若数组中只有一个，大括号可省略。回忆Spring中注解

				* 基本数据类型
				* String
				* 枚举
				* 注解
				* 以上类型的数组
## 3.3 元注解

用于描述注解的注解

- `@Target`：描述注解的位置
  - `ElementType`取值
    - TYPE：可以作用于类上
    - METHOD：可以作用于方法上
    - FIELD：可以作用于成员变量上
- `@Retention`：描述注解是被保留的阶段
  - `@Retention(RetentionPolicy.RUNTIME)`：当前被描述的注解，会保留到class字节码文件中，并被`JVM`读取到
- `@Documented`：描述注解是否被抽取到api文档中
- `@Inherited`：描述注解是否被子类继承

## 3.4 注解的解析

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Pro {
    String className();
    String methodName();
}

@Pro(className = "com.bbxx.Demo1",methodName = "show1")
public class RefelectDemo {
    public static void main(String[] args) throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, IllegalAccessException, InstantiationException {
        /**
         * 解析注解
         */
        Class<RefelectDemo> refelectDemoClass = RefelectDemo.class;
        Pro annotation = refelectDemoClass.getAnnotation(Pro.class);
        String s = annotation.className();
        String s1 = annotation.methodName();
        Class<?> aClass = Class.forName(s);
        Object o = aClass.newInstance();
        Method declaredMethod = aClass.getDeclaredMethod(s1);
        declaredMethod.setAccessible(true);
        declaredMethod.invoke(o);
    }
}
```



# 四、类加载器

> Bootstrap类加载器，扩展类加载器，系统类加载器

Bootstrap类加载器，cpp实现

扩展类加载器，java实现。扩展包（ext）下的由它加载。

系统类加载器，java实现。我们写的代码都是由系统类加载器加载的。

## 4.1 类加载器的双亲委派机制

委托父加载器加载，父可以加载就让父加载。父无法加载时再自己加载。

> **优点**

- 可避免类的重复加载，父类加载器已经加载了该类时，就没必要子ClassLoader再加载一次了/
- 考虑到安全因素，java核心api中定义类型不会被随意替换。

## 4.2 ClassLoader

所有的类加载器（除了根类加载器）都必须继承java.lang.ClassLoader.它是一个抽象类，主要方法如下：

### 4.2.1 loadClass

存在父类加载的委托机制

### 4.2.2 findClass



### 4.2.3 defineClass



### 4.2.4 resolveClass

## 4.3 URLClassLoader

![](E:\69546\Documents\image\URLClassload.png)

```java

public class ClassLoadDemo2 {
    public static void main(String[] args) throws Exception {
        // E盘下
        File file = new File("E:/");
        URI uri = file.toURI();
        URL url = uri.toURL();

        URLClassLoader urlClassLoader = new URLClassLoader(new URL[] { url });
        // 类所在的包是com.xxbb  具体路径是 E:/com/xxbb/App.class
        Class<?> loadClass = urlClassLoader.loadClass("com.xxbb.App");
        loadClass.newInstance();
    }
}
```



## 4.4 自定义类加载器

>文件类加载器

```java
public class MyClassLoader extends ClassLoader {
    private String directory;

    public MyClassLoader(String _directory, ClassLoader paraent) {
        super(paraent);
        this.directory = _directory;
    }

    protected Class<?> findClass(String name) throws ClassNotFoundException {
        try {
            // 把类名转为目录
            String file = directory + File.separator + name.replace(".", File.separator) + ".class";
            // 构建输入流
            InputStream fis = new FileInputStream(file);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int len = -1;
            while ((len = fis.read(buf)) != -1) {
                baos.write(buf, 0, len);
            }
            byte[] byteArray = baos.toByteArray();
            fis.close();
            baos.close();

            return defineClass(name, byteArray, 0, byteArray.length);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```



> 网络类加载器

> 热部署，越过双亲委派，就是不用loadClass 用findClass

## 4.5 类的显示与隐式加载

## 4.6 线程上下文类加载器



