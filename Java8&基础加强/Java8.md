# 引言
- 大处着眼，小处着手
- 逆向思维、反证法
- 透过问题看本质

**小不忍则乱大谋**

**识时务者为俊杰**

**适当看点经济学的书**

# 概览
> **并行流与串行流**

把一个内容分成多个数据块，并用不同的线程分别处理每个数据块的流。相比串行流，并行流可以很大程度上提高程序的执行效率。【要不要考虑并发安全问题？】

Java8中Stream API可以声明性地通过parallel()和sequential()在并行流与顺序流直接进行切换。

> **减少空指针**

最大化减少空指针异常：Optional


>**运行js**

Nashorn引擎，允许在JVM上运行JS应用。
----

# 二、lambda表达式
> 语言简洁，代码紧凑。==可将其理解为是一段可以传递的代码==

## 2.1 入门案例使用
> **入门案例**
```java
public static void main(String[] args) {
    Runnable r = new Runnable() {
        @Override
        public void run() {
            System.err.println("hello runnable");
        }
    };
    r.run();

    Runnable r2 = () -> System.out.println("hello lambda");
    r2.run();
}
```
> **不同写法**

```java
// 匿名内部类实现接口
Comparator<Integer> com1 = new Comparator<Integer>(){
    @Override
    public int compare(Integer o1, Integer o2) {
        return o1 - o2;
    }
};

// lambda表达式
Comparator<Integer> com2 = (a, b) -> {
    return a - b;
};

// 方法引用
Comparator<Integer> com3 = Integer::compare;
```

## 2.2 lambda写法简单解释
> 1、举例：

(o1, o2) -> Integer.compare(o1,o2);

> 2、格式：

->:lambda操作符 或 箭头操作符

->左边：lambda形参列表（其实就是接口中的抽象方法的形参列表）

->右边：lambda体 （其实就是重写的抽象方法的方法体）

> 3、lambda表达式的使用：分为六种情况介绍

==总结：==

->左边：lambda形参列表参数类型可以省略（类型判断），只有一个形式参数时（）可以省略

->右边：lambda体应该使用{}包裹，如果只有一条执行语句（可能是return语句），则可以省略这一对{}和return关键字

> 4、lambda表达式本质：

作为接口的实例。接口中只能有一个抽象方法【不会有歧义，所以方法名称可以省略】
```java
@FunctionalInterface
public interface lambda4 {
    void method();
}
// 不写注解也行，但是写上会有检查，校验合法不合法！
// 这个就报错了，因为只能有一个抽象方法。【可以有default方法】
@FunctionalInterface
public interface lambda4 {
    void method();

    void method2(){
        System.err.println(123);
    };
}

// 这个是正确的
@FunctionalInterface
public interface lambda4 {
    void method();

    public default void method2() {
        System.err.println(123);
    }

}
```
----
> **举例**

**无参，无返回值**
```java
Runnable r2 = () -> System.out.println("hello lambda");
```

**需要一个参数，但是无返回值**
```java
Consumer<String> con1 = new Consumer<String>() {
    @Override
    public void accept(String t) {
        System.out.println("accept" + t);
    }
};

con1.accept(" hello");
Consumer<String> con2 = (String value) -> {
    System.out.println(value);
};
con2.accept("hello");
```

**数据类型可以省略，由编译器推断，称为类型判断**
```java
Consumer<String> con2 = (value) -> {
    System.out.println(value);
};
con2.accept("hello");
```

**只有一个参数，参数小括号可以省略**
```java
Consumer<String> con2 = value -> { System.out.println(value); };
```

**需要两个或以上的参数，多条执行语句，且有返回值**
```java
public static void main(String[] args) {
    Comparator<Integer> c = (o1, o2) -> {
        System.out.println(o1);
        System.out.println(o2);
        return o1 - o2;
    };
}
```

**lambda体只有一条语句，return 与大括号若有，都可省略**
```java
public static void main(String[] args) {
    Comparator<Integer> c2 = (o1, o2) -> o1 - o2;
}
```

----
# 三、四大函数式接口
> 只包含一个抽象方法的接口，称为函数式接口。java.util.function包下定义了java8丰富的函数式接口。

> 函数式接口就是，你把这个接口当成形式参数传递过去，你在方法里用了这个接口的方法，你需要使用这个方法就需要去实现；实现可以用匿名内部类实现或者函数式实现这个方法。


## 3.1 四大函数式接口简介
- 消费型接口：void Consumer<T> 
    - 对类型为T的对象应用操作，包含方法
    - void accept(T t)
- 供给型接口：T Supplier<T>
    - 返回类型为T的对象，包含方法
    - T get()
- 函数型接口：R Function<T,R>
    - 对类型为T的对象应用操作，并返回结果。结果是R类型的对象。包含方法
    - R apply(T t)
- 断定型接口：boolean Predicate<T>
    - 确定类型为T的对象是否满足某约束，并返回boolean值。包含方法
    - boolean test(T t)

> 当发现所需要定义的接口满足以上某一个时，我们就不用自己定义接口了，用jdk给我们提供的就可以了。

## 3.1.1 消费型接口：Consumer
> 只需要消费对象，无需返回值。

```java
public class ConsumerDemo {
    public static void main(String[] args) {
        happy(10.0, new Consumer<Double>() {
            @Override
            public void accept(Double t) {
                System.err.println("I get the money = " + t);
            }
        });
        // 函数式接口写法
        happy(23.2, money -> System.out.println("I get the money = " + money));
        
    }

    // 本质con就是一个对象，我们需要传入一个对象，可以用匿名内部类实现或者lambda表达式
    public static void happy(double money, Consumer<Double> con) {
        con.accept(money);
    }
}
```

## 3.1.2 断定型接口 Predicate
> 就是判断是否符合要求
```java
public class PredicateDemo {
    public static void main(String[] args) {

        List<Integer> list = Arrays.asList(12, 234, 56, 31, 23, 54, 34);
        // 匿名内部类
        filterNumber(list, new Predicate<Integer>() {

            @Override
            public boolean test(Integer t) {
                return t % 2 == 0;
            }

        });

        // lambda写法 我们只用到了predicate的test方法
        filterNumber(list, s -> s % 2 == 0);
    }

    // 根据给定规则 过滤数据，方法时Predicate中的抽象方法
    public static List<Integer> filterNumber(List<Integer> list, Predicate<Integer> predicate) {
        List<Integer> arrayList = new ArrayList<>();
        for (Integer number : list) {
            if (predicate.test(number)) {
                arrayList.add(number);
            }
        }
        return arrayList;
    }
}
```

----
# 四、方法引用与构造器引用
> 当要传递给lambda体的操作，已经有实现的方法了，可以使用方法引用！本质上就是lambda表达式
## 4.1 概述
**使用格式**

类（或对象）::方法名

**具体分为如下三种情况**
- 对象::非静态方法
- 类::静态方法
- 类::非静态方法【居然可以这样做！】

**方法引用的使用要求**

接口中的抽象方法的形參列表和返回值类型与方法引用的方法的形參列表和返回值类型相同

## 4.2 方法引用案例
> 例子

```java
public class Demo1 {
    public static void main(String[] args) {
        test5();
    }

    // 对象::非静态方法
    public static void test1() {
        // accept(T t)
        // println(T t)
        // 参数一致，参数都省了。。。。
        Consumer<String> c = System.out::println;
        c.accept("hello");
    }

    // 对象::非静态方法
    public static void test2() {
        Employee employee = new Employee("123", 25620);

        // supplier T get()
        // employee T getName()
        Supplier<String> sup1 = () -> employee.getName();
        Supplier<String> sup2 = employee::getName;

        System.out.println(sup1.get());
        System.out.println(sup2.get());
    }

    // 类::静态方法
    public static void test3() {
        Comparator<Integer> c1 = (a, b) -> a - b;
        Comparator<Integer> c2 = (a, b) -> Integer.compare(a, b);

        // int compare(T o1, T o2);
        // public static int compare(int x, int y)
        // 形参列表一致
        Comparator<Integer> c3 = Integer::compare;

    }

    // 类::静态方法 T泛型，R返回值类型
    public static void test4() {
        // Function<T, R>中的R apply(T t)
        // Mathi中的Long round(Double d)
        Function<Double, Long> fn1 = d -> Math.round(d);
        Function<Double, Long> fn2 = Math::round;
        System.err.println(fn2.apply(15.6));
    }

    // 类::实例方法
    // Comparator 中的int comapre(T t1, T t2)
    // String 中的int t1.compareTo(t2)
    public static void test5() {
        Comparator<String> c2 = (t1, t2) -> t1.compareTo(t2);

        Comparator<String> c1 = String::compareTo;
        System.err.println(c1.compare("abc", "abd"));
    }
}
```

## 4.3 构造器引用案例
> **语法**

```java
类名::new
数组类型[]::new 把数组当成特殊的对象
```
```java
public class Construct {
    public static void main(String[] args) {
        Supplier<Employee> sup = Employee::new;
        // Employee::new;重写了get方法，所以调用get方法才会创建对象
        sup.get();

        // 调用了一个String参数的构造方法
        Function<String, Employee> fun = Employee::new;
        fun.apply("kkx");

        // 调用了两个参数的构造方法
        BiFunction<String, Integer, Employee> bf = Employee::new;
        bf.apply("ljw", 23);

        // 数组引用
        Function<Integer, String[]> fnn1 = len -> new String[len];
        Function<Integer, String[]> fnn2 = String[]::new;
    }
}
```

# 五、强大的Stream API
> Stream API对集合数据进行操作

> Stream和Collection集合的区别。Collection是一种静态的内存数据结构，而Stream是有关计算的。前者是面向内存的，存储在内存中，后者主要是面向CPU，通过CPU实现计算。

**集合讲的数据，Stream讲的计算**

## 5.1 概述
- 创建Stream
- 中间操作
- 终止操作；一旦执行终止操作，就执行中间操作链，并产生结果，之后不在被使用。

## 5.2 Stream的使用
> **特点：**
- 不会自己存储元素
- 不会改变元素对象
- 延迟执行；只有你调用了终止操作，中间的操作才会执行。

### 5.2.1 Stream实例化
> **通过集合实例化**

Collection集合中定义了Stream方法。可返回顺序流和并行流

```java
public class StreamDemo {
    public static void main(String[] args) {
        ArrayList<String> arrayList = new ArrayList<String>();
        arrayList.add("!@3");
        arrayList.add("!@32");
        arrayList.add("!@233");
        arrayList.add("!@43");
        arrayList.add("!@543");
        // 返回顺序流
        Stream<String> stream = arrayList.stream();
        // 返回并行流
        Stream<String> parallelStream = arrayList.parallelStream();
    }
}
```

> **通过数组实例化**

```java
public static void test1() {
    int[] arr = new int[] { 1, 23, 45, 56, 7, 8 };
    IntStream stream = Arrays.stream(arr);
    stream.forEach(System.out::println);

    Employee e1 = new Employee();
    Employee e2 = new Employee();
    Employee e3 = new Employee();
    Employee[] arre = new Employee[] { e1, e2, e3 };
    Arrays.stream(arre).forEach(System.out::println);
}
```

> **通过Stream的of方法**

```java
public static void test2() {
    Stream<Integer> of = Stream.of(123, 32, 4445, 56, 67, 23, 34, 45);
    of.forEach(System.out::println);
}
```

> **创建无限流**：可用于造数据。

```java
// 迭代方式
public static void test3() {
    Stream.iterate(0, t -> t + 2).limit(10).forEach(System.out::println);
}

// 生成方式
public static void test4() {
    Stream.generate(Math::random).limit(20).forEach(System.out::println);
}
```

### 5.2.2 Stream的中间操作

> **筛选与切片**

- filter(Predicate p)	接收lambda，从流中排除某些元素。

- limit(n)	截断流，使其元素不超过给定数量。
- skip(n)     跳过元素，返回一个扔掉了前n个元素的流。若流中元素不足n个，则返回**空**
- distinct()   通过流所生成元素的 hashCode() 和equals（）去除重复元素。

> **映射**

| `方法`                            | `描述`                                                       |
| --------------------------------- | ------------------------------------------------------------ |
| `map(Function f)`                 | 接收一个函数作为参数，该函数会被应用到每个元素上，并将其映射成一个新的元素。**【把某个属性整理成一个Stream】** |
| `mapToDouble(ToDoubleFunction f)` | 接收一个函数作为参数，该函数会被应用到每个元素上，产生一个新的`DoubleStream`。 |
| `mapToInt(ToIntFunction f)`       | 接收一个函数作为参数，该函数会被应用到每个元素上，产生一个新的`IntStream`。 |
| `mapToLong(ToLongFunction f)`     | 接收一个函数作为参数，该函数会被应用到每个元素上，产生一个新的`LongStream`。 |
| `flatMap(Function f)`             | 接收一个函数作为参数，**将流中的每个值都换成另一个流，然后把所有流连接成一个流** |

**map示例代码**

```java
public static void mapDemo() {
    // map 把符合条件的数据归为一类
    Stream<Employee> stream = list.stream();
    // 提取出对象中的某种属性
    Stream<Integer> ageStream = stream.map(e -> e.getAge());
    // 对该属性就行筛选
    ageStream.filter(s -> s % 2 == 0).forEach(System.out::println);

    // 筛选姓名长度大于三的人，把这些人的名字输出
    list.stream().filter(s -> s.getName().length() > 3).forEach(s -> {
        System.err.println(s.getName());
    });
    System.out.println();
    // 或者用map做
    list.stream().map(e -> e.getName()).filter(s -> s.length() > 3).forEach(System.out::println);
}
```

**flatMap**

flatMap和map的区别是：map会把每个元素都当成一个流，而flatMap是把所有的元素拆分开了，在把他们当作一个整体的流。

```java
public static void floatMapDemo() {
    // map 把符合条件的数据归为一类
    Stream<String> stream = Arrays.asList("aa", "bb", "cc").stream();
    Stream<Character> flatMap = stream.flatMap(Middle::fromStringtoChar);
    flatMap.forEach(System.out::println);
}

public static Stream<Character> fromStringtoChar(String str) {
    ArrayList<Character> arrayList = new ArrayList<Character>();
    for (Character character : str.toCharArray()) {
        arrayList.add(character);
    }
    return arrayList.stream();
}
```

**mapToInt之类的**

把xx对象按指定条件转为int类型

```java
public static void mapToInt() {
    Stream<String> stream = Arrays.asList("aa", "bb", "cc").stream();
    // 首字母ASCII码大于68的转为数字99，否则为-1
    stream.mapToInt(s -> {
        if (s.charAt(0) > 68)
            return 99;
        return -1;
    }).forEach(System.out::println);;
}
```

> **排序**

| 方法                   | 描述                                             |
| ---------------------- | ------------------------------------------------ |
| sorted()               | 产生一个新流，按自然顺序排序**【默认从小到大】** |
| sorted(Comparator com) | 产生一个新流，按比较器顺序排序                   |

> **Java层面，涉及到对象的排序就要去想Comparable和Comparator**。排序默认都是从小到大。比对的减法一改，就是从大到小了

```java
public class SortDemo {
    private static ArrayList<Employee> list = null;
    static {
        list = new ArrayList<>();
        list.add(new Employee("name", 123));
        list.add(new Employee("ljw", 23));
        list.add(new Employee("lh", 34));
        list.add(new Employee("ljwg", 56));
        list.add(new Employee("lhf", 72));
        list.add(new Employee("ai", 23));
        list.add(new Employee("haha", 23));
        list.add(new Employee("gg", 45));
    }

    public static void srotDemo1() {
        List<Integer> asList = Arrays.asList(1, 23, 56, 32, -23, -23, 45, -24);
        asList.stream().sorted().forEach(System.out::println);

        list.stream().sorted((e1, e2) -> Integer.compare(e1.getAge(), e2.getAge())).forEach(System.out::println);
    }

    public static void main(String[] args) {
        srotDemo1();
    }
}
```

### 5.2.3 Stream的终止操作

> **匹配与查找**

| 方法名                       | 描述                     |
| ---------------------------- | ------------------------ |
| `allMatch(Predicate p)`      | 检查是否匹配所有元素     |
| `anyMatch(Predicate p)`      | 检查是否至少匹配一个元素 |
| `noneMatch(Predicate p)`     | 检查是否没有匹配所有元素 |
| `findFirst` 返回Optional容器 | 返回第一个元素           |
| `findAny`                    | 返回流中的任意元素       |
| `count`                      | 返回流中元素总个数       |
| `max(Comparator c)`          | 返回流中最大值           |
| `min(Comparator c)`          | 返回流中最小值           |
| `forEach(Consumer c)`        | 内部迭代                 |

> **归约操作**

| 方法名                               | 描述                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `reduce(T identity, BinaryOperator)` | 将流中的元素反复结合起来，得到一个值。返回T           |
| `reduce(BinaryOperator b)`           | 将流中的元素反复结合起来，得到一个值。返回Optional<T> |

map和reduce的连接称为map-reduce。

```java
public class ReduceDemo {
    public static void main(final String[] args) {
        fn1();
    }

    public static void fn1() {
        final List<Integer> asList = Arrays.asList(1, 23, 4, 5, 6, 74, 234, 5, 45);
        final Integer reduce = asList.stream().reduce(0, Integer::sum);
        System.out.println(reduce);
        Optional<Integer> reduce2 = asList.stream().reduce(Integer::sum);
        System.out.println(reduce2.get());

    }
}
```

> **收集**：如收集List，Set，Map

| 方法                   | 描述                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `collect(Collector c)` | 将流转为其他形式。接收一个Collector接口的实现，用于给Stream中元素做汇总的方法 |

Collector接口中方法的实现决定了如何对流执行收集操作。

Collectors实现类提供了很多静态方法，可以方便地创建常见收集器实例。

| 方法           | 返回类型             | 作用                                      |
| -------------- | -------------------- | ----------------------------------------- |
| toList         | List<T>              | 把流中元素收集到List                      |
| toSet          | Set<T>               | 把流中元素收集到Set                       |
| toCollection   | Collection<T>        | 把流中元素收集到创建的集合                |
| counting       | Long                 | 计算流中元素的个数                        |
| summingInt     | Integer              | 对流中元素的整数属性求和                  |
| averagingInt   | Double               | 计算流中元素Integer属性的平均值           |
| summarizingInt | IntSummaryStatistics | 收集流中的Integer属性的统计值。如平均值！ |

> 示例代码

```java
public class Colle {
    public static void main(final String[] args) {
        final List<Integer> asList = Arrays.asList(1, 23, 4, 5, 78, 34, 456, 678, 34456, 234);
        Stream<Integer> stream = asList.stream();
        List<Integer> collect = stream.collect(Collectors.toList());
        collect.forEach(System.out::println);
    }
}
```

## 5.3 Optional类

> 