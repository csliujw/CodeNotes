# 设计模式原则

## 单一职责原则

做角色访问控制的时候，动作主体（用户）与资源的行为（权限）分离。

单一职责原则的定义是：应该有且仅有一个原因引起类的变更。==通俗来说就是一个接口或类只有一个职责，它就负责一件事情。==

但是单一职责原则最难划分的就是职责，一个职责一个接口，职责该如何量化？这些都是不可度量的，因项目而异，因环境而异。

只能说这个原则尽量遵守，也最容易被打破。

## 里氏替换原则

里氏替换原则的定义是：所有引用基类的地方必须能透明地使用其子类的对象。通俗说就是，只要父类能出现的地方子类就可以出现，而替换为子类也不会产生任何错误或异常。

注意：如果子类不能完整地实现父类的方法，或者父类的某些方法在子类中已经发生“畸变”，则建议断开父子继承关系，采用依赖、聚集、组合等关系替代

## 依赖倒置原则

依赖倒置原则的定义是：

- 高层模不应该依赖低层模块，两者都应该依赖其抽象；
- 抽象不应该依赖细节；
- 细节应该依赖抽象；

在Java中的表现是：

- 模块间的依赖通过抽象发生，实现类之间不能发生直接的依赖关系，其依赖关系是通过接口或抽象类产生的。
- 接口或抽象类不依赖于实现类；
- 实现类依赖接口或抽象类
- 简而言之：面向接口编程

## 接口隔离原则

接口隔离原则的定义：客户端不应该依赖它不需要的接口。依赖它需要的接口，客户端需要什么接口就提供什么接口，把不需要的接口剔除，细化接口，保证接口的纯洁性。

注意：设计是有限度的，不能无限地考虑未来的变更情况，否则会陷入设计的泥潭中而不能自拔。

## 迪米特法则

迪米特法则的定义：一个对象应该对其他对象有最少的了解。一个类应该对自己需要耦合或调用的类直到的最少，你（被耦合或调用的类）的内部是如何复杂都和我没关系，知道它提供了多少public方法即可。

尽量不要对外公布太多的public方法和非静态的public变量，尽量内敛。

> 案例

```java
public class Wizard {
    private Random rand = new Random(System.currentTimeMillis());

    private int first() {
        System.out.println("first");
        return rand.nextInt(100);
    }

    private int second() {
        System.out.println("second");
        return rand.nextInt(100);
    }

    private int third() {
        System.out.println("third");
        return rand.nextInt(100);
    }

    public void installWizard() {
        int first = this.first();
        if (first > 50) {
            int second = this.second();
            if (second > 50) {
                int third = this.third();
                if (third > 50) {
                    System.out.println("install success");
                }
            }
        }
    }
}
```

## 开闭原则

对扩展开放，对关闭修改。是最基础的原则，前面五个原则都是开闭原则的具体形态，前5个原则是指导设计的工具和方法，开闭原则是其核心精神。

# 大纲

<img src="../pics/geek/designPatterns/relationship.png">

# 创建型

## 单例模式

> 定义

确保一个类只有一个实例，而且自行实例化并向整个系统提供这个实例。

> 通用代码

```java
public class Singleton {
    private static final Singleton singleton = new Singleton();

    private Singleton() {
    }

    public static Singleton getInstance() {
        return singleton;
    }
}
```

---

```java
public class DCLSingleton {
    // volatile 防止指令重排序
    private volatile DCLSingleton dclSingleton = null;

    private DCLSingleton() {
    }

    public DCLSingleton getInstance() {
        if (dclSingleton == null) {
            synchronized (DCLSingleton.class) {
                if (dclSingleton == null) {
                    dclSingleton = new DCLSingleton();
                }
            }
        }
        return dclSingleton;
    }
}
```

## 工厂方法模式

定义一个用于创建对象的接口，，让子类决定实例化哪一个类。工厂方法使一个类的实例化延迟到其子类中。

工厂方法模式的变种较多，以下为一个比较实用的通用源码。

```java
public interface Human {
    void getColor();
}

class BlackHuman implements Human {

    @Override
    public void getColor() {
        System.out.println("黑人，黑色皮肤");
    }
}

class YellowHuman implements Human {

    @Override
    public void getColor() {
        System.out.println("黄种人，黄色皮肤");
    }
}
```

---

```java
public abstract class AbstractHumanFactory {
    // 泛型的继承。参数类型得是T，即参数类型要是Human的子类
    public abstract <T extends Human> T createHuman(Class<T> c);
}

public class HumanFactory extends AbstractHumanFactory {
    @Override
    public <T extends Human> T createHuman(Class<T> c) {
        Human human = null;
        try {
            human = (T) Class.forName(c.getName()).newInstance();
        } catch (Exception e) {
            System.out.println("生成 Human 子类对象失败");
        }
        return (T) human;
    }
}
```

----

```java
public class Main {
    public static void main(String[] args) {
        HumanFactory humanFactory = new HumanFactory();
        BlackHuman human = humanFactory.createHuman(BlackHuman.class);
        human.getColor();
    }
}
```



## 建造者模式

----



## 原型模式

# 结构型

## 代理模式

## 桥接模式

## 装饰者模式

## 适配器模式

---

## 门面模式

## 组合模式

## 享元模式

# 行为型

## 观察者模式

## 模板模式

## 策略模式

## 职责链模式

## 迭代器模式

## 状态模式

## 访问者模式

## 备忘录模式

## 命令模式

## 解释器模式

## 中介模式

