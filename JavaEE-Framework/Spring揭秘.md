# Spring IOC

## 基本概念

IOC理念，让别人为你服务。由自己new对象变成别人送对象给你。

IOC常见注入方式

- 构造方法注入（不可被继承，对象一多，参数就长。同类型的参数，反射处理起来还麻烦）
- setter方法注入（可被继承，但是无法在新建对象后就立马注入）
- 接口注入（逐渐被抛弃）

IOC的好处

- 不会对业务对象构成很强的侵入性
- 对象具有更好的可测试性、可重用性和可扩展性， 等等

IoC是一种可以帮助我们解耦各业 10 务对象间依赖关系的对象绑定方式！ 

## IoC Service Provider

IoC Service Provider在这里是一个抽象出来的概念，它可以指代任何将IoC场景中的业务对象绑定 到一起的实现方式。它可以是一段代码，也可以是一组相关的类，甚至可以是比较通用的IoC框架或 者IoC容器实现。

```java
// 依赖绑定代码
IFXNewsListener newsListener = new DowJonesNewsListener();
IFXNewsPersister newsPersister = new DowJonesNewsPersister();
FXNewsProvider newsProvider = new FXNewsProvider(newsListener,newsPersister);
newsProvider.getAndPersistNews(); 
//要将系统中几十、几百甚至数以千计的业务对象绑定到一起，采用这种方式显然是不切实际的
```

### IoC Service Provider的职责

- 业务对象的构建
  - `IoC`场景中，业务对象无需关心所依赖的对象如何构建。`IoC Service Provider`需要将对象的构建逻辑从客户端对象那里剥离出来，以免这部分逻辑污染业务对象的实现。（<span style="color:red">就是不在业务代码中new出对象？</span>）
- 业务对象的依赖绑定
  - `IoC Service Provider`通过结合之前构建和管理的所有业务对象，以及各个业务对象间可以识别的依赖关系， ① 这里指代使用某个对象或者某种服务的对象。如果对象A需要引用对象B，那么A就是B的客户端对象，而不管A 处于Service层还是数据访问层。将这些对象所依赖的对象注入绑定，从而保证每个业务对象在使用的时候，可以处于就绪状态。(<span style="color:red">`Spring`通过加注解或`xml`的配置方式将所依赖的对象注入</span>)

### 管理对象之间的依赖

直接编码方式

```java
IoContainer container = ...;
container.register(FXNewsProvider.class,new FXNewsProvider());
container.register(IFXNewsListener.class,new DowJonesNewsListener());
...
FXNewsProvider newsProvider = (FXNewsProvider)container.get(FXNewsProvider.class);
newProvider.getAndPersistNews(); 
```

配置文件方式

```xml
<bean id="newsProvider" class="..FXNewsProvider">
 <property name="newsListener">
 <ref bean="djNewsListener"/>
 </property>
 <property name="newPersistener">
 <ref bean="djNewsPersister"/>
 </property>
</bean>
<bean id="djNewsListener"
 class="..impl.DowJonesNewsListener">
</bean>
<bean id="djNewsPersister"
 class="..impl.DowJonesNewsPersister">
</bean> 
```

元数据方式

```java
public class FXNewsProvider{
	private IFXNewsListener newsListener;
 	private IFXNewsPersister newPersistener;
 	@Inject // 元数据配置方式~ 加注解~~
	public FXNewsProvider(IFXNewsListener listener,IFXNewsPersister persister){
		this.newsListener = listener;
		this.newPersistener = persister;
	}
} 
```

## `BeanFactory`

Spring提供了两种容器类型：`BeanFactory`和`ApplicationContext`。

- `BeanFactory`。基础类型`IoC`容器，提供完整的`IoC`服务支持。如果没有特殊指定，<span style="color:red">默认采用延迟初始化策略（lazy-load）</span>。只有当客户端对象需要访问容器中的某个受管对象的时候，才对 该受管对象进行初始化以及依赖注入操作。所以，<span style="color:red">相对来说，容器启动初期速度较快，所需要的资源有限。</span>对于资源有限，并且功能要求不是很严格的场景，`BeanFactory`是比较合适的`IoC`容器选择。
- `ApplicationContext`。`ApplicationContext`在`BeanFactory`的基础上构建，是相对比较高 级的容器实现，除了拥有`BeanFactory`的所有支持，`ApplicationContext`还提供了其他高级特性，比如事件发布、国际化信息支持等。`ApplicationContext`所管理 的对象，<span style="color:red">在该类型容器启动之后，默认全部初始化并绑定完成。</span>所以，<span style="color:red">相对于`BeanFactory`来 说，`ApplicationContext`要求更多的系统资源，同时，因为在启动时就完成所有初始化，容器启动时间较之`BeanFactory`也会长一些。</span>在那些系统资源充足，并且要求更多功能的场景中， `ApplicationContext`类型的容器是比较合适的选择。

`ApplicationContext`间接继承自`BeanFactory`

作为Spring提供的基本的`IoC`容器， `BeanFactory`可以完成作为`IoC Service Provider`的所有职责，包括业务对象的注册和对象间依赖关系的绑定。

建议：即使不使用`BeanFactory`之类的轻量级容器支持开发，开发中也应该尽量使用`IoC`模式

**P24截止**