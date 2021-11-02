# HashMap

## Java中的 HashMap 的⼯作原理是什么？

HashMap 类有⼀个叫做 Entry 的内部类。这个 Entry 类包含了 key-value 作为实例变量。 每当往 hashmap ⾥⾯存放 key-value 对的时候，都会为它们实例化⼀个 Entry 对象，这个 Entry 对象就会存储在前⾯提到的 Entry 数组 table 中。Entry 具体存在 table 的那个位置是根据 key 的 hashcode() ⽅法计算出来的 hash 值（来决定）。

> 放入元素的源码逻辑

```java
final V putVal(int hash, K key, V value, boolean onlyIfAbsent,
               boolean evict) {
    Node<K,V>[] tab; Node<K,V> p; int n, i;
    // 散列表的table 为 null 或者 散列表长度不够了。
    if ((tab = table) == null || (n = tab.length) == 0)
        n = (tab = resize()).length;
    // 散列表中没有这个 key，就加入这个 key value 进去。
    if ((p = tab[i = (n - 1) & hash]) == null)
        tab[i] = newNode(hash, key, value, null);
    else { // 如果存在这个散列值
        Node<K,V> e; K k;
        if (p.hash == hash && // 查看 hash 值是否相等，
            // 如果 hash 值相等则看 key 是否相同
            ((k = p.key) == key || (key != null && key.equals(k))))
            // 如果发现 hash 和 key 都相同的话，则把原先的节点对象赋值给 e
            e = p;
        else if (p instanceof TreeNode)
            e = ((TreeNode<K,V>)p).putTreeVal(this, tab, hash, key, value);
        else {
            for (int binCount = 0; ; ++binCount) {
                if ((e = p.next) == null) {
                    p.next = newNode(hash, key, value, null);
                    if (binCount >= TREEIFY_THRESHOLD - 1) // -1 for 1st
                        treeifyBin(tab, hash);
                    break;
                }
                if (e.hash == hash &&
                    ((k = e.key) == key || (key != null && key.equals(k))))
                    break;
                p = e;
            }
        }
        if (e != null) { // existing mapping for key
            V oldValue = e.value;
            // 如果可以覆盖 火鹤 旧值为 null
            if (!onlyIfAbsent || oldValue == null)
                e.value = value; // 则将之前的 value 进行覆盖。
            afterNodeAccess(e);
            return oldValue; // 并返回旧值
        }
    }
    ++modCount;
    // 大小超过了就进行 resize
    if (++size > threshold)
        resize();
    afterNodeInsertion(evict);
    return null;
}
```

> 获取元素

```java
public V get(Object key) {
    Node<K,V> e;
    return (e = getNode(hash(key), key)) == null ? null : e.value;
}

// 查找是否有这个结点。
final Node<K,V> getNode(int hash, Object key) {
    Node<K,V>[] tab; Node<K,V> first, e; int n; K k;
    // 如果 table 中有这个元素的话，
    if ((tab = table) != null && (n = tab.length) > 0 &&
        (first = tab[(n - 1) & hash]) != null) { // 此处的判断说明，table 中有这个元素。
        // 开始查找。看第一个元素的 hash 值和 key 是否一样
        if (first.hash == hash && // always check first node
            ((k = first.key) == key || (key != null && key.equals(k))))
            return first;
        // hash 值一样，但是 key 不一样则说明发生了 hash 冲突，查看节点的下一个链表。
        if ((e = first.next) != null) {
            if (first instanceof TreeNode) // 链表过长，HashMap 会 treeify
                return ((TreeNode<K,V>)first).getTreeNode(hash, key);
            do {
                if (e.hash == hash &&
                    ((k = e.key) == key || (key != null && key.equals(k))))
                    return e;
            } while ((e = e.next) != null);
        }
    }
    return null;
}
```

> 扩容策略

扩容为原来的两倍，再旧 table 中的一个一个放进新 table。

```java
final Node<K,V>[] resize() {
    Node<K,V>[] oldTab = table; // 保存旧的 table
    int oldCap = (oldTab == null) ? 0 : oldTab.length; // 保存原先散列表的大小
    int oldThr = threshold;
    int newCap, newThr = 0;
    if (oldCap > 0) {
        if (oldCap >= MAXIMUM_CAPACITY) {
            threshold = Integer.MAX_VALUE;
            return oldTab;
        }
        // 新容量为原先容量的 2 倍
        else if ((newCap = oldCap << 1) < MAXIMUM_CAPACITY &&
                 oldCap >= DEFAULT_INITIAL_CAPACITY)
            newThr = oldThr << 1; // double threshold
    }
    else if (oldThr > 0) // initial capacity was placed in threshold
        newCap = oldThr;
    else {               // zero initial threshold signifies using defaults
        newCap = DEFAULT_INITIAL_CAPACITY;
        newThr = (int)(DEFAULT_LOAD_FACTOR * DEFAULT_INITIAL_CAPACITY);
    }
    if (newThr == 0) {
        float ft = (float)newCap * loadFactor; // 新容量 * 装载因子
        newThr = (newCap < MAXIMUM_CAPACITY && ft < (float)MAXIMUM_CAPACITY ?
                  (int)ft : Integer.MAX_VALUE);
    }
    threshold = newThr; // 更新数组最大容量（放入元素的最大容量 length * loadFactor）
    @SuppressWarnings({"rawtypes","unchecked"})
    Node<K,V>[] newTab = (Node<K,V>[])new Node[newCap];
    table = newTab;
    if (oldTab != null) {
        for (int j = 0; j < oldCap; ++j) {
            Node<K,V> e;
            if ((e = oldTab[j]) != null) { // 取出旧 tab 中的数据
                oldTab[j] = null;
                if (e.next == null) // 可能散列冲突，所有要检查 next
                    newTab[e.hash & (newCap - 1)] = e;
                else if (e instanceof TreeNode)
                    ((TreeNode<K,V>)e).split(this, newTab, j, oldCap);
                else { // preserve order
                    Node<K,V> loHead = null, loTail = null;
                    Node<K,V> hiHead = null, hiTail = null;
                    Node<K,V> next;
                    do {
                        next = e.next;
                        if ((e.hash & oldCap) == 0) {
                            if (loTail == null)
                                loHead = e;
                            else
                                loTail.next = e;
                            loTail = e;
                        }
                        else {
                            if (hiTail == null)
                                hiHead = e;
                            else
                                hiTail.next = e;
                            hiTail = e;
                        }
                    } while ((e = next) != null);
                    if (loTail != null) {
                        loTail.next = null;
                        newTab[j] = loHead;
                    }
                    if (hiTail != null) {
                        hiTail.next = null;
                        newTab[j + oldCap] = hiHead;
                    }
                }
            }
        }
    }
    return newTab;
}
```

## 什么是 HashMap

HashMap 是⼀个散列表，它存储的内容是键值对(key-value)映射。 

HashMap 继承于AbstractMap，实现了 Map、Cloneable、java.io.Serializable 接⼝。 HashMap 的实现不是同步的，这意味着它不是线程安全的。它的key、value都可以为null。此外 HashMap 中的映射不是有序的。

HashMap 的实例有两个参数影响其性能：“初始容量” 和 “加载因⼦”。容量是哈希表中桶的量，初始容量只是哈希表在创建时的容量。加载因⼦是哈希表在其容量⾃动增加之前可以达到多满的⼀种尺度。当哈希表中的条⽬数超出了加载因⼦与当前容量的乘积时，则要对该哈 希表进⾏ rehash 操作（即重建内部数据结构），从⽽哈希表将具有⼤约两倍的桶数。 通常，默认加载因⼦是 0.75, 这是在时间和空间成本上寻求⼀种折衷。加载因⼦过⾼虽然减少了空间开销，但同时也增加了查询成本（在⼤多数 HashMap 类的操作中，包括 get 和 put 操作，都反映了这⼀点）。在设置初始容量时应该考虑到映射中所需的条⽬数及其加载因⼦，以便最⼤限度地减少 rehash 操作次数。如果初始容量⼤于最⼤条⽬数除以加载因⼦，则不会发⽣ rehash 操作。 

> hashmap共有4个构造函数： 

```java
// 默认构造函数。
HashMap() 

// 指定“容量⼤⼩”的构造函数 
HashMap(int capacity) 

// 指定“容量⼤⼩”和“加载因⼦”的构造函数 
HashMap(int capacity, float loadFactor) 

// 包含“⼦Map”的构造函数 
HashMap(Map map)
```

## 如何构造一致性哈希算法

参考回答： 先构造⼀个⻓度为232的整数环（这个环被称为⼀致性Hash环），根据节点名称的Hash值 （其分布为[0, 232-1]）将服务器节点放置在这个Hash环上，然后根据数据的Key值计算得到 其Hash值（其分布也为[0, 232-1]），接着在Hash环上顺时针查找距离这个Key值的Hash值 最近的服务器节点，完成Key到服务器的映射查找。 这种算法解决了普通余数Hash算法伸缩性差的问题，可以保证在上线、下线服务器的情况下 尽量有多的请求命中原来路由到的服务器。

# ArrayList

# 遍历

## 遍历Collection

对List和Set的遍历，有四种方式，下面以ArrayList为例进行说明。

### 1）普通for循环

代码如下：

```
1 for (int i = 0; i < list.size(); i++) {
2     System.out.println(i);
3 }
```

如果要在普通for循环里对集合元素进行删除操作，可能会出现问题：

```java
 1 public static void main(String[] args) {
 2     List<Integer> list = new ArrayList<Integer>();
 3     list.add(1);
 4     list.add(2);
 5     list.add(2);
 6     list.add(4);
 7     list.add(5);
 8     for (int i = 0; i < list.size(); i++) {
 9         if (list.get(i) == 2) {
10             list.remove(i);
11         }
12     }
13     System.out.println(list);
14 }
```

运行结果如下：

```
1 [1, 2, 4, 5]
```

结果说明：

集合中有两个值为2的元素，但是在代码执行之后，值为2的元素并没有完全移除。

原因就在于当第一次判断 i = 1 位置上的值为2时，将这个元素删除，导致这个位置之后的所有元素都向前挪动一个位置，导致 i = 1 位置上的值变成了后面的2。

下次遍历时，判断 i = 2 位置上的值，也就是4，导致2被跳过去了，从而导致最后打印的结果和预期的不一致。

改进方法是在删除之后手动设置 i-- 即可。

### 2）增强for循环

代码如下：

```
1 for (Integer i : list) {
2     System.out.println(i);
3 }
```

如果想在增强for循环里删除或者添加集合元素，那么一定会报异常：

```
 1 public static void main(String[] args) {
 2     List<Integer> list = new ArrayList<Integer>();
 3     list.add(1);
 4     list.add(2);
 5     list.add(2);
 6     list.add(4);
 7     list.add(5);
 8     for (Integer i : list) {
 9         if (i == 2) {
10             list.remove(i);
11         }
12     }
13     System.out.println(list);
14 }
```

运行结果如下：

```
1 java.util.ConcurrentModificationException
```

结果说明：

因为增强for循环（foreach循环）本质上是隐式的iterator，由于在删除和添加的时候会导致modCount发生变化，但是没有重新设置expectedModCount，当你使用list.remove()后遍历执行iterator.next()时，方法检验modCount的值和的expectedModCount值，如果不相等，就会报ConcurrentModificationException。

### 3）使用迭代器

代码如下：

```
1 Iterator<String> iterator = list.iterator();
2 while (iterator.hasNext()) {
3     System.out.println(iterator.next());
4 }
```

如果在迭代的循化里使用list方法的add()方法和remove()方法，同样会报错：

```
 1 public static void main(String[] args) {
 2     List<Integer> list = new ArrayList<Integer>();
 3     list.add(1);
 4     list.add(2);
 5     list.add(2);
 6     list.add(4);
 7     list.add(5);
 8     Iterator<Integer> iterator = list.iterator();
 9     while (iterator.hasNext()) {
10         Integer i = (Integer) iterator.next();
11         if (i == 2) {
12             list.add(6);
13         }
14     }
15 }
```

运行结果如下：

```
1 java.util.ConcurrentModificationException
```

如果在迭代器的循环里使用迭代器的remove()方法，则不会报错。

iterator()方法返回的Iterator类型的迭代器没有提供添加的方法，但是listIterator()方法返回的ListIterator类型的迭代器提供了add()方法和set()方法。

使用迭代器删除原数的代码如下：

```
 1 public static void main(String[] args) {
 2     List<Integer> list = new ArrayList<Integer>();
 3     list.add(1);
 4     list.add(2);
 5     list.add(2);
 6     list.add(4);
 7     list.add(5);
 8     Iterator<Integer> iterator = list.iterator();
 9     while (iterator.hasNext()) {
10         Integer i = (Integer) iterator.next();
11         if (i == 2) {
12             iterator.remove();
13         }
14     }
15     System.out.println(list);
16 }
```

运行结果如下：

```
1 [1, 4, 5]
```

结果说明：

迭代器的remove()方法同时维护了modCount和expectedModCount，所以使用remove()方法可以达到预期的效果。

### 4）使用forEach方法

forEach()方法是JDK1.8新增的方法，需要配合Lambda表达式使用，代码如下：

```
1 list.forEach(String -> System.out.println(String));
```

运行结果如下：

```
1 1
2 2
3 3
4 4
```

## 遍历Map

对Map的遍历，有四种方式，下面以HashMap为例进行说明。

### 1）通过keySet()方法遍历key和value

通过keySet()方法获取到map的所有key值，遍历key值的集合，获取对应的value值。代码如下：

```
1 for (Integer i : map.keySet()) {
2     System.out.println(i + " >>> " + map.get(i));
3 }
```

运行结果如下：

```
1 0 >>> 000
2 1 >>> 111
3 2 >>> 222
4 3 >>> 333
5 4 >>> 444
```

在遍历的时候是可以修改的，但是不能添加和删除，否则会ConcurrentModificationException异常，代码如下：

```
1 for (Integer i : map.keySet()) {
2     System.out.println(i + " >>> " + map.get(i));
3     if (map.get(i) == "222") {
4         map.put(i, "999");
5     }
6 }
```

运行结果如下：

```
1 key= 0 and value= 000
2 key= 1 and value= 111
3 key= 2 and value= 999
4 key= 3 and value= 333
5 key= 4 and value= 444
```

### 2）通过entrySet()方法遍历key和value

这种方式同样支持修改，但不支持添加和删除，这种方式的效率最高，推荐使用，代码如下：

```
1 for (Map.Entry<Integer, String> entry : map.entrySet()) {
2     System.out.println(entry.getKey() + " >>> " + entry.getValue());
3 }
```

### 3）通过entrySet()方法获取迭代器遍历key和value

这种方式同样支持修改，但不支持添加和删除，代码如下：

```
1 Iterator<Map.Entry<Integer, String>> iterator = map.entrySet().iterator();
2 while (iterator.hasNext()) {
3     Map.Entry<Integer, String> entry = iterator.next();
4     System.out.println(entry.getKey() + " >>> " + entry.getValue());
5 }
```

### 4）通过values()方法遍历所有的value

这种方式只能遍历value，不能遍历key，代码如下：

```
1 for (String value : map.values()) {
2     System.out.println(value);
3 }
```