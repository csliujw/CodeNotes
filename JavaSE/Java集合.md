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