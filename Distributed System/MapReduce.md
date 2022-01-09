# Google MapReduce

## 摘要

MapReduce是一个编程模型，也是一个处理和生成超大数据集的算法模型的相关实现。用户首先创建一 个Map函数处理一个基于key/value pair的数据集合，输出中间的基于key/value pair的数据集合；然后 再创建一个Reduce函数用来合并所有的具有相同中间key值的中间value值。

MapReduce架构的程序能够在大量的普通配置的计算机上实现并行化处理。这个系统在运行时只关心： <span style="color:green">**如何分割输入数据，在大量计算机组成的集群上的调度，集群中计算机的错误处理，管理集群中计算机之间必要的通信。**</span>采用 MapReduce 架构可以使那些没有并行计算和分布式处理系统开发经验的程序员有效利用分布式系统的丰富资源。

## 介绍

​		需求：需要在可接受的时间内处理大量数据的计算。这只能将计算分布在成百上千的主机上。在分布计算的时候如何处理并行计算、如何分发数据、如何处理错误？这个问题是十分复杂的。
​		将这个复杂的问题抽象为一个模型：MapReduce。我们只要表述我们想要执行的简单运算，不必关心并行计算、容错、数据分布、负载均衡等复杂的细节。
​		我们大多数的运算都包含这样的操作：在输入数据的“逻辑”记录上应用Map操作得出一个中间 key/value pair集合，然后在所有具有相同key值的value值上应用Reduce操作，从而达到合并中间的数据，得到一个想要的结果的目的。使用MapReduce模型，再结合用户实现的Map和Reduce函数，我们就可以非常容易的实现大规模并行化计算；通过MapReduce模型自带的“再次执行”（re-execution）功 能，也提供了初级的容灾实现方案
​		这个工作(实现一个MapReduce框架模型)的主要贡献是通过简单的接口来实现自动的并行化和大规模的分布式计算，通过使用MapReduce模型接口实现在大量普通的PC机上高性能计算。
​		第二部分描述基本的编程模型和一些使用案例。第三部分描述了一个经过裁剪的、适合我们的基于集群的 计算环境的MapReduce实现。第四部分描述我们认为在MapReduce编程模型中一些实用的技巧。第五部 分对于各种不同的任务，测量我们MapReduce实现的性能。第六部分揭示了在Google内部如何使用 MapReduce作为基础重写我们的索引系统产品，包括其它一些使用MapReduce的经验。第七部分讨论相 关的和未来的工作。

## 编程模型

​		用户自定义的Map函数接受一个输入的key/value pair值，然后产生一个中间key/value pair值的集合。 MapReduce库把所有具有相同中间key值I的中间value值集合在一起后传递给reduce函数。
​		MapReduce库的用户用两个函数表达这个计算：Map和Reduce。
​		用户自定义的Map函数接受一个输入的key/value pair值，然后产生一个中间key/value pair值的集合。 MapReduce库把所有具有相同中间key值I的中间value值集合在一起后传递给reduce函数。
​		用户自定义的Reduce函数接受一个中间key的值I和相关的一个value值的集合。Reduce函数合并这些 value值，形成一个较小的value值的集合。一般的，每次Reduce函数调用只产生0或1个输出value值。 通常我们通过一个迭代器把中间value值提供给Reduce函数，这样我们就可以处理无法全部放入内存中的 大量的value值的集合。

### 举例

计算一个大的文档集合中每个单词出现的次数。（是不是每个map计算大文档中的一部分小文档，比如一共有10个key，那么就有10个reduce去计算key，从每个map中拿到的相同的一个key对应的value集合--iterator，对这个Iterator进行累加）

```vim
// Map函数输出文档中的每个词、以及这个词的出现次数(在这个简单的例子里就是1)。
map(String key,String value){
	// key document name
	// value document contents
	for each word w in value:
		EmitIntermediate(w,"1")
}
// Reduce函数把Map函数产生的每一个特定的词的计数累加起来。
reduce(String key,Iterator values){
	// key: a word
	// value: a list of count
	int result = 0;
	for each v in values:
		result += ParseInt(v);
    Emit(AsString(result));
}
```

### 类型

用户定义的 Mao 和 Reduce 函数都有相关联的类型：

```vim
map(k1,v1)->list(k2,v2) // 输入的类型和输出的类型可以不一致
reduce(k2,list(v2))->list(v2) // 输入的类型和输出的类型可以一致
```

### 应用举例

- 分布式的Grep：Map函数输出匹配某个模式的一行，Reduce函数是一个恒等函数，即把中间数据 复制到输出。
- 计算URL访问频率：Map函数处理日志中web页面请求的记录，然后输出(URL,1)。Reduce函数把 相同URL的value值都累加起来，产生(URL,记录总数)结果。
- 倒转网络链接图：Map函数在源页面（source）中搜索所有的链接目标（target）并输出为(target,source)。Reduce函数把给定链接目标（target）的链接组合成一个列表，输出 (target,list(source))。
- 每个主机的检索词向量：检索词向量用一个(词,频率)列表来概述出现在文档或文档集中的最重要的 一些词。Map函数为每一个输入文档输出(主机名,检索词向量)，其中主机名来自文档的URL。 Reduce函数接收给定主机的所有文档的检索词向量，并把这些检索词向量加在一起，丢弃掉低频的 检索词，输出一个最终的(主机名,检索词向量)。 
- 倒排索引：Map函数分析每个文档输出一个(词,文档号)的列表，Reduce函数的输入是一个给定词的所有（词，文档号），排序所有的文档号，输出(词,list（文档号）)。所有的输出集合形成一个简单 的倒排索引，它以一种简单的算法跟踪词在文档中的位置。 
- 分布式排序：Map函数从每个记录提取key，输出(key,record)。Reduce函数不改变任何的值。这个运算依赖分区机制(在4.1描述)和排序属性(在4.2描述)。**==>没明白**

## 实现

MapReduce模型可以用多种实现方式，这里描述的是一个适用于Google内部广泛使用的运算环境的实现：用以太网交换机连接、由普通PC机组成的大型集群。具体的环境如下：

- x86架构、运行Linux操作系统、双处理器、2-4GB内存的机器。
- 普通的网络硬件设备，每个机器的带宽为百兆或者千兆。
- 集群中包含成百上千的机器，因此，机器故障是常态。
- 存储为廉价的内置IDE硬盘。一个内部分布式文件系统用来管理存储在这些磁盘上的数据。文件系统通过数据复制来在不可靠的硬件上保证数据的可靠性和有效性。
- 用户提交工作（job）给调度系统。每个工作（job）都包含一系列的任务（task），调度系统将这些任务调度到集群中多台可用的机器上

### 执行概括

通过将Map调用的输入数据自动分割为M个数据片段的集合，Map调用被分布到多台机器上执行。输入的数据片段能够在不同的机器上并行处理。使用分区函数将Map调用产生的中间key值分成R个不同分区（例如，hash(key) mod R），Reduce调用也被分布到多台机器上执行。分区数量（R）和分区函数由用户来指定

![img](https://mr-dai.github.io/img/mapreduce_summary/mapreduce_architecture.png)

当用户调用MapReduce函数时，将发生下面的一系列动作（下面的序号和图1中的序号一一对应）：

- 1.用户程序首先调用的MapReduce库将输入文件分成M个数据片度，每个数据片段的大小一般从16MB到64MB(可以通过可选的参数来控制每个数据片段的大小)。然后用户程序在机群中创建大量的程序副本。
- 2.这些程序副本中的有一个特殊的程序–master。副本中其它的程序都是worker程序，由master分配任务。有M个Map任务和R个Reduce任务将被分配，master将一个Map任务或Reduce任务分配给一个空闲的worker。
- 3.被分配了map任务的worker程序读取相关的输入数据片段，从输入的数据片段中解析出key/value pair，然后把key/value pair传递给用户自定义的Map函数，由Map函数生成并输出的中间key/value pair，并缓存在内存中。
- 4.缓存中的key/value pair通过分区函数分成R个区域，之后周期性的写入到本地磁盘上。缓存的 key/value pair在本地磁盘上的存储位置将被回传给master，由master负责把这些存储位置再传送给 Reduce worker。
- 5.当Reduce worker程序接收到master程序发来的数据存储位置信息后，使用RPC从Map worker所在主机的磁盘上读取这些缓存数据。当Reduce worker读取了所有的中间数据后，通过对key进行排序后使 得具有相同key值的数据聚合在一起。由于许多不同的key值会映射到相同的Reduce任务上，因此必须进行排序。如果中间数据太大无法在内存中完成排序，那么就要在外部进行排序。
- 6.Reduce worker程序遍历排序后的中间数据，对于每一个唯一的中间key值，Reduce worker程序将这个key值和它相关的中间value值的集合传递给用户自定义的Reduce函数。Reduce函数的输出被追加到所属分区的输出文件。
- 7.当所有的Map和Reduce任务都完成之后，master唤醒用户程序。在这个时候，在用户程序里的对 MapReduce调用才返回。

在成功完成任务之后，MapReduce的输出存放在R个输出文件中（对应每个Reduce任务产生一个输出文 件，文件名由用户指定）。一般情况下，用户不需要将这R个输出文件合并成一个文件–他们经常把这些文件作为另外一个MapReduce的输入，或者在另外一个可以处理多个分割文件的分布式应用中使用

### Master数据结构

Master持有一些数据结构，它存储每一个Map和Reduce任务的状态（空闲、工作中或完成)，以及 Worker机器(非空闲任务的机器)的标识。 
Master就像一个数据管道，中间文件存储区域的位置信息通过这个管道从Map传递到Reduce。因此，对于每个已经完成的Map任务，master存储了Map任务产生的R个中间文件存储区域的大小和位置。当Map 任务完成时，Master接收到位置和大小的更新信息，这些信息被逐步递增的推送给那些正在工作的Reduce任务。

### 容错

因为MapReduce库的设计初衷是使用由成百上千的机器组成的集群来处理超大规模的数据，所以，这个库必须要能很好的处理机器故障。

> worker故障

master周期性的ping每个worker。如果在一个约定的时间范围内没有收到worker返回的信息，master 将把这个worker标记为失效。所有由这个失效的worker完成的Map任务被重设为初始的空闲状态，之后这些任务就可以被安排给其他的worker。同样的，worker失效时正在运行的Map或Reduce任务也将被重新置为空闲状态，等待重新调度。

当worker故障时，由于已经完成的Map任务的输出存储在这台机器上，Map任务的输出已不可访问了，因 此必须重新执行。而已经完成的Reduce任务的输出存储在全局文件系统上，因此不需要再次执行。

当一个Map任务首先被worker A执行，之后由于worker A失效了又被调度到worker B执行，**这个“重新 执行”的动作会被通知给所有执行Reduce任务的worker**。任何还没有从worker A读取数据的Reduce任 务将从worker B读取数据。

MapReduce可以处理大规模worker失效的情况。比如，在一个MapReduce操作执行期间，在正在运行 的集群上进行网络维护引起80台机器在几分钟内不可访问了，MapReduce master只需要简单的再次执 行那些不可访问的worker完成的工作，之后继续执行未完成的任务，直到最终完成这个MapReduce操 作。

