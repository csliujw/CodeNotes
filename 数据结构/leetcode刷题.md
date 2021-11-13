# 刷题注意事项

先理清思路，考虑好边界问题，临时变量问题。

# 链表结点定义

```java
class ListNode {
	int val;
	ListNode next = null;

	ListNode(int val) {
		this.val = val;
	}
}
```

# 链表解题思路

-  设置哑结点简化操作。
- 如链表元素的删除，如果有哑结点，那么删除第一个和第k个元素的操作就是一样的。
-  双指针 三指针 快慢指针
- 双指针找倒数第k个结点
- 三指针删除链表中重复过的所有元素
- 快慢指针判断链表有无换
-  合理使用数据结构：栈，队列，优先队列[堆]等。
- 利用栈找链表公共结点。
- 利用优先队列找出k个最小（大）的元素
-  注意常见的边界问题。
- 如是否为空
- next是否存在等
- 链表的插入删除操作要熟练书写。

# 热身题

> **题目描述**：

> 给定一棵二叉搜索树，请找出其中的第k小的结点。例如,(5，3，7，2，4，6，8)中，按结点数值大小顺序第三小结点的值为4。

> 利用二叉搜索树的性质，中序遍历的结果是从小到大的排序顺序。

```java
TreeNode KthNode(TreeNode pRoot, int k) {
	// 中序遍历非递归算法
	// 遍历规则左根右。
	Stack<TreeNode> stack = new Stack<TreeNode>();
	while(pRoot!= null || !stack.empty()) {
		//左子树一直入栈
		while(pRoot!=null) {
			stack.add(pRoot);
			pRoot = pRoot.left;
		}
		if(!stack.empty()) {
			k--;
			pRoot = stack.pop();
			if(k==0) return pRoot;
			pRoot = pRoot.right;
		}
	}
	return pRoot;
}

 本题链接 https://www.nowcoder.com/practice/ef068f602dde4d28aab2b210e859150a?tpId=13&tqId=11215&tPage=4&rp=4&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking
 扩展练习 https://leetcode-cn.com/problems/validate-binary-search-tree/
 
 * 思考 求第k大的呢？
 * 1.暴力求解。遍历2次，第一次统计数目，第二次求解。
 * 2.用数组存储中序的顺序，然后直接访问。
 * 3.维护一个只含k个数的优先队列（小顶堆。第k大的数正好会在堆顶。），边遍历，边向里面加数据。替换里面的数据。
```

# 链表类

> **输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。**

- 利用栈
- 先单链表逆置在输出

```java
public class Solution6 {
	public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
		Stack<Integer> stack = new Stack<Integer>();
		while (listNode != null) {
			stack.add(listNode.val);
			listNode = listNode.next;
		}
		ArrayList<Integer> arrayList = new ArrayList<Integer>();
		while (!stack.empty()) {
			arrayList.add(stack.pop());
		}
		return arrayList;
	}

	public ArrayList<Integer> printListFromTailToHead2(ListNode listNode) {
		ArrayList<Integer> arrayList = new ArrayList<Integer>();
		// 定义一个哑结点 方便操作
		ListNode head = new ListNode(-1);
		ListNode cur = listNode;
		while (cur != null) {
			ListNode temp = cur.next;
			cur.next = head.next;
			head.next = cur;
			cur = temp;
		}
		head = head.next;
		while (head != null) {
			arrayList.add(head.val);
			head = head.next;
		}
		return arrayList;
	}
}
```

> **输入一个链表，反转链表**

- 定义哑结点 + 头插法

```java
public ListNode ReverseList(ListNode head) {
	ListNode node = new ListNode(-1);
	ListNode cur = head;
	while(cur!=null) {
		ListNode temp = cur.next;
		cur.next = node.next;
		node.next = cur;
		cur = temp;
	}
	return node.next;
}
```

> **输入一个链表，输出该链表中倒数第k个结点。**

- 双指针。pre和cur,开始pre和cur指向同一个结点，然后 cur比pre先走k个结点，在同时移动。当cur==null，pre所指的就是第k个结点。
- 定义哑结点简化操作

```java
public ListNode FindKthToTail(ListNode head,int k) {
	ListNode node = new ListNode(-1);
	node.next = head;
	ListNode pre = node;
	ListNode rear = node;
	while(k!=0&&rear!=null) {
		rear = rear.next;
		k--;
	}
	if(k>=0&&rear==null) return null;
	while(rear!=null) {
		pre=pre.next;rear = rear.next;
	}
	return pre;
}
```

> **输入两个单调递增的链表，输出两个链表合成后的链表，合成后的链表递增【非严格的递增】。**

- 两个链表不为空时，找到较小的元素，哑结点.next指向较小的元素。
- 若有一个链表未被遍历完毕，则将cur的next指向未被遍历完的链表即可。【指针的特点】

```java
public ListNode Merge(ListNode list1, ListNode list2) {
	ListNode head = new ListNode(-1);//哑结点
	ListNode cur = head;
	while(list1!=null && list2!=null) {
		if(list1.val < list2.val) {
			cur.next = list1;
            cur = cur.next;
			list1 = list1.next;
		}else {
			cur.next = list2;
            cur = cur.next;
			list2 = list2.next;
		}
	}
	if(list1!=null) cur.next = list1;
	if(list2!=null) cur.next = list2;
	return head.next;
}
```

> **输入两个链表，找出它们的第一个公共结点。**

- 法一：利用两个栈。先全部入栈。然后出栈，两个栈中第一个不相等的元素的前一个出栈元素就是第一个公共结点。
- 法二：先获得两个链表的长度，让长的链表先走一定的距离直到他和短的链表一样长。

```java
public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
	Stack<ListNode> stack1 = new Stack<ListNode>();
	Stack<ListNode> stack2 = new Stack<ListNode>();
	while(pHead1!=null) {
		stack1.add(pHead1);
		pHead1 = pHead1.next;
	}
	while(pHead2!=null) {
		stack2.add(pHead2);
		pHead2 = pHead2.next;
	}
	ListNode pre = null;
	while(!stack1.empty()&&!stack2.empty()) {
		ListNode peek = stack1.peek();
		//找到第一个公共结点 即栈中的最后一个公共结点
		if(stack1.pop().val==stack2.pop().val) pre=peek;
	}
	return pre;
}
```

> **在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。链表1->2->3->3->4->4->5处理后为 1->2->5**

- 三指针法。pre cur next
- cur和next值一样时，说明遇到重复的元素了。让next一直走直到cur和next值不一样
- 然后pre.next = next; cur = next; next = next.next;
- 注意判断next为空的情况
- 定义哑结点，方便操作。pre开始指向哑结点
- 注意边界问题

```java
public ListNode deleteDuplication(ListNode pHead) {
	if (pHead == null || pHead.next == null) return pHead;
	ListNode ya = new ListNode(-1); ya.next = pHead;
	ListNode pre = ya; ListNode cur = pHead;
	ListNode rear = pHead.next;
	while (rear != null) {
		if (cur.val != rear.val) {
			pre = pre.next;
			cur = cur.next;
			rear = rear.next;
		} else {
			// 出现相等的点
			while (rear != null) {
				if (rear.val == cur.val) rear = rear.next;
				else break;
			}
			// 循环结束后找到了一个和pre不等的数
			if (rear == null)
				pre.next = null;
			else {
				pre.next = rear; cur = rear;
				rear = rear.next;
			}
		}
	}
	return ya.next;
}
```

> 后面会对题目进行补充，同时推出一份剑指offer的刷题笔记。解题思路默认不给暴力解答。