# 每日一题

## 3月29日

[题目来源](https://leetcode-cn.com/problems/reverse-bits/)

```shell
输入: 00000010100101000001111010011100
输出: 00111001011110000010100101000000
解释: 输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
     因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
```

最佳的是分治法，但是不好背，不记。

---

- 如何找到n的第k位？把n右移k位，把n的第k位移到个位上。 n>>k

- eg：n=11101

    ​		右移两位n>>2 = 111

    ​		111 & 1 就可以得到最后一位数字是多少了。&是相同为1，不同为0

    ​		n>>k&1 右移运算符优先级大于&，所以不用加括号

- 二进制==>十进制

> 题解：

```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int ret = 0;
        for(int i=0;i<32;i++){
            // n>>i & 1 找到 n中第i位的数是什么。
            // 因为是要反转，所以最低位变成了最高位。ret<<1把当前累计数一直左移动（放大），最后最低位的数会变成最高位的数
            // 原数要先移位 再加新的进位。
            ret = (ret<<1)+(n>>i &1);
        }
        return ret;
    }
}
```

计算方式：

11101	ret = 0

tmp = 11101 >>0 & 1 =  1 （个位）

ret = (ret<<1) + tmp 

---

翻转单词顺序

[题目来源](https://www.acwing.com/problem/content/73/)

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

 

示例 1：

输入: "the sky is blue"
输出: "blue is sky the"

----

- 先翻转字符串一次
- 在将单词逐个翻转

```java
public class ReverseWords {

// 
// leetcode做了加强，字符之间会有多个空格，需要先进行多余空格的去除
    public static void main(String[] args) {
        String liujiawei = reverseWords("  liujiawei hello  ");
        System.out.println(liujiawei);
    }

    public static String reverseWords(String s) {
        char[] chars = s.toCharArray();
        reverse(chars, 0, chars.length);
        int start = 0;
        for (int i = 0; i <= chars.length; i++) {
            if (i == chars.length || chars[i] == ' ') {
                reverse(chars, start, i);
                start = i + 1;
            }
        }
        return new String(chars);
    }

    public static void reverse(char[] chars, int start, int length) {
        for (int i = start, j = length - 1; i <= j; i++, j--) {
            char tmp = chars[i];
            chars[i] = chars[j];
            chars[j] = tmp;
        }
    }
}
```

## 3月28日

[反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

o -> o -> o -> o -> o -> o

```java
public class Offer_24_reverseList {
    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode pre = head;
        ListNode nexts = head.next;
        // 双指针做法，改变nexts指针的指向，让他指向前一个点。
        while (nexts != null) {
            // 保存next后面的点，防止丢失。
            ListNode c = nexts.next;
            nexts.next = pre;
            pre = nexts;
            nexts = c;
        }
        head.next = null;
        return pre;
    }
}
```

## 3月18日

[翻转链表2](https://leetcode-cn.com/problems/reverse-linked-list-ii/)