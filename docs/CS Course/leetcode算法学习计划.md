# 3周攻克数据结构

## 数组

### 存在重复元素

给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，返回 true ；如果数组中每个元素互不相同，返回 false 。

示例 

```shell
输入：nums = [1,2,3,1]
输出：true

输入：nums = [1,2,3,4]
输出：false

输入：nums = [1,1,1,3,3,4,3,2,4,2]
输出：true

```

$1 <= nums.length <= 10^5$
$-10^9 <= nums[i] <= 10^9$

> 解题思路

- set 判重
- for循环双重遍历

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        // Set 判重
        HashSet<Integer> set = new HashSet<Integer>();
        for(int number : nums){
            boolean result = set.add(number);
            if(!result) return !result;
        }
        return false;
    }
}

// Stream 炫技。时间复杂度高
class Solution {
    public boolean containsDuplicate(int[] nums) {
        return IntStream.of(nums).distinct().count() != nums.length;
    }
}
```

### 最大子数组和

给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组 是数组中的一个连续部分。

 ```shell
 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
 输出：6
 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
 
 输入：nums = [1]
 输出：1
 
 输入：nums = [5,4,-1,7,8]
 输出：23
 ```

$1 <= nums.length <= 10^5$
$-10^4 <= nums[i] <= 10^4$

> 解题思路

- 动态规划题

![image-20220223231618694](img\image-20220223231618694.png)

```java
// dp 题
class Solution {
    public int maxSubArray(int[] nums) {
        /*
        f(1),f(2),f(3),f(4)...f(n) 中的 max。 f(i) 表示以 i 结尾的最大子数组和。
        状态转移方程是：f(i) = Max(f(i),f(i)+preMax)
         */
         int preMax = 0,ans = nums[0];
         for(int i: nums){
             preMax = Math.max(i,preMax+i);
             ans = Math.max(ans,preMax);
         }
         return ans;
    }
}
```

### 螺旋矩阵

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

 <img src="https://assets.leetcode.com/uploads/2020/11/13/spiral1.jpg">

```shell
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

<img src="https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg">

```shell
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

```shell
m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100
```

> 解题思路

- 实际上是一个找规律的题目。

### 两数之和

```java
public int[] twoSum3(int[] nums, int target) {
    HashMap<Integer, Integer> map = new HashMap<>();
    // 找是否存在 a+b=target ==> 就是找是否存在 target-b 的值。用 map去缓存值即可。
    // 由于一个数字只能用一次，所以不能先把所有数据都放到map里去。只能放遍历过的。
    // 为什么可以只放遍历过的？因为 a + b = c。开始没把 a 存入map，没发现a。但是 b + a = c
    // 我们把 b 存入了，下次可以找a
    for(int i=0;i<nums.length;i++){
        int find = target-nums[i];
        int index = map.getOrDefault(find,-1);
        if(index!=-1) return new int[]{index,i};
        map.put(nums[i],i);
    }
    return new int[]{};
}
```



# 20天算法刷题计划

