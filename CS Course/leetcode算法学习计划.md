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
```



# 20天算法刷题计划

