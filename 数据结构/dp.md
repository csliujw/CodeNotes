# DP

## 阎式DP分析法

DP一般用于求某个集合的最大值、最小值、总个数



寻找某个状态，用状态表示一类东西。

- 状态表示（每个状态表示一类东西）
    - 集合是什么，即这个状态表示那一类东西。
    - 属性（状态表示这个集合的那种值、最大、最小、数量）

- 状态计算（集合的划分）

> **举例** `LeetCode` `53 Maximum Subarray`

-2  1  -3  4  -1  2  1  -5  4

DP

- 状态表示 f[i]
    - 集合 所有以i结尾的字段
    - 属性 Max/Min/数量，`此题求的Max`
- 状态计算 -- 集合的划分 `每个f[i]如何算出来`
    - `以i结尾的最大值 = f[i-1] + nums[i]  即以i-1结尾的最大值+nums[i]`
    - `f[i] = max(f[i-1],0) + nums[i]`

```java
package leetcode;

public class _53_maxSubArray {
    public static void main(String[] args) {
        _53_maxSubArray maxSubArray = new _53_maxSubArray();
        int[] array = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
        System.out.println(maxSubArray.max(array));
    }

    public int max(int[] nums) {
        int res = Integer.MIN_VALUE;
        int last = 0;
        for (int i = 0; i < nums.length; i++) {
            int now = Math.max(last, 0) + nums[i];
            res = Math.max(res, now);
            last = now;
        }
        return res;
    }
}
```

> **举例** `LeetCode 120 Triangle`

给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

就是只能向下直走，或者向右侧走。

----

- 状态表示
    - 集合 所有从起点走到第i行，第j个数的路径
    - 属性 所有路径上的数的和的最小值
- 状态计算 -- 集合的划分 
    - `类Ⅰ：最后一步从左上下来 f[i-1,j-1] + nums[i,j]`
    - `类Ⅱ：最后一步从右上下来 f[i-1,j] + nums[i,j]`
    - `两类最小值取一个min Math.min(类Ⅰ,类Ⅱ)`

直接DP，它的空间复杂度是$O(n^2)$，但是我们可以发现，第i层，只用到了i-1层的信息，因此可以用滚动数组来做。

```java
public int minimumTotal(List<List<Integer>> triangle) {
    int n = triangle.size();
    int[][] dp = new int[n][n];
    dp[0][0] = triangle.get(0).get(0);

    for (int i = 1; i < n; i++) {
        // <=i的原因好像是避免一个问题。当前点上方的dp[i-1][j] == 0的情况，
        for (int j = 0; j <= i; j++) {
            int nums = triangle.get(i).get(j);
            dp[i][j] = 0xfffffff;
            if (j > 0) dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - 1] + nums);
            if (j < i) dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + nums);
        }
    }
    int retVal = 0xfffffff;
    for (int i = 0; i < n; i++) {
        retVal = Math.min(retVal, dp[n - 1][i]);
    }
    return retVal;
}
```

> **举例** `LeetCode unique Paths Ⅱ`

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/robot_maze.png)




- 状态表示
    - 集合：`所有从起点走到（i，j）的路径`
    - 属性：`数目`
- 状态计算
    - `最后一步向下走：f[i-1，j]`
    - `最后一步向右走：f[i，j-1]`
    - 