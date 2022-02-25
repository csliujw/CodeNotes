# 剑指专项练习

## 整数

### 整数除法

给定两个整数 `a` 和 `b` ，求它们的除法的商 `a/b` ，要求不得使用乘号 `'*'`、除号 `'/'` 以及求余符号 `'%'` 。

注意：

整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2
假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−2^31, 2^31−1]。本题中，如果除法结果溢出，则返回 231 − 1

#### 解题思路

- 注意边界条件。int 的最小负数 / -1 会爆 int 的max。
- O(N)的解法，直接用减法的话，复杂度太高
- O(logN)的解法：
    - a 减去 $b*2^n$ 的整数倍数，得到部分的商。
    - $a  -  b*2^n$  的结果继续减去 $b*2^n$ 的整数倍数，得到部分的商。

#### 代码

```java
// 暴力解题 out of time。
public int divide(int a,int b){
    if(a == Integer.MIN_VALUE && b == -1) return Integer.MAX_VALUE;
    int neg = 2;
    if(a>0){
        a = -a;neg--;
    }
    if(b>0){
        b=-b;neg--;
    }
    int retVal = 0;
    while(a<=b){
        a-=b;
        retVal++;
    }
    return neg%2==0?retVal:-retVal;
}
```

```java
// longN的解法 
// 先考虑 a 是 b 的多少偶数倍（2^n）。然后在 a = a - b*2^n，继续考虑
class Solution {
    public int divide(int a,int b){
        if(a == Integer.MIN_VALUE && b == -1) return Integer.MAX_VALUE;
        int neg = 2;
        if(a>0){
            a = -a;neg--;
        }
        if(b>0){
            b=-b;neg--;
        }
        int retVal = divideCore(a,b);
        return neg % 2 ==0?retVal:-retVal;
    }

    public int divideCore(int a,int b){
        int retVal = 0; // 商的结果
        while(a<=b){
            int position = 1;
            int tmp = b;
            while(tmp>(Integer.MIN_VALUE>>1) &&a<=tmp+tmp){
                position = position<<1;
                tmp = tmp<<1;
            }
            a=a-tmp;
            retVal+=position;
        }
        return retVal;
    }
}
```

### 二进制加法

给定两个 01 字符串 `a` 和 `b` ，请计算它们的和，并以二进制字符串的形式输出。

输入为 **非空** 字符串且只包含数字 `1` 和 `0`。

#### 解题思路

- 按位相加记录进位
- 没加完的，后面for循环继续加。
- 累加的时候记得加上进位

#### 代码

```java
class Solution {
    public String addBinary(String a, String b) {
        // 字符串最后一位是低位。位数短的，高位补0
        // 从低位向高位加。记录进位。
        int add = 0;
        int aindex = a.length() - 1;
        int bindex = b.length() - 1;
        StringBuffer sb = new StringBuffer();
        while(aindex>=0 || bindex>=0){
            int n1 = aindex>=0?a.charAt(aindex--)-'0':0;
            int n2 = bindex>=0?b.charAt(bindex--)-'0':0;
            int tmp = n1+n2+add;
            add = tmp>=2?1:0;
            sb.append(tmp%2);
        }
        if(add==1) sb.append(add);
        return sb.reverse().toString();
    }
}
```

### 前n个数字二进制中1的个数

给定一个非负整数 `n` ，请计算 `0` 到 `n` 之间的每个数字的二进制表示中 1 的个数，并输出一个数组。

```shell
输入: n = 2
输出: [0,1,1]
解释: 
0 --> 0
1 --> 1
2 --> 10
```

#### 解题思路

- 法一：暴力循环，求出每个数字1的个数。
    - 如何求1的数量？
    - 1.数字的每位一次 & 1，统计。
    - 2.直接数1的个数。 n & (n-1)可以做到。
        - n= 0111；n-1 = 0110
        - n & (n-1) = 0111 & 0110 = 0110 直接把最后一位的1去掉了。
        - 如果最后 n == 0，则说明没有1了。 
- 法二：发现 n 比 n & (n-1) 多一个1！！ 

#### 代码

```java
// 暴力求解 KlogN
class Solution {
    public int[] countBits(int n) {
        int[]nums = new int[n+1];
        for(int i=1;i<=n;i++){
            int tmp = i;
            while(tmp>0){
                if((tmp & 1)==1){
                    nums[i]++;
                }
                tmp = tmp>>1; 
            }
        }
        return nums;
    }
}
```

```java
// O(N)
class Solution {
    public int[] countBits(int n) {
        int nums[]=new int[n+1];
        for(int i=1;i<=n;i++){
            nums[i] = nums[i&(i-1)]+1;
        }
        return nums;
    }
}
```

### 只出现一次的数字

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次 。**请你找出并返回那个只出现了一次的元素。

#### 解题思路

- 哈希表判重
- 根据数字的 bit 位来判断。
    - 统计所有数字每个bit位出现1的次数。
    - 如果，该bit位出现的次数是3的倍数，则说明只出现一次的元素该bit位是0，否则是1。

#### 解法

```java
// hash 暴力解题
class Solution {
    public int singleNumber(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                map.replace(nums[i], map.get(nums[i]) + 1);
            } else {
                map.put(nums[i], 1);77\\\7
                
            }
        }
        Set<Integer> integers = map.keySet();
        Iterator<Integer> iterator = integers.iterator();
        while (iterator.hasNext()) {
            Integer key = iterator.next();
            if (map.get(key) == 1) {
                return key;
            }
        }
        return -1;
    }
}
```

```java
// bit 位 统计，还原，怎么计算出原数字的不明白。
class Solution {
    public int singleNumber(int[] nums) {
        // 统计每个bit出现1的次数
        int res = 0;
        int[] cnt = new int[32];
        for(int num : nums){
            for(int i = 0; i < 32; i++){
                cnt[i] += (num >> i) & 1;
            }
        }
        for(int i = 31; i >= 0; i--){
            // 如果t=0 说明这位是0
            int t = cnt[i] % 3 == 0 ? 0 : 1;
            res = (res << 1) + t;
        }
        return res;
    }
}
```

### 单词长度的最大乘积

给定一个字符串数组 words，请计算当两个字符串 words[i] 和 words[j] 不包含相同字符时，它们长度的乘积的最大值。假设字符串中只包含英语的小写字母。如果没有不包含相同字符的一对字符串，返回 0。

#### 解题思路

- 都是双重for循环判断那几个字符串不同。
- 问题在于如何判断是否不同
    - 法一：map判重。
    - 法二：用位运算判重；字符串中存在a则第0个位置的bit设置为1，存在b则第1个位置的bit设置为1。最后判断两个字符串是否有相同的字符，做一次 & 运算就知道了

#### 代码

```java
class Solution {
    public  int maxProduct(String... words) {
        // for 循环找出两两不同的，然后找max 会超时
        int max = 0;
        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                // 判断 i j 是否包含相同的字符
                if (!judge(words[i], words[j])) {
                    max = max > words[i].length() * words[j].length() ? max : words[i].length() * words[j].length();
                }
            }
        }
        return max;
    }

    public static boolean judge(String a, String b) {
        boolean[]c = new boolean[26];
        for (int i = 0; i < a.length(); i++) {
            c[a.charAt(i)-'a']=true;
        }

        for (int i = 0; i < b.length(); i++) {
            if(c[b.charAt(i)-'a']){
                return true;
            }
        }
        return false;
    }
}
```

```java
// 位运算判重
class Solution {
    public  int maxProduct(String... words) {
        int[] flags = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            for (int j = 0; j < word.length(); j++) {
                flags[i]|=1<< (word.charAt(j)-'a'); // 比如一个字符串中可能有多个 e e e，我们想把e对应的1放到flags中，所以用|，因为有多个，所以不用+
            }
        }
        int max = 0;
        for (int i = 0; i <words.length ; i++) {
            for (int j = i+1; j <words.length ; j++) {
                if((flags[i] & flags[j])==0){
                    int curlen = words[i].length()*words[j].length();
                    max = Math.max(max,curlen);
                }
            }
        }
        return max;
    }
}
```

## 数组

### 排序数组中两个数字之和

给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加之和等于目标数 target 。

函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标 从 0 开始计数 ，所以答案数组应当满足 0 <= answer[0] < answer[1] < numbers.length 。

假设数组中存在且只存在一对符合条件的数字，同时一个数字不能使用两次。

#### 解题思路

- 双指针。

#### 代码

```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        for (int start = 0, end = numbers.length - 1; start < end; ) {
            if (numbers[start] + numbers[end] == target) return new int[]{start, end};
            if (numbers[start] + numbers[end] > target) {
                end--;
            } else {
                start++;
            }
        }
        return null;
    }
}
```

### ★数组中和为0的三个数

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a ，b ，c ，使得 a + b + c = 0 ？请找出所有和为 0 且不重复的三元组。

#### 解题思路

- 法一：固定一个数字，在其他数字中找是否存在 a = -(b+c) 的数
- 法二：先排序，在固定一个数，双指针查找其余两个数

#### 代码

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int len = nums.length;
        if(len < 3) return res;
        Arrays.sort(nums);
        // 三元组的下标 i, j , k
        for(int i = 0; i < len-2; i++){
            int a = nums[i];
            int j = i + 1, k = len - 1;
            // 固定 a ,等价于 b + c 两数之和为 -a
            while(j < k){
                int b = nums[j], c = nums[k];
                if(a + b + c > 0){
                    k--;
                }else if (a + b + c < 0){
                    j++;
                }else{
                    res.add(Arrays.asList(a, b, c));
                    // 保证不重复的三元组
                    while(j < k && nums[j] == b){
                        j++;
                    }
                }
            }
            // 保证不重复的三元组
            while(i < len - 2 && nums[i + 1] == a){
                i++;
            }
        }
        return res;
    }
}
```

### ★和大于等于 target 的最短子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

#### 解题思路

- 法一：双指针。判断指针区域内的数组和是否符合条件，符合则更新长度。时间复杂度较高。可以继续优化。
- 法二：对法一进行优化。记录先前

#### 代码

```java
class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int ans = Integer.MAX_VALUE;
        int[] sums = new int[n + 1]; 
        // 为了方便计算，令 size = n + 1 
        // sums[0] = 0 意味着前 0 个元素的前缀和为 0
        // sums[1] = A[0] 前 1 个元素的前缀和为 A[0]
        // 以此类推
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }
        for (int i = 1; i <= n; i++) {
            int target = s + sums[i - 1];
            int bound = Arrays.binarySearch(sums, target);
            if (bound < 0) {
                bound = -bound - 1;
            }
            if (bound <= n) {
                ans = Math.min(ans, bound - (i - 1));
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }
}
```

### ★乘积小于k的子数组

给定一个正整数数组 `nums`和整数 `k` ，请找出该数组内乘积小于 `k` 的连续的子数组的个数。

#### 解题思路

双指针的思路，求出双指针范围内数组的乘积。right-left+1

比如某次遍历符合题意的子数组为 ABCX，那么在该条件下符合条件的有X，CX，BCX，ABCX共四个（可以进行多个例子，发现个数符合right-left+1）
我们可能会有疑问：AB，BC也算，为什么不算进去？
记住一点我们是以最右边的X为必要条件，进行计算符合条件的子数组，否则会出现重复的！
比如在X为右侧边界时（ABCX），我们把BC算进去了，可是我们在C为最右侧时（ABC），BC已经出现过，我们重复加了BC这个子数组两次！
换言之，我们拆分子数组时，让num[right]存在，能避免重复计算。

#### 代码

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int ret = 0;
        int total = 1;
        int left = 0;
        for(int right=0;right<nums.length;right++){
            total*=nums[right];
            while(total>=k && left<=right){
                total/=nums[left++];
            }
            ret+= (right>=left?right-left+1:0);
        }
        return ret;
    }
}
```

### 和为k的子数组

给定一个整数数组和一个整数 `k` **，**请找到该数组中和为 `k` 的连续子数组的个数。

#### 解题思路

由于数字并非是正数数组，所以双指针的思路行不通。最简单的办法是 双重for，暴力求解。
比较高效的解法是：用hash表记录前面累加的和。查找前面有多少满足 sum-k=0 的。就是（？~k范围内）符合要求的子数组个数。

#### 代码

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        /**
        存在负数。所以不能用双指针。
        用hash表记录前面累加的和。
         */
        HashMap<Integer,Integer> map = new HashMap<>();
        int sum = 0;
        int countNumber=0;
        map.put(0, 1);
        for(int i=0;i<nums.length;i++){
            sum+=nums[i];
            countNumber+=map.getOrDefault(sum-k,0);
            map.put(sum,map.getOrDefault(sum,0)+1);
        }
        return countNumber;
    }
}
```

### 0和1个数相同的子数组

#### 解题思路

把0改成-1，解题思路就和上面的一样了。求和为0的子数组的个数。

#### 代码

```java
class Solution {
    public int findMaxLength(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int sum = 0;
        int ret = 0;
        int tmp = 0;
        for(int i=0;i<nums.length;i++){
            sum+=nums[i]==0?-1:1;
            // 存储上一次出现 sum-k 数字出现的最远的距离
            tmp = map.getOrDefault(-sum,i);
            if(!map.containsKey(-sum)){
                map.put(-sum,i);
            }
            ret = Math.max(ret,i-tmp);
        }
        return ret;
    }
}
```

### 左右两边子数组的和相等

给你一个整数数组 nums ，请计算数组的中心下标 。

数组中心下标是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。

如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。

如果数组有多个中心下标，应该返回最靠近左边的那一个。如果数组不存在中心下标，返回 -1 。

#### 解题思路

先求出数组的总和sum。然后在遍历一次数组，统计当前遍历过的数字的总和curSum，如果curSum-cur = sum - curSum，则找到了中心下标。 

#### 代码

```java
class Solution {
    public int pivotIndex(int[] nums) {
        /**
        先求出total
        在扫描数组，判断 cur 前面的数据和 和 cur 后面的数据和是不是一样
         */
         int total = 0;
         for(int i=0;i<nums.length;i++){
             total+=nums[i];
         }
         int sum = 0;
         for(int i=0;i<nums.length;i++){
             sum+=nums[i];
             if(total-sum == sum-nums[i]) return i;
         }
         return -1;
    }
}
```

### ★★二维子矩阵的和

不会

## 字符串

### 字符串中的变位词

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的某个变位词。

换句话说，第一个字符串的排列之一是第二个字符串的 **子串** 。

```shell
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
```

#### 解题思路

主要是要明白题目的意思。比如 s1="ab", 如果s2中的子串包含 “ab” 或 “ba”，那就是有变位词。

先统计 s1 中字符串的频率（大数组当hash表）。在用双指针遍历 s2 中的字符串，统计字符出现的次数。如果当前范围内，所有字符出现的频率都一样，则这个范围内的字符串是变位词。

#### 代码

```java
class Solution {
    /**
    很简单，就是判断 s1 是否是 s2 其中一个连续子串的全排列。
    存在特殊的边界条件嘛？
    - s1.length > s2.length
    两个map记录。mapA 记录 s1的词频，map2记录当前s2子串的词频。对比看词频是否一样。
     */

    public boolean checkInclusion(String s1, String s2) {
        if(s1.length()>s2.length()) return false;
        int[]map1 = new int[26];
        int[]map2 = new int[26];
        for(int i=0;i<s1.length();i++){
            map1[s1.charAt(i)-'a']++;
            map2[s2.charAt(i)-'a']++;
        }
        if(s1.length()==s2.length()) return judege(map1,map2);
        // 不符合要求，则继续查找
        for(int pre=0,next=s1.length()-1;next<s2.length()-1;){
            if(judege(map1,map2)) return true;
            map2[s2.charAt(pre++)-'a']--;
            map2[s2.charAt(++next)-'a']++;
        }
        return judege(map1,map2);
    }

    public boolean judege(int[]map1,int[]map2){
        for(int i=0;i<26;i++){
            if(map1[i]!=map2[i]) return false;
        }
        return true;
    }
}
```

### 字符串中的所有变位词

给定两个字符串 s 和 p，找到 s 中所有 p 的变位词的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

变位词指字母相同，但排列不同的字符串。

```shell
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的变位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的变位词。
```

#### 解题思路

和上一题类似，不过需要遍历完字符串。边界条件需要注意下。

#### 代码

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> list = new ArrayList<Integer>();
        if(p.length()>s.length()) return list;
        int[]map1 = new int[26];
        int[]map2 = new int[26];
        for(int i=0;i<p.length();i++){
            map1[p.charAt(i)-'a']++;
            map2[s.charAt(i)-'a']++;
        }
        if(s.length()==p.length()) {
            if(juedge(map1,map2)) list.add(0);
            return list;
        }
        if(juedge(map1,map2)) list.add(0);
        for(int pre=0,next=p.length()-1;next<s.length()-1;){
            map2[s.charAt(pre++)-'a']--;
            map2[s.charAt(++next)-'a']++;
            if(juedge(map1,map2)) list.add(pre);
        }
        return list;
    }
    private boolean juedge(int[]map1,int []map2){
        for(int i=0;i<26;i++){
            if(map1[i]!=map2[i]) return false;
        }
        return true;
    }
}
```

### 不含重复字符串的最长子字符串

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长连续子字符串** 的长度。

```shell
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子字符串是 "abc"，所以其长度为 3。
```

#### 解题思路

双指针，hash表判重

#### 代码

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // 双指针
        if(s.length()==0 || s.length()==1) return s.length();
        int[]map = new int[128];
        int ret = 0;
        int pre = 0,next=1;
        map[s.charAt(pre)]+=1;
        while(next<s.length()){
            map[s.charAt(next)]+=1;
            // 加完后，看是否重复
            if(map[s.charAt(next)]>1){
                while(map[s.charAt(next)]>1){
                    map[s.charAt(pre++)]-=1;
                }
            }
            ret = Math.max(ret,next-pre+1);
            next++;
        }
        return ret;
    }
}
```

### ★含有所有字符的最短字符串

直接放弃

### 有效回文

给定一个字符串 `s` ，验证 `s` 是否是 **回文串** ，只考虑字母和数字字符，可以忽略字母的大小写。

本题中，将空字符串定义为有效的 **回文串** 。

```shell
输入: s = "A man, a plan, a canal: Panama"
输出: true
解释："amanaplanacanalpanama" 是回文串
```

#### 解题思路

双指针，start，end 移动。唯一的问题是需要过滤掉非数组和字母的字符，考察 API `Character.isLetterOrDigit` 的使用。

#### 代码

```java
class Solution {
    public boolean isPalindrome(String s) {
        for (int start = 0, end = s.length() - 1; start < end; ) {
            int startChar = s.charAt(start);
            int endChar = s.charAt(end);
            if (Character.isLetterOrDigit(startChar) && Character.isLetterOrDigit(endChar)) {
                if(Character.toLowerCase(startChar)!=Character.toLowerCase((endChar))) return false;
                start++;end--;
            }
            while (!Character.isLetterOrDigit(s.charAt(start)) && start<end) start++;
            while (!Character.isLetterOrDigit(s.charAt(end))&&end>0) end--;
        }
        return true;
    }
}
```

### 最多删除一个字符得到回文

给定一个非空字符串 `s`，请判断如果 **最多** 从字符串中删除一个字符能否得到一个回文字符串。

```shell
输入: s = "abca"
输出: true
解释: 可以删除 "c" 字符 或者 "b" 字符

输入: s = "abc"
输出: false
```

#### 解题思路

也是用双指针去做。如果发现，start 和 end 两个字符不相等，则考虑是去除 start 的字符还是去除 end 的字符。两者都试一试，继续判断剩下还未判断的，只要有一个为 true，那就是true。

#### 代码

```java
class Solution {
    public boolean validPalindrome(String s) {
        /**
        也是双指针。
        start,end
        如果发现当前的start!=end 则考虑是删除start 还是 end。
         */
         int start=0,end=s.length()-1;
         while(start<end){
             if(s.charAt(start)!=s.charAt(end)) break;
             start++;end--;
         }
        return judeg(start+1,end,s) || judeg(start,end-1,s);
    }

    public boolean judeg(int start,int end,String s){
        while(start<end){
            if(s.charAt(start)!=s.charAt(end)) return false;
            start++;end--;
        }
        return true;
    }
}
```

### 回文字符串的个数

给定一个字符串 `s` ，请计算这个字符串中有多少个回文子字符串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

```shell
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"

输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

#### 解题思路

这个题的思路是：找到每个可能是回文串的中心点，发散开来找回文串。
如：以index=0为中心的找回文串，以index=1为中心的找回文串，以index=2为中心的找回文串...

#### 代码

```java
class Solution {
    public int countSubstrings(String s) {
        if(s.length()==0 || s.length()==1) return s.length();
        int count = 0;
        for(int i=0;i<s.length();i++){
            count += counter(i,i,s);
            count += counter(i,i+1,s);
        }
        return count;
    }

    private int counter(int start,int end, String s){
        int ret = 0;
        while(start>=0 && end<s.length() && s.charAt(start)==s.charAt(end)){
            start--;
            end++;
            ret++;
        }
        return ret;
    }

    //从字符串的第start位置向左，end位置向右，比较是否为回文并计数
    private int countPalindrome(String s, int start, int end) {
        int count = 0;
        while (start >= 0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
            count++;
            start--;
            end++;
        }
        return count;
    }
}
```

## 链表

### 删除链表的倒数第n个结点

给定一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点

![img](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```shell
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

#### 解题思路

设置一个哑节点（方便删除第一个node）+快慢指针。具体块指针多走几步，举个例子算下就知道了。

#### 代码

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    // 典型的双指针。
    public ListNode removeNthFromEnd(ListNode head, int n) {
        // 快慢指针，快指针比慢指针领先 n 步骤。
        // 特殊情况为删除头节点
        if(head.next==null && n==1) return null;
        ListNode dummy = new ListNode(-1,head);
        ListNode fast=dummy,slow=dummy;
        while(n>0){
            fast = fast.next;
            n--;
        }
        while(fast.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
```

### 链表中环的入口节点

给定一个链表，返回链表开始入环的第一个节点。 从链表的头节点开始沿着 next 指针进入环的第一个节点为环的入口节点。如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/07/circularlinkedlist.png)

```shell
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

#### 解题思路

- 直接上 hash 表判重。第一个遇到的重复的节点就是环的入口节点
- 快慢指针。发现有环后，快指针重新指向头节点，然后快慢指针每次都直走一步。快慢指针相遇的节点就是环的入口节点（这题刷过就会，不刷就不会）

#### 代码

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        HashMap<ListNode,Boolean> map = new HashMap<>();
        while(head!=null){
            if(!map.getOrDefault(head,false)) map.put(head,true);
            else return head;
            head = head.next;
        }
        return null;
    }
}
```



### 两个链表的第一个重合节点

给定两个单链表的头节点 `headA` 和 `headB` ，请找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。

#### 解题思路

- 法一：哈希表判重
- 法二：stack 存入，然后同时出栈对比是否是同一个节点。

#### 代码

```java
// 法一：哈希表判重
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        // 法一：hashmap。判断。
        if(headA == null || headB == null) return null;
        HashMap<ListNode,Object> map = new HashMap<>();
        while(headA!=null){
            map.put(headA,new Object());
            headA = headA.next;
        }
        while(headB!=null){
            if(map.containsKey(headB)) return headB;
            headB = headB.next;
        }
        return null;
    }
}
```

```java
// stack
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) return null;
        LinkedList<ListNode> s1 = new LinkedList<>();
        LinkedList<ListNode> s2 = new LinkedList<>();
        while(headA!=null){
            s1.push(headA);
            headA = headA.next;
        }
        while(headB!=null){
            s2.push(headB);
            headB = headB.next;
        }
        ListNode lastNode = null;
        while(!s1.isEmpty() && !s2.isEmpty()){
            ListNode t1 = s1.pop();
            ListNode t2 = s2.pop();
            if(t1==t2) lastNode = t1;
        }
        return lastNode;
    }
}
```

### 反转链表

给定单链表的头节点 `head` ，请反转链表，并返回反转后的链表的头节点。

#### 解题思路

- 哑节点，头插法，先插入的数据在后面。
- 用 stack。
- 双指针？

#### 代码

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        // 虚拟头节点。头插法
        if(head == null || head.next == null) return head;
        ListNode dummy = new ListNode(-1);
        while(head!=null){
            // 头插
            ListNode tmp = head;
            head = head.next;
            tmp.next = dummy.next;
            dummy.next = tmp;
        }
        return dummy.next;
    }
}
```

### 链表中的两数相加

给定两个 非空链表 l1和 l2 来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。

可以假设除了数字 0 之外，这两个数字都不会以零开头。

#### 解题思路

- 链表中的数据分别放入，两个 stack，然后出stack，计算，创建节点。

#### 代码

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        LinkedList<Integer> s1 = new LinkedList<>();
        LinkedList<Integer> s2 = new LinkedList<>();
        // 链表的数据入栈7 --> 2 --> 4 --> 3 依次入栈
        while(l1!=null){
            s1.push(l1.val); // 高位就在下面了
            l1 = l1.next;
        }
        while(l2!=null){
            s2.push(l2.val);
            l2 = l2.next;
        }
        int add = 0;
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while(!s1.isEmpty() || !s2.isEmpty()){
            int n1 = s1.isEmpty()?0:s1.pop();    
            int n2 = s2.isEmpty()?0:s2.pop();
            System.out.println(n1+":"+n2+":"+add);
            dummy.next = new ListNode((n1+n2+add)%10,dummy.next);
            add = (n1+n2+add)>=10?1:0;
        }
        if(add==1)dummy.next = new ListNode(1,dummy.next);
        return dummy.next;
    }
}
```

### 回文链表

给定一个链表的 **头节点** `head` **，**请判断其是否为回文链表。

如果一个链表是回文，那么链表节点序列从前往后看和从后往前看是相同的。

#### 解题思路

- 法一：数组存储数据，双指针判断
- 法二：扫一遍链表，找到中间位置，再把后面的链表反转，前半部分和后半部分分别比较，后半部分比较的时候，再将链表反转

#### 代码

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        int[]arr = new int[100000];
        int countsNode = -1;
        while(head!=null){
            arr[++countsNode] = head.val;
            head = head.next; 
        }
        for(int start=0,end = countsNode;start<=end;){
            if(arr[start++]!=arr[end--]) return false;
        }
        return true;
    }
}
```

### 展平多级链表

这题真的有难度。

多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给定位于列表第一级的头节点，请扁平化列表，即将这样的多级双向链表展平成普通的双向链表，使所有结点出现在单级双链表中

> 解题思路

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/multilevellinkedlist.png">

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/multilevellinkedlistflattened.png">

- 先搞清楚展平规则
- node.next = child; child.pre = node; child.next = node.next; node.next.pre = child.
- 搞清楚展平规则后就是想怎么做了。
- 可以看出，这是一个递归的结构。展平 3 的子链表。如果 3 的子链表有子链表，那就先展平 3 子链表的子链表。

<img src="https://assets.leetcode-cn.com/solution-static/jianzhi_II_028/1.png">

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node prev;
    public Node next;
    public Node child;
};
*/

class Solution {
    public Node flatten(Node head) {
        flattenGetTail(head);
        return head;
    }

    // 返回 尾部指针，用户展平
    private Node flattenGetTail(Node head){
        // 展平链表
        Node node = head;
        Node tail = null;
        while(node!=null){
            Node next = node.next;
            if(node.child!=null){ // 如果有子链表就展平
                Node child = node.child;
                Node childTail = flattenGetTail(child); // 递归展平子链表

                node.child = null; // child 需要置空奥

                node.next = child; // next 指向 child，child 的尾指针指向node 的next节点
                child.prev = node; 

                childTail.next = next; // 子链表的 next 指向 node 的 next。
                if(next!=null){ // 达成双链表
                    next.prev = childTail;
                }
                tail = childTail;
            }else{
                // 如果没有 child的话，尾部指针就是
                tail = node;
            }
            node = next;
        }
        return tail;
    }
}
```



# 老版剑指offer

## **二维数组中的查找**

- [x] 矩阵有序
- 基本思路就是给矩阵降低阶数
- 第一行最右边的那个元素A是此行最大的，是此列最小的
- 如果A>target说明这一列都大于target的，就可以把这一列都移除不进行比较
- 如果A<target说明这一行都是小于target的，就可以把这一行都移除不进行比较
- 移除的策略是限定二维数字行列的坐标
- 列的下标最后会趋于0
- 行的下标最后会趋于length-1
```java
public boolean Find(int target, int[][] array) {
	boolean find = false;
	if (array[0].length > 0 && array.length > 0) {
		int row = 0;
		int col = array.length - 1;
		while (row <= array[0].length-1 && col >= 0) {
			if (array[row][col] == target) {
				return true;
			} else if (array[row][col] > target) {
				col--;
			} else {
				row++;
			}
		}
	}
	return find;
}
```

## **替换空格**

- [x] 建立队列，元素全部入队再出队拼接字符串，空格用"%20"替代。
- [x] 先扫描数组的长度，统计空字符串的数目，确定好新数组的大小，在遍历替换存入数组
- [x] 以上方法都不优，未考虑内存消耗。
```java
public String replaceSpace(StringBuffer str) {
	String s = "";
	for(int i=0;i<str.length();i++) {
		if(str.charAt(i)==' ') {
			s+="%20";
		}else {
			s+=str.charAt(i);
		}
	}
	return s;
}
```

## 从头到尾打印链表

- [x] 用堆栈
- [x] 单链表逆置再输出
```java
public ArrayList<Integer> printList(ListNode listNode) {
	ArrayList<Integer> arrayList = new ArrayList<Integer>();
	//定义一个哑结点 方便操作
	ListNode head = new ListNode(-1);
	ListNode cur = listNode;
	while(cur!=null) {
		ListNode temp = cur.next;
		cur.next = head.next;
		head.next = cur;
		cur = temp;
	}
	head = head.next;
	while(head!=null) {
		arrayList.add(head.val);
		head = head.next;
	}
	return arrayList;
}
```
## 前序中序重建二叉树

- [x] 递归建立左右子树
- 先获取前序发现根元素，在遍历中序序列得到它的左右子树
- 找到了就对这个左右子树进行递归在确定左右子树
- 下面的代码挺好理解但是效率不行
```java
int index = 0;
public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
    return helper(pre, in, 0, in.length - 1);
}

TreeNode helper(int[] preorder, int[] inorder, int inStart, int inEnd){
    if(inStart > inEnd) return null;
    int val = preorder[index++];//获取根节点
    for(int i = inStart; i <= inEnd; i++){
    //在中序序列中找到左右子树
        if(inorder[i] == val){//如果找到根了
            TreeNode root = new TreeNode(val);//建立根节点
            root.left = helper(preorder, inorder, inStart, i - 1);//递归创建其左子树
            root.right = helper(preorder, inorder, i + 1, inEnd);//递归创建其右子树
            return root;//返回root给上一级做子树
        }
    }
    return null;
}
```
## 用两个栈实现队列

- [x] 题目思路
- stack1负责元素入栈。stack2负责元素出栈
- 1 2 3 4 5入栈 栈顶是5
- 然后输出队列元素。先把stack1的所有元素移动到到stack2中
- 5 4 3 2 1 栈顶是1
- 1 2 3 4 5入队 1 2 3 4 5出队 ok！
```java
Stack<Integer> stack1 = new Stack<Integer>();
Stack<Integer> stack2 = new Stack<Integer>();

public void push(int node) {
	stack1.push(node);
}

public int pop() {
	if (!stack2.isEmpty())
		return stack2.pop();
	else {
		while (!stack1.isEmpty())
			stack2.push(stack1.pop());
		return stack2.pop();
	}
}
```
## 旋转数组的最小数字

>题目描述：

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
- [x] 暴力解
- 旋转数组是从小到大的序列的。
- 遍历时若发现 array[pre]>array[next] 说明旋转数组的被旋转部分的起始找到了，就是最小的那个元素
- [x] 二分查找 left mid right
- 我们把数组分为A B两部分。A部分的值都大于B部分的值
- **我们采用二分查找，==当mid.val>left.val是left=mid+1==。**
    - 此时mid肯定是A部分的【因为mid.val>left.val】
    - 但是mid+1不一定，所以需要与right.val比较，如果恰好小于则说明要找的元素就是mid+1.val。【因为mid+1是第一个不满足条件的】
- **如果mid.val<left.val。**
    - 说明找到了B部分的元素，此时right=mid【不取mid-1是因为怕mid正好是要找的元素】
    - 然后继续在left和mid中查找。
- **==当left和right相邻时，计算中位数，mid还是指向left。==**
    - left.val不小于mid.val
    - right.val不大于mid.val
    - left++。left==right,元素找到。
- **对于若A B两部分出现了重复的元素，如 ==[1 1 1 0 1 1]==** 
    - left.val = 1 ; mid.val = 1 ; right.val = 1;
    - 我们无法进行二分查找，因为不知道mid是在A部分还是在B部分
    - 此时只能进行顺序查找，移动left【这样也和前面移动left统一起来了】
```java
//暴力解
public int minNumberInRotateArray(int[] array) {
	int len = array.length;
	if (len == 0)
		return 0;
	for(int i=0;i<len-1;i++) {
		if(array[i]>array[i+1])
			return array[i+1];
	}
	return 0;
}

//二分查找
public int minNumberInRotateArray(int[] array) {
	int len = array.length-1;
	int left = 0;
	int right = len;
	while(left<right) {
		if(array[left]<array[right])
			return array[left];
		int mid = left + (right-left)/2;
		if(array[left]<array[mid])
			left = mid+1;
		else if(array[mid]<array[right])
			right = mid;
		else
			left++;//遇到相同的数
	}
	return array[left];
}
//牛客上的测试结果 我的暴力解答速度更快。
```
## 斐波那契数列

- [x] 递归太慢
- [x] 循环计算 速度可以
```java
public int Fibonacci(int n) {
    if(n==0) return 0;
    if(n==1) return 1;
    int f1 = 0;
    int f2 = 1;
    int fn = 0;
    for(int i=2;i<=n;i++){
        fn = f1+f2;
        f1 = f2;
        f2 = fn;
    }
    return fn;
}

public int F(int n) {
	if(n==0) return 0;
	if(n==1) return 1;
	return F(n-1)+F(n-2);
}
```

## 跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
- [x] 这是一个斐波那契的变形题。
- 递归分析 f(n) = f(n-1) + f(n-2)
- 青蛙跳上第n个条件的方式有2中，第一种 从n-1跳到n，第二种，从n-2跳到n。
- 从上述表达式看出，这是一个斐波那契数列问题
```java
public int JumpFloor(int n) {
    if(n==1) return 1;
    if(n==2) return 2;
    int f1 = 1;
    int f2 = 2;
    int fn = 0;
    for(int i=3;i<=n;i++){
        fn = f1+f2;
        f1 = f2;
        f2 = fn;
    }
    return fn;
}
```
## 变态跳台阶

- [x] fn = f1+f2+f3+...fn
- 计算就行
- [x] 官方题解 数学归纳证明出了 2^(n-1)种跳法

>**矩形覆盖**
>用2x1的矩阵覆盖2xn的矩阵有多少种方法
- [x] 画图看看
- 发现还是斐波那契数列
```java
public int RectCover(int n) {
    if(n==1) return 1;//竖着放只有一种
    if(n==2) return 2;//横着放有两种
    int f1 = 1;
    int f2 = 2;
    int fn = 0;
    for(int i=3;i<=n;i++){
        fn = f1+f2;
        f1 = f2;
        f2 = fn;
    }
    return fn;
}
```

## 链表中倒数第k个结点

- [x] 双指针。
- 设置快慢指针。快指针比慢指针多走k个结点
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
## 反转链表

- [x] 头插法
```java
public ListNode ReverseList(ListNode head) {
	ListNode node = new ListNode(-1);
	ListNode cur = head;
	while(cur!=null) {
		ListNode temp = cur.next;//保存next结点 防止丢失
        //下面是插入
		cur.next = node.next;
		node.next = cur;
		cur = temp;
	}
	return node.next;
}
```

## 合并两个排序的链表

```java
public ListNode Merge(ListNode list1, ListNode list2) {
	ListNode head = new ListNode(-1);//定义一个哑结点方便操作
	ListNode cur = head;
	while(list1!=null&&list2!=null) {
		if(list1.val<list2.val) {
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

## 从上往下打印二叉树

- [x] 层序遍历
```java
public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
	ArrayList<Integer> arrayList = new ArrayList<Integer>();
	Queue<TreeNode> queue = new LinkedBlockingQueue<TreeNode>();
	if(root==null) return arrayList;
	queue.add(root);
	while (!queue.isEmpty()) {
		root = queue.poll();
		arrayList.add(root.val);
		if (root.left != null)
			queue.add(root.left);
		if(root.right!=null)
			queue.add(root.right);
	}
	return arrayList;
}
```

## 二叉搜索树的后序遍历序列

```java
public boolean VerifySquenceOfBST(int [] sequence) {
    return Verify(sequence,0,sequence.length);
}

public boolean Verify(int[] sequence, int start, int end) {
    if (sequence == null || end == 0)
        return false;
    int root = sequence[end - 1];
    int i = 0;
    for (; i < end - 1; i++) {
        if (sequence[i] > root) {
            break;
        }
    }
    int j = i;
    for (; j < end - 1; j++) {
        if (sequence[j] < root)
            return false;
    }
    boolean left = true;
    boolean right = true;
    if (i > 0) {
        left = Verify(sequence, 0, i);
    }
    if (i < end - 1)
        right = Verify(sequence, i, end - 1);
    return (left && right);
}
```

## 二叉树的下一个结点

题目描述
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
我是看不懂他到底想求什么结点的下一个结点。
也没有提输入输出。

最后还是看懂了。方法中给出的参数A，我们要求的就是A的下一个结点
下面是结构体

```java
public class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;

    TreeLinkNode(int val) {
        this.val = val;
    }
}
```
- [x] 解法一
- 先获得根结点。再求中序遍历。假如我们要求A的下一个结点
- 设置temp = null 标识有没有发现A
- 在中序遍历时，发现结点A，我们用temp=A
- 在出栈时发现temp!=null,则当前出栈的结点就是我们要的结点。
- [x] 解法二
实际上 最后两种情况是一样的。
```java
//采用解法一进行解题
public TreeLinkNode GetNext(TreeLinkNode pNode) {
	TreeLinkNode temp = null;
	int val = pNode.val;// 我们要找到中序遍历val的下一个点
	// 找到根结点 好进行中序遍历
	while (pNode.next != null) {
		pNode = pNode.next;
	}
	TreeLinkNode root = pNode;
	Stack<TreeLinkNode> stack = new Stack<TreeLinkNode>();
	while (root != null || !stack.isEmpty()) {
		while (root != null) {
			stack.push(root);
			root = root.left;
		}
		if (!stack.isEmpty()) {
			root = stack.pop();
			if (temp != null) {
				return root;
			}
			if (root.val == val)
				temp = root;
			root = root.right;
		}
	}
	return null;
}
//采用解法二进行解题
TreeLinkNode GetNext(TreeLinkNode node) {
	if (node == null)
		return null;
	if (node.right != null) { // 如果有右子树，则找右子树的最左节点
		node = node.right;
		while (node.left != null)
			node = node.left;
		return node;
	}
	while (node.next != null) { // 没右子树，则找第一个当前节点是父节点左孩子的节点
		if (node.next.left == node)
			return node.next;
		node = node.next;
	}
	return null; // 退到了根节点仍没找到，则返回null
}
```
>**PS 我想不到解法二 太强了！**

## 最小的k个数

- [x] 优先队列 == Max(O(KlongN),O(N)) 建堆时间最优是O(N)
- [x] 先快排 在输出 == NlongN + K
```java
public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
	//Java中默认是小顶堆
    ArrayList<Integer> list = new ArrayList<Integer>();
    if(k>input.length){
        return list;
    }
    PriorityQueue<Integer> queue = new PriorityQueue<Integer>();
    for(int i=0;i<input.length;i++) {
    	queue.add(input[i]);
    }
    
    while(k>0&&!queue.isEmpty()) {
    	list.add(queue.remove());
    	k--;
    }
    return list;
}
```

## 二进制中1的个数

> 位运算知识点

```cpp
class Solution {
public:
    int NumberOf1(int n) {
        unsigned int flag = 1;
        int count = 0;
        for( int i=0; i<32; i++){
            if ( n & flag ) count++;
            flag = flag << 1;
        }
        return count;
    }
};
```

```java
public int NumberOf1(int n) {
    int flag = 1;
    int count = 0;
    for (int i = 0; i < 32; i++) {
        // 与运算，不同就为0，flag只有需要判断的最高位才是1，所以只要结果不是0，那么当前需要判断的位的数值就是1
        if ((n & flag) != 0) {
            count++;
        }
        flag = flag << 1;
    }
    return count;
}
```

## 旋转数组的最小值

> 按他的最佳思路，时间更慢。应该是程序的局部性原理。

```java
public int minArray(int[] numbers) {
    int min = 0X7fffffff;
    for (int i = 0; i < numbers.length; i++) {
        min = numbers[i] > min ? min : numbers[i];
    }
    return min;
}
```

## 数值的整数次方

```java
public double myPow(double x, int n) {
    if (n == 0) return 1;
    if (n == 1) return x;
    if (n == -1) return 1 / x;
    // 奇数就是 2次half * 多余的mod
    double half = myPow(x, n >> 1);
    double mod = myPow(x, n % 2);
    return half * half * mod;
}
```

