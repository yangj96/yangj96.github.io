---
title: 剑指Offer题目总结
date: 2019-11-21 17:33:12
categories: Algorithm
---

##### 旋转数组的最小数字

二分，注意处理单调递增和存在多个相同元素的情况，例如1，0，1，1，1

    int findMin(vector<int>& a) {
        int n = a.size();
        if (n == 0) return -1;
        while (n > 1 && a[0] == a[n-1]) n--;
        if (a[0] < a[n-1]) 
            return a[0];
        int x = a[0];
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (a[mid] < x) r = mid;
            else l = mid + 1;
        }
        return a[l];
    }
二维数组查找

```
class Solution {
public:
    bool searchArray(vector<vector<int>> array, int target) {
        int n = array.size();
        if (!n) return false;
        int m = array[0].size();
        if (!m) return false;
        
        for (int i = 0, j = m - 1; i < n && j >= 0;) {
            int t = array[i][j];
            if (t == target) return true;
            if (t > target) j--;
            else i++;
        }
        return false;
    }
};
```



##### 找到数组中的重复数字

置换法

```
int duplicateInArray(vector<int>& nums) {
    int n = nums.size();
    for (int i = 0; i < n; i++) {
        if (nums[i] < 0 || nums[i] > n - 1) 
            return -1;
    }
    for (int i = 0; i < n; i++) {
        while(nums[i] != i) {
            if (nums[i] == nums[nums[i]])
                return nums[i];
            swap(nums[i], nums[nums[i]]);
        }
	  }
 	  return -1;
}
```

下标对应元素个数二分

统计根据下标划分的区间内的元素个数，根据有重复元素的区间元素个数大于区间长度的性质二分

```
int duplicateInArray(vector<int>& nums) {
    int l = 1, r = nums.size() - 1;
    while (l < r) {
        int mid = l + r >> 1; // 划分的区间：[l, mid], [mid + 1, r]
        int s = 0;
        for (auto x : nums) s += x >= l && x <= mid;
        if (s > mid - l + 1) r = mid;
        else l = mid + 1;
    }
    return r;
}
```

##### 寻找缺失的数字

数组有序，对下标进行二分

    // 二分，特殊情况：当所有数都满足nums[i] == i时，表示缺失的是 n
    int getMissingNumber(vector<int>& nums) {
        if (nums.empty()) return 0;
    
        int l = 0, r = nums.size() - 1;
        while (l < r)
        {
            int mid = l + r >> 1;
            if (nums[mid] != mid) r = mid;
            else l = mid + 1;
        }
    
        if (nums[r] == r) r ++ ;
        return r;
    }
数组无序，索引补位，将所有的索引和元素做异或运算，只剩落单元素

异或运算满足结合律和交换律

```
int getMissingNumber(vector<int> &nums) {
	int n = nums.size();
	int res = 0;
	res ^= n;
	for (int i = 0; i < n; i++) {
		res ^= i ^ nums[i];
	}
	return res;
}
```





##### 调整数组使奇数排在偶数前

不要求保证原始稳定顺序的话可以使用双指针

    void reOrderArray(vector<int> &a) {
         int l = 0, r = a.size() -1;
         while (l < r) {
             while (l < r && a[l] % 2 == 1) l++;
             while (l < r && a[r] % 2 == 0) r--;
             if (l < r) swap(a[l], a[r]);
         }
    }
##### 数组中出现次数超过一半的数字

hash计数

排序，输出位于len/2位置的元素

多数投票问题，Boyer-Moore Majority Vote Algorithm，时间复杂度为 O(N)

使用 cnt 来统计一个元素出现的次数，当遍历到的元素和统计元素相等时，令 cnt++，否则令 cnt--，当cnt为0时将统计元素置为当前元素并令cnt = 1。

如果前面查找了 i 个元素，且 cnt == 0，说明前 i 个元素没有 majority，或者有 majority，但是出现的次数少于 i / 2 ，因为如果多于 i / 2 的话 cnt 就一定不会为 0 。此时剩下的 n - i 个元素中，majority 的数目依然多于 (n - i) / 2，因此继续查找一定能找出 majority。

    int moreThanHalfNum_Solution(vector<int>& nums) {
        int cnt = 0, val = -1;
        for (auto x : nums) {
            if (!cnt) {
                cnt = 1;
                val = x;
            } else {
                if (val == x) cnt++;
                else cnt--;
            }
        }
        return val;
    }
##### 最长不含重复字符的子字符串

双指针

    int longestSubstringWithoutDuplication(string s) {
        unordered_map<char,int> hash;
        int res = 0;
        for (int i = 0, j = 0; j < s.size(); j ++ ) {
            if ( ++ hash[s[j]] > 1) {
                while (i < j) {
                    hash[s[i]]--;
                    i ++ ;
                    if (hash[s[j]] == 1) break;
                }
            }
            res = max(res, j - i + 1);
        }
        return res;
    }


##### 用两个栈实现队列

```
// push(x)，我们直接将x插入主栈中即可。
// pop()，此时我们需要弹出最先进入栈的元素，也就是栈底元素。我们可以先将所有元素从主栈中弹出，压入辅助栈中。/ // 则辅助栈的栈顶元素就是我们要弹出的元素，将其弹出即可。然后再将辅助栈中的元素全部弹出，压入主栈中。
```

##### 顺时针打印矩阵

```
vector<int> printMatrix(vector<vector<int> > matrix) {
    int n = matrix.size();
    vector<int> res;
    if (n == 0) return res;
    int m = matrix[0].size();
    int dx[4] = {0, 1, 0, -1};
    int dy[4] = {1, 0, -1, 0};
    int x = 0, y = 0, d = 0;
    vector<vector<bool>> st(n, vector<bool>(m, false));
    for (int i = 0; i < n * m; i++) {
        res.push_back(matrix[x][y]);
        st[x][y] = true;
        int a = x + dx[d], b = y + dy[d];
        if (a < 0 || a >= n || b < 0 || b >= m || st[a][b]) {
            d = (d + 1) % 4;
            a = x + dx[d], b = y + dy[d];
        }
        x = a;
        y = b;
    } 
    return res;
}
```

##### 矩阵中是否存在字符串路径

```
class Solution {
public:
    bool dfs(vector<vector<char>>& matrix, string &str, int x, int y, int u) {
        if (matrix[x][y] != str[u])
            return false;
        if (u == str.size() - 1)
            return true;
    
        matrix[x][y] = '.';
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && matrix[nx][ny] != '.') {
                if (dfs(matrix, str, nx, ny, u+1)) return true;
            }
        }
        matrix[x][y] = str[u];
        return false;
    }
    
    bool hasPath(vector<vector<char>>& matrix, string &str) {
        n = matrix.size(); if (n == 0) return false;
        m = matrix[0].size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (dfs(matrix, str, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }
    
private:
    int n, m;
    int dx[4] = {0, 0, 1, -1};
    int dy[4] = {1, -1, 0, 0};
};
```

##### 2sum

    vector<int> findNumbersWithSum(vector<int>& nums, int target) {
        unordered_map<int, int> hash;
        for (auto x : nums) {
            if (hash.count(target-x)) {
                return vector<int>{target - x, x};
            }
            hash[x] = 1;
        }
        return vector<int>();
    }



##### 斐波那契数列

滚动数组递推

矩阵快速幂



##### 丑数

    int getUglyNumber(int n) {
        vector<int> dp(n);  
        dp[0] = 1;
        int i2 = 0, i3 = 0, i5 = 0;
        for(int i = 1; i < n; i++) {
            int next2 = dp[i2] * 2, next3 = dp[i3] * 3, next5 = dp[i5] * 5;
            dp[i] = min(next2, min(next3, next5));
            if (next2 == dp[i]) i2 ++ ;
            if (next3 == dp[i]) i3 ++ ;
            if (next5 == dp[i]) i5 ++ ;
        }
        return dp[n-1];
    }

##### 剪绳子

给你一根长度为 n 绳子，请把绳子剪成 m 段（m、n 都是整数，2≤n≤58 并且 m≥2）。

每段的绳子的长度记为k[0]、k[1]、……、k[m]。k[0]k[1] … k[m] 可能的最大乘积是多少？



贪心+数学推导

    int maxProductAfterCutting(int length) {
        if (length == 2) return 1;
        int res = 1;
        while (length % 3 == 1) {
            res *= 4;
            length -= 4;
        }
    
        while (length % 3 == 2) {
            res *= 2;
            length -= 2;
        }
        while (length) {
            res *= 3;
            length -= 3;
        }
        return res;
    }
dp



##### 连续子数组的最大和

    int maxSubArray(vector<int>& nums) {
        int s = 0, res = -2e9;
        for (auto x : nums) {
            if (s < 0) s = x;
            else s += x;
            res = max(res, s);
        }
        return res;
    }
##### 数字序列中某一位的数字

    int digitAtIndex(int n) {
        long long i = 1, num = 9, base = 1;
        while (n > i * num) {
            n -= i * num;
            i ++;
            num *= 10;
            base *= 10;
        }
    
        int number = base + (n + i - 1) / i - 1;
        int r = n % i ? n % i : i;
        for (int j = 0; j < i - r; j ++ ) number /= 10;
        return number % 10;
    }

#####正则表示式匹配

状态表示：f\[i][j]表示p从j开始到结尾，是否能匹配s从i开始到结尾
状态转移：

如果p[j+1]不是`*`通配符：

p[j]是正常字符，`f[i][j] = s[i] == p[j] && f[i+1][j+1] `

p[j]是`.`，`f[i][j] = f[i+1][j+1] `
如果p[j+1]是星号通配符，`f[i][j] = f[i][j+2] || ((s[i] == p[j]  || p[j] == '.' ) && f[i+1][j])`

```
class Solution {
public:
    int n, m;
    string s, p;
    vector<vector<int>> f;
    bool isMatch(string _s, string _p) {
        s = _s, p = _p;
        n = s.size(), m = p.size();
        f = vector<vector<int>> (n+1, vector<int> (m+1, -1));
        return dp(0, 0);
    }
    
    bool dp (int i, int j) {
        if (f[i][j] != -1) return f[i][j];
        if (j == m) {
            return f[i][j] = i == n;
        }
        bool firstMatch = i < n && (s[i] == p[j] || p[j] == '.');
        if (j + 1 < m && p[j+1] == '*') {
            f[i][j] = dp(i, j+2) || (firstMatch && dp(i+1, j));
        } else {
            f[i][j] = firstMatch && dp(i+1, j+1);
        }
        return f[i][j];
    }
};
```

##### 不用加减乘除实现加法



##### 扑克顺子

    bool isContinuous( vector<int> nums) {
        sort(nums.begin(), nums.end());
        for (int i = 1; i < nums.size(); i ++ )
            if (nums[i] && nums[i] == nums[i - 1])
                return false;
        for (auto x : nums)
            if (x)
            {
                return nums.back() - x <= 4;
            }
    }
