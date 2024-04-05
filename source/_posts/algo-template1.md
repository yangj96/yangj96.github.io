---
 
title: 算法模版-初级篇
date: 2019-12-31 12:33:54
categories: Algorithm
---

### 目录

1. 二分

2. 双指针

3. 单调队列

4. 滑动窗口

5. 哈希

6. 单调栈

7. 二叉树

8. 快速排序/快速选择

9. 堆/优先队列

10. 归并排序

11. 回溯/递归/DFS

12. BFS

13. 图

14. 并查集

15. 动态规划

16. 贪心

17. 数组trick

    ​    

#### 二分

二分本质不是单调性，只需要区间针对某个性质能够分成两段，一段满足一段不满足即可。

找到能够划分区间左右两半的性质，如果if (check(mid)) 条件成立，判断答案在左区间还是右区间，如果答案在**左区间并且mid也可能是答案**，按模板**1**来划分；如果答案在**右区间并且mid也可能是答案**，按模板**2**来划分（）。

模板1mid使用下中位数，模板2使用下中位数+1，终结条件为$low==high$，注意区间左右均为闭区间

**版本1**
最大值最小问题，第一个>=target的元素，满足check条件的**区间左边界**
区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用，其更新操作是r = mid或者l = mid + 1。计算mid时不需要加1。

```
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        // 两个int相加减会溢出 中间加个长整型常量
        int mid = l + 0ll + r >> 1;
        // 第一个大于等于key的数 if (a[mid] >= key)
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}
```

**版本2**
最小值最大问题，最后一个<= target的元素，找满足check条件的**区间右边界**
区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用，其更新操作是r = mid - 1或者l = mid。
因为r更新为mid-1，如果mid仍然计算下取整，则l和r差1时大者永远取不到，会死循环，因此计算mid时需要加1。

```
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + 1ll + r >> 1;
        // 最后一个小于等于key的数 if (a[mid] <= key)
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

**浮点数二分**

注意while判断条件考虑浮点误差应为`while (r - l > eps)`

```
double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```

**旋转数组二分**



#### 双指针

常见题型：

左右指针：三数之和、[盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

快慢指针：利用某种单调性确保 i，j两指针保持相同的移动方向

原地移除数据元素、求满足条件的子串

```
for (int i = 0, j = 0; i < n; i++) {
	// j在左，i在右
	while(j < i && check(i, j)) {
		// 具体逻辑
		j++;
	}
	// 具体逻辑
}
```

##### 最长无重复字符子串

注意check的条件是`cnt[a[i]]`意味着只需要检查新加入的最右端元素的出现次数

```
const int N = 100010;
int a[N], cnt[N];

int res = 0;
for (int i = 0, j = 0; i < n; i++) {
	cnt[a[i]]++;
	while(j < i && cnt[a[i]] > 1) {
		cnt[a[j]]--;
		j++;
	}
	res = max(res, i - j + 1);
}
```

#### 单调队列

```
int hh = 0, tt = -1;
for (int i = 0; i < n; i++) {
	// 队头滑出
	while(hh <= tt && check_out(hh, i)) hh++;
	// 队尾保持单调性
	while(hh <= tt && check(tt, i)) tt--;
	q[++tt] = i; //注意区分队列中存放下标还是元素值
}
```

##### 滑动窗口的最小值

```
int hh = 0, tt = -1;
for (int i = 0; i < n; i++) {
    // 下标间隔判断不是队列本身，而是队头和当前元素i的下标距离
    while(hh <= tt && i - q[hh] + 1 > k) hh++;
    while(hh <= tt && a[q[tt]] >= a[i]) tt--;
    q[ ++ tt] = i;
    if (i >= k - 1)
    	cout << a[q[hh]] << " ";
}
cout << endl;
```

#### 滑动窗口

```
int left = 0, right = 0;

while (left < right && right < nums.size()) {
    // 增大窗口
    window.add(nums[right]);
    right++;
    
    while (window needs shrink) {
        // 缩小窗口
        window.remove(nums[left]);
        left++;
    }
}
```

##### 最小覆盖子串

```C++
string minWindow(string s, string t) {
    unordered_map<char, int> hash;
    for (auto &c : t) {
        hash[c]++;
    }
    int cnt = hash.size();
    int l = 0, r = 0, valid = 0;
    string res;
    int minLen = 1e9;
    while (r < s.size()) {
        if (hash[s[r]] == 1) valid++;
        hash[s[r]]--;
        // 窗口满足覆盖条件时，尽可能缩小长度
        while (valid == cnt && hash[s[l]] < 0) {
            hash[s[l]]++;
            l++;
        }
        if (valid == cnt) {
            if (res.empty() || r - l + 1 < minLen) {
                res = s.substr(l, r - l + 1);
                minLen = r - l + 1;
            }
        }
        r++;
    }
    return res;
}
```

##### 找到字符串中所有给定子串的字母异位子串

```C++
vector<int> findAnagrams(string s, string p) {
    unordered_map<char, int> umap;
    int cnt = 0;
    for (auto &c : p) {
        if (!umap[c]) cnt++;
        umap[c]++;
    }
    int valid = 0;
    vector<int> res;
    for (int i = 0, j = 0; i < s.size(); i++) {
        umap[s[i]]--;
        if (umap[s[i]] == 0) valid++;
        // 长度为k的子数组/子串，可直接根据窗口长度判断是否需要缩小窗口
        while (i - j + 1 > p.size()) {
            if (umap[s[j]] == 0) valid--;
            umap[s[j]]++;
            j++;
        }
        if (valid == cnt) res.push_back(j);
    }
    return res;
}
```

##### 和为k的子数组

注意这道题如果数组存在负数，则右指针右移，左指针不满足右移的单调性，不能使用滑动窗口。应使用前缀和+哈希：

```
    int subarraySum(vector<int>& a, int k) {
        int n = a.size();
        vector<int> sum(n+1, 0);
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i-1] + a[i-1];
        }
        unordered_map<int, int> hash;
        int res = 0;
        hash[0] = 1;
        for (int i = 1; i <= n; i++) {
            res+= hash[sum[i] - k];
            hash[sum[i]]++;
        }
        return res;
    }
```

#### 哈希

常见题目：两数之和、和为k的子数组、字母异位词分组、最长连续序列

**最长连续序列**

```
int longestConsecutive(vector<int>& nums) {
    int res = 0;
    // unordered_map<int, int> tr_left, tr_right;
    // for (auto& x: nums) {
    //     int left = tr_right[x - 1];
    //     int right = tr_left[x + 1];
    //     tr_left[x - left] = max(tr_left[x - left], left + 1 + right);
    //     tr_right[x + right] = max(tr_right[x + right], left + 1 + right);
    //     res = max(res, left + 1 + right);
    // }
    unordered_set<int> set(nums.begin(), nums.end());
    for (auto& x: nums) {
        if (set.find(x-1) != set.end()) continue;
        int cur = 1, curVal = x;
        while (set.find(curVal+1) != set.end()) {
            cur++;
            curVal++;
        }
        res = max(res, cur);
    }
    return res;
}
```





#### 单调栈

单调性：元素下标i < j 但元素值 a[i] > a[j]时，a[j]必定有更长的生命周期，a[i]可被删除，因此最终栈内元素始终单调递增

##### 求数组每个元素左边第一个比它小/大的元素

```
int hh = 0;
for (int i = 0; i < n; i++) {
	while (hh > 0 && stk[hh] >= a[i]) hh--;
	stk[++hh] = a[i]; // 注意区分栈中存放下标还是元素值
}
```

##### 接雨水

```
int trap(vector<int>& a) {
    int n = a.size();
    stack<int> st;
    int res = 0;
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[i] >= a[st.top()]) {
            int t = st.top();
            st.pop();
            if (st.empty()) break;
            res += (min(a[i], a[st.top()]) - a[t]) * (i - st.top() - 1);
        } 
        st.push(i);
    }
    return res;
}
```

##### 柱状图最大矩形面积

```
def largestRectangleArea(self, heights) -> int:
    stack = []
    heights = [0] + heights + [0]
    n = len(heights)
    res = 0
    for i in range(n):
        # print(stack)
        while stack and heights[stack[-1]] > heights[i]:
            cur = stack.pop()
            res = max(res, (i - stack[-1] - 1) * heights[cur])
        stack.append(i)
    return res
```



#### 二叉树

数据结构的遍历形式主要分为迭代和递归，二叉树由于其非线性，只能采用递归遍历形式：

```cpp
void traverse(TreeNode* root) {
    // 前序位置
    traverse(root->left);
    // 中序位置
    traverse(root->right);
    // 后序位置
}
```

二叉树的解题思路分为两种：一是递归遍历二叉树，二是分解子问题。回溯通常定义一个返回值为空的函数做递归遍历，分治定义的函数返回子树的计算结果。以前序遍历为例：

```
// 回溯思想
vector<int> preorder(TreeNode* root) {
    vector<int> res;
    traverse(root, res);
    return res;
}
void traverse(TreeNode* root, vector<int>& res) {
    if (root == NULL) {
        return;
    }
    res.push_back(root->val);
    traverse(root->left);
    traverse(root->right);
}

// 分治思想
vector<int> preorder(TreeNode* root) {
    vector<int> res;
    if (root == NULL) {
        return res;
    }
    res.push_back(root->val);
    vector<int> left = preorder(root->left);
    vector<int> right = preorder(root->right);
    // 后序位置可以通过函数返回值获取子树传递回来的数据
    res.insert(res.end(), left.begin(), left.end());
    res.insert(res.end(), right.begin(), right.end());
    return res;
}
```

二叉树的层序遍历：本质是BFS

```
vector<vector<int>> levelTraverse(TreeNode* root) {
		vector<vector<int>> res;
    if (root == NULL) return res;
    queue<TreeNode*> que;
    que.push(root);
    
    while (!que.empty()) {
    		int sz = que.size();
    		vector<int> tmp;
        for (int i = 0; i < sz; i++) {
            TreeNode* cur = que.front();
            tmp.push_back(cur->val);
            que.pop();
            if (cur->left) que.push(cur->left);
            if (cur->right) que.push(cur->right);
        }
        res.push_back(tmp);
    }
    return res;
}
```



二叉搜索树：左子树均小于根节点，右子树均大于根节点

**二叉搜索树的最近公共祖先**

```c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    TreeNode* ancestor = root;
    while (true) {
        if (p->val < ancestor->val && q->val < ancestor->val) {
            ancestor = ancestor->left;
        }
        else if (p->val > ancestor->val && q->val > ancestor->val) {
            ancestor = ancestor->right;
        }
        else {
            break;
        }
    }
    return ancestor;
}
```



#### 快速排序/快速选择

快速排序 $O(nlgn)$

```
void qSort(int q[], int l, int r)
{
    if (l >= r) return;
	// x选择q[l]或下中位数，递归子区间选[l, j], [j + 1, r]
    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    qSort(q, l, j), qSort(q, j + 1, r);
}
```

快速选择$O(n)$

```
int qSelect(vector<int>& a, int l, int r, int k) {
    if (l == r) 
        return a[l];
    int x = a[l], i = l - 1, j = r + 1;
    while (i < j) {
        while(a[++ i] < x);
        while(a[-- j] > x);
        if (i < j) {
            swap(a[i], a[j]);
        }
    }
    int cnt = j - l + 1;
    if (cnt >= k)
        return qSelect(a, l, j, k);
    else
        return qSelect(a, j + 1, r, k - cnt);
}
```

#### 

#### 堆/优先队列

堆是满足任一父节点大于/小于其子节点的完全二叉树。



插入元素 右下插入

```
heap[++size] = x;
up(size);
```

删除堆顶 交换后删除右下元素

```
heap[1] = heap[size];
size--;
down(1);
```

删除任一元素

```
heap[k] = heap[size];
size--;
down(k);
up(k);
```

$O(1)$时间建堆，数列错位相减可证明

```
for (int i = n / 2; i; i--) {
	down(i);
}
```

通用操作

```C++
void down(int u) {
	int t = u;
	if (u * 2 <= size && h[u * 2] < h[t]) t = u * 2; 	
	if (u * 2 + 1 <= size && h[u * 2 + 1] < h[t]) t = u * 2 + 1;
	if (u != t) {
		swap(h[u], h[t]);
		down(t);
	}
}
void up(int u) {
	while (u / 2 && h[u] < h[u / 2]) {
		swap(h[u], h[u / 2]);
		u >>= 1;
	}
}
```

最大堆

```
priority_queue, greater > que;
```

最小堆

```
priority_queue que;
```



##### 前k大的数

快速选择算法 $O(n)$ + 排序$O(klgk)$

```
class Solution {
public:
    vector<int> qSelect(vector<int>& a, int l, int r, int k) {
        if (l == r) {
            vector<int> res(a.begin(), a.begin()+l+1);
            return res;
        }
        int x = a[l], i = l - 1, j = r + 1;
        while (i < j) {
            while(a[++ i] < x);
            while(a[-- j] > x);
            if (i < j) {
                swap(a[i], a[j]);
            }
        }
        int cnt = j - l + 1;
        if (cnt >= k)
            return qSelect(a, l, j, k);
        else
            return qSelect(a, j + 1, r, k - cnt);
    }
    
    vector<int> getLeastNumbers_Solution(vector<int> input, int k) {
        int n = input.size();
        if (!n) return {};
        vector<int> res = qSelect(input, 0, n - 1, k);
        sort(res.begin(), res.end());
        return res;
    }
};
```

最小堆（注意求前k大的数应该用最小堆）$O(nlgk)$

C++优先队列默认为最大堆，greater为最小堆

```
vector<int> getLastNumbers_Solution(vector<int> input, int k) {
        priority_queue<int, vector<int>, greater<int> > heap;
        for (int i = 0; i < input.size(); i++) {
            // heap.push(input[i]);
            // if (heap.size() > k) heap.pop();
            if (heap.size() < k) {
                heap.push(input[i]);
            } else if (input[i] > heap.top()) {
                heap.push(input[i]);
                heap.pop();
            }
        }
        vector<int> res;
        for (int i = 0; i < k; i++) {
            res.push_back(heap.top());
            heap.pop();
        }
        reverse(res.begin(), res.end());
        return res;
    }
```

##### 数据流的中位数

维护一个最大堆来存放较小一半的数和一个最小堆来存放较大一半的树，保证两个堆的数目保持一致或最多差一，如果两堆顶逆序交换即可。

```
priority_queue<int> maxHeap;
priority_queue<int, vector<int>, greater<int>> minHeap;
void insert(int num){
    maxHeap.push(num);
    if (minHeap.size() && minHeap.top() < maxHeap.top()) {
        auto p = minHeap.top(), q = maxHeap.top();
        minHeap.pop(); maxHeap.pop();
        minHeap.push(q);
        maxHeap.push(p);
    }
    if (maxHeap.size() - minHeap.size() > 1) {
        minHeap.push(maxHeap.top());
        maxHeap.pop();
    }
}

double getMedian(){
    if (maxHeap.size() + minHeap.size() & 1) return maxHeap.top();
    else return (maxHeap.top() + minHeap.top()) / 2.0; 
}
```

#### 归并排序

数组归并排序

注意合并时所需额外空间的处理 `	vector<int> tmp(r - l + 1);`

```
void mergeSort(vector<int>& a, int l, int r) {
	if (l >= r) return;
	int mid = l + r >> 1;
	mergeSort(a, l, mid);
	mergeSort(a, mid + 1, r);
	
	vector<int> tmp(r - l + 1);
	int k = 0, i = l, j = mid + 1;
	while (i <= mid && j <= r) {
		if (a[i] <= a[j]) tmp[k++] = a[i++];
		else tmp[k++] = a[j++];
	}
	while(i <= mid) tmp[k++] = a[i++];
	while(j <= r) tmp[k++] = a[j++];
	for (int i = l, j = 0; i <= r; i++, j++) a[i] = tmp[j];
}
```

##### 合并两个有序链表

```
struct ListNode {
	int val;
	ListNode* next;
};

ListNode* mergeList(ListNode* l1, ListNode* l2) {
	ListNode* dummy = new ListNode(-1);
	ListNode* cur = dummy;
	while(l1 && l2) {
		if (l1->val <= l2->val) {
			cur->next = l1;
			l1 = l1->next;
			cur = cur->next;
		} else {
			cur->next = l2;
			l2 = l2->next;
			cur = cur->next;
		}
	}
	while(l1) {
		cur->next = l1;
        l1 = l1->next;
        cur = cur->next;
	}
	while(l2) {
		cur->next = l2;
        l2 = l2->next;
        cur = cur->next;
	}
	return dummy->next;
}
```

##### 链表归并排序

快慢指针寻找链表中点

##### 逆序对的数量

分治：构成逆序对的两个数同在分治后的左侧区间或右侧区间，或者分别位于左右两个区间需要在归并时计算

归并计算逆序对：对右侧区间的每个数计算左侧区间中大于它的数的个数，最后全部求和

```C++
long long mergeSort(int l, int r) {
    if (l >= r) return 0;
    int mid = l + r >> 1;
    long long res = mergeSort(l, mid) + mergeSort(mid + 1, r);
    int k = 0, i = l, j = mid + 1;
  	vector<int> tmp(r - l + 1);
    while (i <= mid && j <= r) {
        if (a[i] <= a[j])
            tmp[k++] = a[i++];
        else {
            res += mid - i + 1;
            tmp[k++] = a[j++];
        }
    }
    while(i <= mid) 
        tmp[k++] = a[i++];
    while(j <= r) 
        tmp[k++] = a[j++];

    for (int i = l, j = 0; i <= r; i++, j++) 
        a[i] = tmp[j];

    return res;
}
```



##### 合并k个有序链表

	分治或最小堆

####



#### 回溯法

DFS回溯需要恢复状态主要是考虑每次枚举状态转移时当前起始点应保持一致，如果枚举导致其发生变化则需要恢复起始状态



回溯考虑的三要素：

可选择列表、当前路径、枚举顺序(可能需要标记枚举位置)

如果当前选择列表为空：

​	将当前路径加入答案集合

否则：

​	for choice in 选择列表：

​		将choice加入当前路径

​		将choice移出选择列表

​		递归调用

​		将choice移除当前路径

​		将choice重新加入选择列表



##### 指数枚举 $O(2^n)$

```
// 递归 状态压缩
// state记录当前路径，u从1到n枚举每个位置，每个位置的选择列表即选或不选不受变化影响
void dfs(int u, int state) {
    if (u == n) {
        for (int i = 0; i < n; i++) {
            if (state >> i & 1) {
                cout << i + 1 << " "; 
            }
        }
        cout << endl;
        return;
    }
    dfs(u + 1, state);
    dfs(u + 1, state | 1 << u);
}
// 递推 状态压缩
int n;
for (int state = 0; state < 1 << n; state++) {
	for (int j = 0; j < n; j++) {
		if (state >> j & 1) 
			cout << j + 1 << " ";
	}
	cout << endl;
}
```

##### 组合枚举 $O(C^k_n)$

枚举每个数是否被选中，增加选择k个数的限制条件，为避免组合型枚举重复枚举，人为指定顺序按顺序枚举

```
// 递归
int n, m;
vector<int> path;

void dfs(int u, int num) {
    if (num + n - u < m) {
        return;
    }
    if (num == m) {
        for(int i = 0; i < m; i++) {
            cout << path[i] << " ";
        }
        cout << endl;
        return;
    }    
    path.push_back(u+1);
    dfs(u+1, num+1);
    path.pop_back();
    dfs(u+1, num);
}
// 组合与顺序无关，使用state状态压缩代替path
// num记录已选择数目，u按顺序记录可选数的列表
void dfs(int u, int num, int state) {
    if (num + n - u < m) {
        return;
    }
    if (num == m) {
        for(int i = 0; i < n; i++) {
            if (state >> i & 1)
                cout << i + 1 << " ";
        }
        cout << endl;
        return;
    }   
    dfs(u + 1, num + 1, state | 1 << u);
    dfs(u + 1, num, state);
}
// 非递归
栈模拟
```

##### 排列枚举 $O(n!)$

```
int n;
vector<int> path;

// u从1到n枚举每个位置，visited记录可选列表
void dfs(int u, int visited) {
    if (u == n) {
        for (int i = 0; i < n; i++) {
            cout << path[i] << " ";
        }
        cout << endl;
        return;
    }
    for (int i = 0; i < n; i++) {
        if (!(visited >> i & 1)) {
            path.push_back(i + 1);
            dfs(u + 1, visited | 1 << i);
            path.pop_back();
        }
    }
}
```

```


class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;
    vector<vector<int>> permutation(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        path.resize(nums.size());
        dfs(nums, 0, 0, 0);
        return ans;
    }
    
    void dfs(vector<int>& nums, int u, int start, int state) {
        if (u == nums.size()) {
            ans.push_back(path);
            return;
        }
        
        if (!u || nums[u] != nums[u-1]) start = 0;
        for (int i = start; i < nums.size(); i++) {
            if (!(state >> i & 1)) {
                path[i] = nums[u];
                dfs(nums, u+1, i+1, state | 1 << i);
            }
        }
        
    }
};
```

[不同的枚举顺序](https://www.acwing.com/solution/AcWing/content/776/)



枚举每一个位置 i , 用state确定位置 i 是否用过，在每个位置上都尝试填数组第u个数



##### n皇后

u标记决策树每层表示棋盘的行数，每行枚举列的每个位置

```
const int N = 10;
int n;
char maze[N][N];
bool col[N], dg[N], udg[N];

void dfs(int u) {
    if (u == n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << maze[i][j];
            }
            cout << endl;
        }
        cout << endl;
    }    
    for (int i = 0; i < n; i++) {
        if (!col[i] && !dg[u + i] && !udg[u - i + n]) {
            col[i] = dg[u + i] = udg[u - i + n] = true;
            maze[u][i] = 'Q';
            dfs(u + 1);
            maze[u][i] = '.';
            col[i] = dg[u + i] = udg[u - i + n] = false;
        }
    } 
}
```





##### 带返回值的DFS/矩阵路线型

判断矩阵中是否存在某字符串路径

```
bool dfs(vector<vector<char>> &matrix, string &str, int u, int x, int y) {
    if (matrix[x][y] != str[u]) return false;
    if (u == str.size() - 1) return true;
    int dx[4] = {-1, 0, 1, 0}, dy[4] = {0, 1, 0, -1};
    char t = matrix[x][y];
    matrix[x][y] = '*';
    for (int i = 0; i < 4; i ++ ) {
        int a = x + dx[i], b = y + dy[i];
        if (a >= 0 && a < matrix.size() && b >= 0 && b < matrix[a].size()) {
            if (dfs(matrix, str, u + 1, a, b)) return true;
        }
    }
    matrix[x][y] = t;
    return false;
}

bool hasPath(vector<vector<char>>& matrix, string str) {
    for (int i = 0; i < matrix.size(); i ++ )
        for (int j = 0; j < matrix[i].size(); j ++ )
            if (dfs(matrix, str, 0, i, j))
                return true;
    return false;
}
```

寻找矩阵中价值最大的路径

```
int n, m;
int dfs(int x, int y, vector<vector<int>>& grid) {
    int dx[4] = {0, 0, 1, -1};
    int dy[4] = {1, -1, 0, 0};
    int tmp = grid[x][y]; 
    int res = tmp, ans = 0;
    grid[x][y] = 0;
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx >= 0 && nx < n && ny >= 0 && ny < m && grid[nx][ny]) {
            ans = max(ans, dfs(nx, ny, grid));
        }
    }
    grid[x][y] = tmp;
    res += ans;
    return res;
}

int getMaximumGold(vector<vector<int>>& grid) {
    n = grid.size();
    if (!n) return 0;
    m = grid[0].size();
    int res = 0;
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < m; j++) {
            if (grid[i][j]) {
                res = max(res, dfs(i, j, grid));
            }
        }

    return res;
}
```



#### 递归/DFS

空间复杂度 $O(最大递归深度)$

#### BFS

时间复杂度 $O(状态数*转移方式)$

空间复杂度 $O(状态数)$

##### 最短距离模型

###### 迷宫起点到终点的最少步数

```
const int INF = 1e8;
typedef pair<int, int> PII;
queue<PII> que;
char maze[N][M];
int d[N][M];
int dx[4] = {1, 0, -1, 0};
int dy[4] = {0, 1, 0, -1};

for (int i = 0; i < n; i ++) {
	for (int j = 0; j < m; j++) {
		d[i][j] = INF;
	}
}

int sx, sy, gx, gy;
que.push(PII(sx, sy));
d[sx][sy] = 0;

while(!que.empty()) {
    PII p = que.front();
    que.pop();
	if (p.first == gx && p.second == gy) break;
	
	for (int i = 0; i < 4; i++) {
		int nx = p.first + dx[i];
		int ny = p.second + dy[i];
		if (nx >= 0 && nx < n && ny >= 0 && ny < m && maze[nx][ny] != '#' && d[nx][ny] == INF) {
			que.push(PII(nx, ny));
			d[nx][ny] = d[p.first][p.second] + 1;
		}
	}
}
```



##### Flood Fill/连通域计数

DFS和BFS均可实现，可在线性时间找到某个点的连通块，但DFS数据较大可能会爆栈

湖泊计数

```
typedef pair<int, int> PII;
const int maxn = 100;

int n,m;
char field[maxn][maxn];
int dx[8]={0,1,1,1,0,-1,-1,-1};
int dy[8]={1,1,0,-1,-1,-1,0,1};

void dfs(int sx,int sy)
{
    field[sx][sy]='.';
    for (int i = 0; i < 8; i++) {
        int nx = sx + dx[i];
        int ny = sy + dy[i];
        if (nx >= 0 && nx < n && ny >= 0 && ny < m && field[nx][ny]=='W')
            dfs(nx, ny);//深度优先搜索无需在judge后标记节点，因为会递归调用；而bfs只调用一次
    }
}

void bfs(int sx,int sy)
{
    queue<P> que;
    que.push(make_pair(sx, sy));
    field[sx][sy]='.';
    
    while (!que.empty()) {
        PII p = que.front();
        que.pop();
        for (int i = 0; i < 8; i++) {
            int nx = p.first + dx[i];
            int ny = p.second + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && field[nx][ny]=='W')
            {
                que.push(make_pair(nx, ny));
                field[nx][ny] = '.';
            }
        }
    }
}
```

##### 最小步数模型

棋盘整体从一个状态变换为另一状态所需的最小步数，状态表示通常使用字符串，距离使用哈希

e.x 八数码 （[题目链接](https://www.acwing.com/problem/content/847/)）

```
int bfs(string start) {
	string end = "12345678x";
	
	queue<string> que;
	unordered_map<string, int> d;
	
	que.push(start);
	d[start] = 0;

	while(que.size()) {
		auto t = q.front();
		q.pop();
		
		int dist = d[t];
		if (t == end) return dist;
		
		int dx[4] = {-1, 0, 1, 0};
		int dy[4] = {0, 1, 0, -1};
	
		int k = t.find('x');
		int x = k / 3, y = k % 3;
		
		for (int i = 0; i < 4; i++) {
			int nx = x + dx[i];
			int ny = y + dy[i];
			if (nx >= 0 && nx < 3 && ny >= 0 && ny < 3) {
				swap(t[k], t[nx * 3 + b]);
				if (!d.count(t)) {
					d[t] = dist + 1;
					que.push(t);
				}
				swap(t[k], t[nx * 3 + b]); // 注意恢复状态
			}
		}
	}
	return -1;
}
```



e.x 魔板（[题目链接](https://www.acwing.com/problem/content/1109/)）

```
char g[2][4];
unordered_map<string, int> dist;
unordered_map<string, pair<char, string> > pre;

string get() {
    string res;
    for (int i = 0; i < 4; i++) res += g[0][i];
    for (int i = 3; i >= 0; i--) res += g[1][i];
    return res;
}

void set(string s) {
    for (int i = 0; i < 4; i++) g[0][i] = s[i];
    for (int i = 4, j = 3; i < 8; i++, j--) {
        g[1][j] = s[i];
    }
}

string move1(string s) {
    set(s);
    for (int i = 0; i < 4; i++) swap(g[0][i], g[1][i]);
    return get();
}

string move2(string s) {
    set(s);
    for (int i = 0; i < 2; i++) {
        char t = g[i][3];
        for (int j = 2; j >= 0; j--) 
            g[i][j+1] = g[i][j];
        g[i][0] = t;
    }
    return get();
}

string move3(string s) {
    set(s);
    char t = g[0][1];
    g[0][1] = g[1][1];
    g[1][1] = g[1][2];
    g[1][2] = g[0][2];
    g[0][2] = t;
    return get();
}


int bfs(string start, string end) {
    if (start == end) return 0;
    queue<string> que;
    que.push(start);
    dist[start] = 0;
    
    while(!que.empty()) {
        auto p = que.front();
        que.pop();
        
        string m[3];
        m[0] = move1(p);
        m[1] = move2(p);
        m[2] = move3(p);
        
        for(int i = 0; i < 3; i++) {
            if (!dist.count(m[i])) {
                dist[m[i]] = dist[p] + 1;
                pre[m[i]] = {'A' + i, p};
                que.push(m[i]);
                if (m[i] == end) return dist[m[i]];
            } 
        }
    }
    
    return -1;
}

int main() {
    string end;
    for (int i = 0; i < 8; i++) {
        int a;
        cin >> a;
        end += a + '0';
    }
    string start = "12345678";
    int cnt = bfs(start, end);
    cout << cnt << endl;
    if (cnt > 0) {
        string res;
        string s = end;
        while (s != start) {
            res += pre[s].first;
            s = pre[s].second;
        }
        reverse(res.begin(), res.end());
        cout << res << endl;
    }
  
    return 0;
}
```



#### 图

**图的表示：**

邻接矩阵
`int G [maxv][maxv] `或` <vector<vector<int> > G`

邻接表
`vector<int> G[maxv]`

邻接表边带权：

`struct edge {
  int to;
  int cost;
}`
`vector<edge> G[maxv]`

``` 
//读入
cin >> V >> E;
for (int i = 0; i < E; i++) {
	int s, t;
	cin >> s >> t;
	G[s].push_back(G[t]);
	//无向图
	G[t].push_back(G[s]);
}
```





#### 并查集

静态连通性问题使用BFS/DFS，动态连通性问题使用并查集

连通本质上是一种等价关系，满足自反性、对称性和传递性



按秩合并：增加树高rank数组，每次从rank小的树向rank大的树连边，避免退化

路径压缩：每次查询到根节点将该节点的parent直接连到根

对n个元素的并查集操作一次时间$O(α(n))$，$α$为阿克曼函数的反函数，比$O(lgn)$快。

```
void init() {
	vector<int> par(n);
	vector<int> rank(n, 0);
	for (int i = 0; i < n; i++) {
		par[i] = i;
	}
}

int find (int x, vector<int>& par) {
	return par[x] == x? x : par[x] = find(par[x], par);
}

void unite(int x, int y, vector<int>& par, vector<int>& rank) {
	x = find(x, par);
	y = find(y, par);
	if (x == y) 
		return;
	else if (rank[x] < rank[y]) {
		par[x] = y;
	} else {
		par[y] = x;
		// x为根且高度需加1
		if (rank[x] == rank[y]) 
			rank[x]++;
	}	
}
```

##### 边带权并查集

根节点绑定集合元素的大小

​	对于迷宫包围问题，可以利用虚拟节点营造出连通特性



每个元素绑定点到根节点的距离，适用于多分类的情况





##### 扩展域并查集

对于合法性问题利用并查集等价关系

POJ1182 食物链



#### 动态规划

1. 状态表示

f(i, j) 表示集合[i, j]的某一属性，例如集合中的最大值、最小值或数量

2. 状态计算

根据集合的划分计算

时间复杂度：状态数目 * 状态转移方式

空间复杂度：子问题的个数

##### 背包问题

N个物品，体积为V的背包，每类物品体积为$v_i$，价值权重为$w_i$，求满足体积限制的背包的最大价值

###### 01背包

每类物品只能用一次

状态f(i, j) 表示从前i类物品中选，所选物品体积小于j的所有选法的集合中 价值最大选法的价值

```
f[0][0-V] = 0
for (int i = 1; i <= n; i++) 
	for (int j = 0; j <= m; j++) {
		f[i][j] = f[i-1][j];
		if (j >= v[i]) f[i][j] = max(f[i][j], f[i - 1][j - v[i]] + w[i]);
	}

// 滚动数组优化，因为f中i只用到i-1且j只用到左侧j和j-v[i]，因此可用一维数组从大到小滚动优化
f[0][0-V] = 0
for (int i = 1; i <= n; i++) 
	for (int j = m; j >= v[i]; j--) {
		f[j] = max(f[j], f[j - v[i]] + w[i]);
	}
```

###### 完全背包

每类物品可以使用无限次，与01背包的区别主要在于集合的划分变为$f[i, j] = f[i-1, j-v[i]*k] + k*w[i]$

![image](https://tva1.sinaimg.cn/large/0082zybply1gc2qahne9fj31760a0agz.jpg)

因此完全背包的状态计算可以优化为$f[i, j] = max(f[i-1, j], f[i, j-v[i]] + w[i])$，优化后可以使用滚动数组进一步简化为一维，和01背包只有j的计算顺序不同



###### 多重背包

每类物品有$s_i$个，与完全背包状态划分计算相同，只不过k由$s[i]$约束.

多重背包的优化 

- 二进制拆分优化 

  由 $O(NS)$优化至$O(N\lg S)$

###### 分组背包

每组物品只能选一个，状态f(i, j)的划分根据第i组物品选第k个来拆分计算

```
// f[i][j] = max(f[i-1][j], f[i-1][j - v[i][k]] + w[i][k])k
```



##### 计数DP

方案数类初始化通常为f[0] = 1，因为空集也可以看作一种划分方案

###### 整数划分方案数

求1到n中任意个数之和为x的方案数

1. 转换为完全背包问题，状态f(i, j)表示为从1-i个数中选择（每个数可选无数次）使得和恰好为j的方案数

状态计算`f[i][j] = f[i-1][j] + f[i-1, j-i] + f[i-1][j-2*i] +... `

`f[i][j] = f[i - 1][j] + f[i, j - i]`

2. 状态f(i, j)表示所有总和为i恰好表示为j个数之和的方案数，状态计算根据j个数的最小值是否为1划分，对于最小值为1的情况，可以由去掉1的状态f(i - 1, j - 1)转移而来；对于最小值大于1的情况，可以由每个数减去1的状态f(i - j, j)转移而来

`f[i][j] = f[i-1, j-1] + f[i - j][j]`



##### 线性DP

递推顺序是线性序列

###### 数字三角形

状态f(i, j) 表示从起点走到(i, j)的所有路径的集合

注意 i 表示水平方向，j表示左下倾斜方向，初始化时需要注意`f[i][j+1]`右哨兵也会被用到

 ```
// f[i][j] = max(f[i-1][j-1], f[i-1][j]) + a[i][j]
 ```

###### 最长上升子序列

状态f(i) 表示以i结尾的所有上升子序列的集合

状态划分根据上一个数位置分类

```
f[i] = max(f[j] + 1), j = 0, 1, 2,...,i-1 && a[j] < a[i]
```

// TODO  优化

状态f(i)表示长度为i+1的上升子序列中末尾元素的最小值

由$O(n^2)$优化为$O(n\lg n)$



###### 最长公共子序列

f(i, j) 表示s1[1..i]和s2[1..j]的所有公共子序列

状态划分根据s1[i]和s2[j]是否包含在子序列中分为四类：

```
f[i, j] = max(f[i-1][j], f[i][j-1], f[i-1][j-1] + 1, f[i-1][j-1]);
```

###### 编辑距离

注意编辑距离的初始化

```
for (int i = 0; i <= p; i++) f[i][0] = i;
for (int j = 0; j <= q; j++) f[0][j] = j;
for (int i = 1; i <= p; i++) 
    for (int j = 1; j <= q; j++) {
        f[i][j] = min(f[i-1][j], f[i][j-1]) + 1;
        if (s1[i-1] == s2[j-1]) 
            f[i][j] = min(f[i][j], f[i-1][j-1]);
        else 
            f[i][j] = min(f[i][j], f[i-1][j-1] + 1);
    }
```



##### 区间DP

状态表示某区间，递推通常先循环区间长度，再循环区间左起点

###### 石子合并

状态f(i, j)表示将第 i 堆到第 j 堆合并的所有合并方式中代价的最小值，因此每个区间的状态初始化为正无穷

状态划分根据最后一次合并的分界线的位置分类

```
for (int len = 2; len <= n; len++) 
    for (int i = 1; i + len - 1 <= n; i++) {
        int l = i, r = i + len - 1;
        f[l][r] = 2e8;
        for (int k = l; k < r; k++) {
            int t = f[l][k] + f[k+1][r] + a[r] - a[l-1];
            f[l][r] = min(f[l][r], t);
        }
    }
```

###### 能量项链

###### 凸多边形的划分方案

状态划分：根据[L, R]边所属的三角形的另一个顶点位置来划分

```
for (int len = 3; len <= n + 1; len ++ )
        for (int l = 1; l + len - 1 <= n * 2; l ++ )
        {
            int r = l + len - 1;
            for (int k = l + 1; k < r; k ++ )
                f[l][r] = max(f[l][r], f[l][k] + f[k][r] + w[l] * w[k] * w[r]);
        }
```



##### 数位DP

数位DP通常用于解决两个整数a，b之间存在多少满足某个条件的数（且条件与数字每一位有关）的问题。
假设给定数x，包含n位，表示为$t_nt_{n-1}...t_1$，那么当我们求解n位数字$t_nt_{n-1}...t_1$的状态所对应的答案时就需重复计算n-1位数字$t_{n-1}t_{n-2}...t_1$的状态所对应的答案，因此具有重复子问题。
考虑DP状态为dp(idx, tight, sum)



###### 计数问题

给定两个整数 a 和 b，求 a 和 b 之间的所有数字中x的出现次数，x属于0到9

count(int n, int x) 假设一个数为abcdefg，对1 <= pppxqqq <= abcdefg分类讨论：

- 如果ppp = 000 到 abc-1:
  - 如果x不为0, qqq可以取000到999, cnt = abc * 1000
  - 如果x为0, qqq可以取000到999, 但由于x为0,ppp不能为0只能从001到abc-1, cnt = (abc-1)* 1000

- 如果ppp = abc :
  - d < x, cnt = 0
  - d = x, qqq可以取000到efg, cnt = efg + 1
  - d > x, qqq可以取000到999, cnt = 1000

```
int getNum(vector<int> &nums, int l, int r) {
    int res = 0;
    for (int i = l; i >= r; i--) {
        res = res * 10 + nums[i];
    }
    return res;
}

int power10(int x) {
    int res = 1;
    while (x--) {
        res *= 10;
    }
    return res;
}

int count (int n, int x) {
    if (!n) return 0;
    vector<int> nums;
    do {
        nums.push_back(n % 10);
        n /= 10;
    } while(n);
    n = nums.size();
    int res = 0;
    for (int i = n - 1 - !x; i >= 0; i--) {
        if (i < n - 1) {
            res += getNum(nums, n-1, i+1) * power10(i);
            if (!x) res -= power10(i);
        }
        if (nums[i] > x) res += power10(i);
        if (nums[i] == x) res += getNum(nums, i-1, 0) + 1;
    }
    return res;
}

int main() {
    int a, b;
    while (cin >> a >> b && (a || b)) {
        if (a > b) swap(a, b);
        for (int i = 0; i < 10; i++) {
            cout << count(b, i) - count(a-1, i) << " ";
        }
        cout << endl;
    }
}
```



##### 状态DP

状态DP的初始化通常将不合法状态的f值初始化为正无穷或负无穷

###### 不能打劫相邻位置的偷盗最大值

状态`f[i]`表示打劫第i家的最大值

`f[i] = max(f[i-1], f[i-2] + a[i])`



状态`f[i]`拆分为两个状态，`f[i][0]`表示打劫至第i家且不选当前位置，`f[i][1]`表示打劫至第i家且选当前位置，状态机的边表示从当前i转移到i+1的路径

```
const int N = 100001;
int a[N], f[N][2];
int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n; 
        for (int i = 0; i < n; i++) {
            cin >> a[i];
        }
        f[0][0] = 0;
        f[0][1] = a[0];
        for (int i = 1; i < n; i++) {
            f[i][0] = max(f[i-1][1], f[i-1][0]);
            f[i][1] = f[i-1][0] + a[i];
        }
        cout << max(f[n-1][0], f[n-1][1]) << endl;
    }
}
```



###### 股票买卖

只能买卖一次 记录最小值和最大差值

无限次买卖 贪心交易所有上涨交易



最多进行k次交易

手中持有股票状态为1，未持有股票状态为0

f[i, j, 0]表示前i天已经进行j次交易且当前无股票

f[i, j, 1]表示前i天正在进行j次交易且当前有股票

![image-20200317113704313](https://tva1.sinaimg.cn/large/00831rSTgy1gcwrtqmbcjj30g2078wgm.jpg)







含一天冷冻期

f[i, 0]表示前i天且当前有股票

f[i, 1]表示前i天且当前在冷冻期

f[i, 2]表示前i天且当前无股票且不在冷冻期

![image-20200317114017754](/Users/jingy/Library/Application Support/typora-user-images/image-20200317114017754.png)

`f[i][0] = max(f[i-1][0], f[i-1][2] - w[i])`

`f[i][1] = f[i-1][0] + w[i]`

`f[i][2] = max(f[i-1][1], f[i-1][2])`

##### 状态压缩DP

状态表示中的某一下标表示的是由状压state表示的集合

###### 集合类 - 最短Hamilton路径

状态f(i, j)表示从0走到j，走过的点的集合是i的二进制表示的所有路径的集合的路径长度的最小值

状态计算根据上一点的位置是0, 1,..., n-1划分

`f[i][j] = min(f[i - {j}][k] + a[k][j]), k = 0, 1, 2,...,n-1`

```
const int N = 20, M = 1 << N;
int a[N][N], f[M][N];

int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            cin >> a[i][j];

    memset(f, 0x3f, sizeof f);
    f[1][0] = 0;
    // f[i][j] 表示走过的点集合为i，走到点j的所有路径
    // 根据上一点k的不同取法划分计算 f[i][j] = f[i - {j}][k] + a[k][j]
    for (int i = 0; i < (1 << n); i++)
        for (int j = 0; j < n; j++) 
            // 注意判断状态的合法性
            if (i >> j & 1) {
                for (int k = 0; k < n; k++) 
                    if (i >> k & 1) {
                        f[i][j] = min(f[i][j], f[i - (1 << j)][k] + a[k][j]); 
                    }
            }
    cout << f[(1 << n) - 1][n-1] << endl;
}
```



###### 棋盘类 - 骨牌的完美覆盖

状态f(i, j)表示第i列第j个状态，j状态位等于1表示上一列有横放格子，本列有格子捅出来

```
const int N = 12, M = 1 << 12;
long long f[N][M];
bool st[M];

bool check(int j, int k, int n) {
    int x = j | k;
    int cnt = 0;
    // 下面做法错误，因为没有考虑二进制状态表示中前导0为奇数个的情况
    // do {
    //     if (x % 2 == 0) cnt ++;
    //     else {
    //         if (cnt & 1) return false;
    //         cnt = 0;
    //     }
    //     x /= 2;
    // } while(x);
    // if (cnt & 1) return false;
    for (int i = 0; i < n; i++) {
        if (x >> i & 1) {
            if (cnt & 1) return false;
            cnt = 0;
        } else cnt ++;
    }
    if (cnt & 1) return false;
    return true;
}

int main() {
    int n, m;
    while (cin >> n >> m && n || m) {
        memset(f, 0, sizeof f);
        f[0][0] = 1;
        
        for (int j = 0; j < (1 << n); j++) {
            int cnt = 0;
            st[j] = true;
            for (int i = 0; i < n; i++) {
                if (j >> i & 1) {
                    if (cnt & 1) { 
                        st[j]=false; 
                        break;
                    }
                    cnt = 0;
                } else cnt ++;
            }
            if (cnt & 1) st[j] = false;
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 0; j < (1 << n); j++) {
                for (int k = 0; k < (1 << n); k++) {
                    // j 和 k 同一位不都为1
                    // j 和 k 不能为连续奇数个0
                    if (!(j & k) && st[j | k]) {
                        f[i][j] += f[i-1][k];
                    }
                }
            }
        }    
        
        cout << f[m][0] << endl;
    }
}
```



##### 树形DP

没有上司的舞会



