---
 
title: 算法模版-初级篇
date: 2019-12-31 12:33:54
categories: Algorithm
---

### 目录

1. 二分
2. 双指针
3. 单调栈
4. 单调队列
5. 二叉树
6. 归并排序
7. 快速排序&快速选择
8. 堆
9. 双指针/尺取法
10. 单调栈/单调队列
11. 回溯
12. 递归/DFS
13. BFS
14. 并查集
15. 位运算
16. 高精度
17. 初级数论
18. 二叉树

#### 二分

二分本质不是单调性，只需要区间针对某个性质能够分成两段，一段满足一段不满足即可。

找到能够划分区间左右两半的性质，如果if (check(mid)) 条件成立，判断答案在左区间还是右区间，如果答案在**左区间并且mid也可能是答案**，按模板**1**来划分；如果答案在**右区间并且mid也可能是答案**，按模板**2**来划分（）。

模板1mid使用下中位数，模板2使用下中位数+1，终结条件为$low==high$，注意区间左右均为闭区间

**版本1**
最大值最小问题，第一个>=target的元素，满足check条件的区间左边界
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
最小值最大问题，最后一个<= target的元素，找满足check条件的区间右边界
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

数组有序，利用某种单调性确保 i，j两指针保持相同的移动方向

常见题型：

单数组 - 满足条件的子串

两数组 - 关键是确定两指针的单向移动方向：快慢指针、左右指针

```
for (int i = 0, j = 0; i < n; i++) {
	// i在前，j在后
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

##### 最长无重复字符子串







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

##### 最小覆盖子串

两个哈希表分别记录被覆盖子串的各字符出现次数和当前窗口各字符出现的次数

滑动窗口right指针右移至所有字符被覆盖，然后left指针右移至满足条件最大值，直至right移动到末尾

##### 找到字符串中所有给定子串的字母异位子串

##### 和最大的子数组

##### 平均数最大子数组

#### 二叉树

二叉树的思路：





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

##### 

#### 快速排序&快速选择

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

#### 堆

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

##### 多源BFS

矩阵各点到多个候选起点的最短距离

可假设一虚拟源点，将其与多个起点分别相连，则转换为单源BFS的最短距离，实际实现时只需要将多个起点在第一轮都加入队列即可

##### 双端队列BFS

适用于不同边权的情况

##### BFS优化

双向广搜

A*

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





#### 位运算

##### 取最靠右的一位1

```
int lowbit(int x) {
	return x & -x;
}
```

##### 二进制中1的个数

```
while(x) {
	x -= lowbit(x);
	res++;
}
```

##### 获取/设置右起第k位数

```
// 获取第k位
n >> k & 1
// 设置第k位为1
n | 1 << k
```



#### 高精度

##### 高精度加法

数据范围为数字位数而非数字本身，使用string读入，vector逆序存储便于进位

```
vector<int> add(vector<int> &a, vector<int> &b) {
    if (a.size() < b.size())    
        return add(b, a);
    int t = 0;
    vector<int> c;
    for (int i = 0; i < a.size(); i++) {
        t += a[i];
        if (i < b.size()) t += b[i];
        c.push_back(t % 10);
        t /= 10;
    }
    if (t) c.push_back(t);
    return c;
}

int main() {
    string a, b;
    cin >> a >> b;
    vector<int> A;
    vector<int> B;
    for (int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0');
    for (int i = b.size() - 1; i >= 0; i--) B.push_back(b[i] - '0');

    vector<int> C = add(A, B);
    reverse(C.begin(), C.end());
    for (int i = 0; i < C.size(); i++) {
        cout << C[i];
    }
    cout << endl;
}
```



##### 高精度减法

负号的判定 `cmp函数：依次判断长度和各个位置的数`

减法进位的处理

```
c.push_back((t + 10) % 10);
if (t < 0) t = 1;
else t = 0;
```

先导0的去除，注意最后结果是0要保留一位0

`while(c.size() > 1 && c.back() == 0) c.pop_back();`

```
bool cmp(vector<int> & a, vector<int> & b) {
    if (a.size() != b.size()) 
        return a.size() > b.size();
    for (int i = a.size() - 1; i >= 0; i--) {
        if (a[i] != b[i])
            return a[i] > b[i];
    }
    return true;
}

vector<int> sub(vector<int> &a, vector<int> &b) {
    int t = 0;
    vector<int> c;
    for (int i = 0; i < a.size(); i++) {
        t = a[i] - t;
        if (i < b.size()) t -= b[i];
        c.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }
    while(c.size() > 1 && c.back() == 0) c.pop_back();
    return c;
}

int main() {
    string a, b;
    cin >> a >> b;
    vector<int> A;
    vector<int> B;
    vector<int> C;
    for (int i = a.size() - 1; i >= 0; i--) {
        A.push_back(a[i] - '0');
    }
    for (int i = b.size() - 1; i >= 0; i--) {
        B.push_back(b[i] - '0');
    }
    if (cmp(A, B)) C = sub(A, B);
    else {
        C = sub(B, A);
        cout << "-";
    }
    for (int i = C.size() - 1; i >= 0; i--) {
        cout << C[i];
    }
    cout << endl;
}
```

##### 高精度乘法

高精度乘整数

```
vector<int> mul(vector<int> &a, int b) {
    vector<int> c;
    int t = 0;
    for (int i = 0; i < a.size() || t; i++) {
        if (i < a.size()) t += a[i] * b;
        c.push_back(t % 10);
        t /= 10;
    }
    return c;
}
```

高精度乘高精度

```
vector<int> mul(vector<int> &a, vector<int> &b) {
    vector<int> res(alen + blen, 0);
    // i*j存放i+j
    for (int i = 0; i < alen; i++) {
        for (int j = 0; j < blen; j++) {
  	          res[i + j] += a[i] * b[j];
        }
    }
    int t = 0;
    for (int i = 0; i < (int)res.size(); i++) {
	      t += res[i];
	      res[i] = t % BASE;
	      t /= BASE;
    }
    while (res.size() > 1 && res.back() == 0) {
        res.pop_back();
    }
}
```

##### 高精度除法

高精度除整数

```
vector<int> div(vector<int> &a, int b, int &r) {
    vector<int> c;
    r = 0;
    for (int i = a.size() - 1; i >= 0; i--) {
        r = r * 10 + a[i];
        c.push_back(r / b);
        r %= b;
    }
    reverse(c.begin(), c.end());
    while(c.size() > 1 && c.back() == 0) c.pop_back();
    return c;
}
```



#### 初级数论

##### 辗转相除法 

###### 最大公约数  

时间复杂度 $O(\lg max(a,b))$

```
int gcd(int a, int b) {
	return b ? gcd(b, a % b) : a;
}
```

###### 最大公倍数

```
int lcm(int a, int b) {
	return a * b / gcd(a, b);
}
```

###### 扩展欧几里得算法

求x, y整数，使得ax + by = gcd(a, b)，时间复杂度 $O(\lg  max(a,b))$

>裴蜀定理
>有任意正整数a, b，gcd（a，b）= d，那么对于任意的整数x，y，ax+by都一定是d的倍数，特别地，一定存在整数x，y，使ax+by=d成立。
>推论
>a,b互素的充要条件是存在整数x，y使ax+by=1

```
int exgcd(int a, int b, int &x, int &y) {
	if (!b) {
		x = 1, y = 0;
		return a;
	}
	int d = exgcd(b, a % b, y, x);
	y -= a / b * x;
	return d;
}
```



##### 素数

###### 素数判定 / 试除法

试除法实现素数判定、约数枚举、整数分解的时间复杂度均为 $O(\sqrt n)$

```
bool isPrime(int x) {
	if (x < 2) return false;
	for (int i = 2; i <= x / i; i++) {
		if (x % i == 0) return false;
	}
	return true;
}
```

约数枚举

```
vector<int> divisor(int x) {
	vector<int> res;
	for (int i = 2; i <= x / i; i++) {
		if (x % i == 0) {
			res.push_back(i);
			if (i != x / i) res.push_back(x / i);
		}
	}
	return res;
}
```



整数分解

```
map<int, int> prime_factor(int x) {
	map<int, int> res;
	for (int i = 2; i <= x / i; i++) {
		while(x % i == 0) {
			res[i]++;
			x /= i;
		}
	}
	if (x != 1) res[x] = 1;
	return res;
}
```



###### 素数筛法

埃氏筛法 时间复杂度 $O(n \lg n lg n)$

```
int prime[N];
int st[N];

int sieve(int n) {
	int p = 0;
	for (int i = 2; i <= n; i++) {
		if (st[i]) continue;
		prime[p++] = i;
		for (int j = i; j <= n; j += i) 
			st[j] = true;
		}
	return p;
}
```

![image-20200903155157579](/Users/bytedance/Library/Application Support/typora-user-images/image-20200903155157579.png)

```
vector<int> primes;
bool st[N];
void get_primes(int n) {
    for (int i = 2; i <= n; i ++ ) {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ ) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}
```

##### 快速幂

求$a^k\mod p$的值，反复平方法 时间复杂度 $O(\lg k)$

预处理出 $a^{2^0} \mod p$, $a^{2^1} \mod p$, $a^{2^2} \mod p$,..., $a^{2^{lgk}} \mod p$的值（反复平方k次），然后根据底数不变指数相加，将k拆分为若干个2的次幂之和，则可以根据k的二进制形式将预处理的值按需相乘

```
typedef long long LL;

ll mod_pow(ll a, ll k, ll p) {
	ll res = 1;
	while (k) {
		if (k & 1) res = res * a % p;
		a = a * a % p;
		k >>= 1;
	} 
	return res;
} 
```







