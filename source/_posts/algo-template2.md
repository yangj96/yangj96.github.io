---
title: 算法模版-中级篇
date: 2020-02-03 19:39:44
categories: Algorithm
---

### 目录
1. 离散化
2. 二分判定
3. 区间合并/区间贪心调度
4. 前缀和/差分
5. 树状数组
6. 线段树
7. 图论
8. 字符串匹配
9. Trie树
10. 动态规划

#### 离散化

待离散值排序、去重，然后二分求离散化对应坐标



单值离散化

```
vector<int> all;
sort(all.begin(), all.end());
all.erase(unique(all.begin(), all.end()), all.end());

// 待离散化值y->离散化后x
int x = find(all.begin(), all.end(), y) - all.begin();
// or
int bs(int k) {
    int l = 0, r = all.size() - 1;
    while(l < r) {
        int mid = l + r >> 1;
        if (all[mid] >= k) r = mid;
        else l = mid + 1;
    }
    return r + 1;
}
int x = bs(y);
```

线段坐标离散化

每个端点需要考虑其本身和前后两点，从而将线段压缩

```
int compress(vector<int> &x1, vector<int> &x2, int w) {
	vector<int> all;
	for (int i = 0; i < x1.size(); i++) {
		for (int d = -1; d <= 1; d++) {
			int tx1 = x1[i] + d, tx2 = x2[i] + d;
			if (tx1 >= 1 && tx1 <= w) all.push_back(tx1);
			if (tx2 >= 1 && tx2 <= w) all.push_back(tx2);
		}
	}
	
	sort(all.begin(), all.end());
	all.erase(unique(all.begin(), all.end()), all.end());
	
	for (int i = 0; i < x1.size(); i++) {
		x1[i] = find(all.begin(), all.end(), x1[i]) - all.begin();
		x2[i] = find(all.begin(), all.end(), x2[i]) - all.begin();
	}
	
	return all.size();
}
```

#### 二分判定

给定N个数，将其分为X组，每组K = N / X个数且需要保证K个数互不重复，求出每组最大数和最小数间的差值，求能够实现的所有组该差值总和的最小值

```
bool check(int n, int k, vector<int>& a, int mid) {
	
}

int xGroup(int n, int k, vector<int>& a) {
	int maxn = 0, minn = 2e9;
	for (int i = 0; i < n; i++) {
		minn = min(a[i], minn);
		maxn = max(a[i], maxn);
	}
	int l = 0, r = (maxn - minn) * (n / k);
	while (l < r) {
		int mid = l + r >> 1;
		if (check(n, k, a, mid)) r = mid;
		else l = mid + 1;
	}
	return l;
}
```

#### 区间问题

##### 区间合并

按区间左端点排序

```
void merge(vector<PII> & segs) {
    vector<PII> res;
    sort(segs.begin(), segs.end());
    int st = -2e9, ed = -2e9; 
    for(auto seg : segs) {
        if (ed < seg.first) {
            if (st != -2e9) 
                res.push_back({st, ed});
            st = seg.first;
            ed = seg.second;
        }
        else 
            ed = max(ed, seg.second);
    }
    if (st != -2e9) {
        res.push_back({st, ed});
    }
    segs = res;
}
```

##### 区间覆盖问题

###### 求覆盖各区间的最少点数(每个区间至少包含一个点)

按区间右端点排序，选取最右点

```
typedef pair<int, int> PII;

bool cmp(const PII &a, const PII &b) {
    return a.second < b.second;
}
int main() {
    int n;
    cin >> n;
    int a, b;
    vector<PII> v;
    for (int i = 0; i < n; i++) {
        cin >> a >> b;
        v.push_back({a, b});
    }
    sort(v.begin(), v.end(), cmp);
    int res = 0, ed = -2e9;
    for (int i = 0; i < n; i++) {
        if (ed < v[i].first) {
            ed = v[i].second;
            res ++;
        }
    }
    cout << res << endl;
}
```

###### 区间调度问题/求不相交区间的最多区间数目

如果最多有x个不相交区间，那么就至少需要x个点覆盖所有区间，因为本题和上一题等价，只有区间边界可能略有不同

选取结束时间最早的活动 -> 按区间右端点排序，选取max_ed和下一个区间起始不相交的位置



###### 求按照不相交区间分组的最少分组数目

按区间左端点排序，记录每个组的最右坐标，依次枚举每个区间能否加入各个组，如果不能则新开一组。实际上枚举每个区间是否有可加入的组只需要找右坐标最小的组即可。

```
#define  l first
#define  r second
typedef pair<int, int> PII;
int main() {
    int n;
    cin >> n;
    vector<PII> a(n);
    for (int i = 0; i < n; i++) {
        int x, y;
        cin >> x >> y;
        a[i] = {x, y};
    }
    sort(a.begin(), a.end());
    
    priority_queue<int, vector<int>, greater<int>> que;
    for (int i = 0; i < n; i++) {
        if (que.empty() || a[i].l <= que.top()) {
            que.push(a[i].r);
        } else {
            que.pop();
            que.push(a[i].r);
        }
    }
    cout << que.size() << endl;
}
```

###### 求覆盖指定线段的最少区间数目

按区间左端点排序，依次枚举选取满足左端点在线段左侧的右端点最大的区间

```
sort(range, range + n);

int res = 0;
bool success = false;
for (int i = 0; i < n; i ++ )
{
    int j = i, r = -2e9;
    while (j < n && range[j].l <= st) {
        r = max(r, range[j].r);
        j ++ ;
    }

    if (r < st) {
        res = -1;
        break;
    }

    res ++ ;
    if (r >= ed) {
        success = true;
        break;
    }

    st = r;
    i = j - 1;
}

if (!success) res = -1;
cout << res << endl;
```

#####区间交集问题

两区间[x1, y1]和[x2, y2]不存在交集的条件是 `y1 < x2 || y2 < x1`，反之，则交集区间是`[max(x1, x2), min(y1, y2)]`



#### 前缀和/差分 

##### 一维前缀和

为便于计算，下标从1开始

`s[0] = a[0] = 0`

`s[i] = s[i-1] + a[i];`

求区间[l, r]元素和`s[r] - s[l-1]`

##### 二维前缀和

`s[i][j] = s[i-1][j] + s[i][j-1] - s[i-1][j-1] + a[i][j]`

求区间[(x1, y1), (x2, y2)]元素和`s[x2][y2] - s[x1-1][y1] - s[x1][y1-1] + s[x1-1][y1-1]`

##### 一维差分

差分可看作前缀和的逆操作，可实现$O(1)$时间的区间修改和单点修改

```
void modify(int l, int r, int c) {
    b[l] += c;
    b[r+1] -= c;
}

// 复原原矩阵更新后的值
for (int i = 1; i <= n; i++) {
    b[i] += b[i-1];
}
```

##### 二维差分

```
const int N = 1005; // 注意数组从1开始且设计N+1，N至少要大于等于2 
int b[N][N];

void modify(int x1, int y1, int x2, int y2, int c) {
    b[x1][y1] += c;
    b[x2+1][y1] -= c;
    b[x1][y2+1] -= c;
    b[x2+1][y2+1] += c;
}

// 复原原矩阵更新后的值
for (int i = 1; i <= n; i++) {
  for (int j = 1; j <= m; j++) {
  	b[i][j] += b[i-1][j] + b[i][j-1] - b[i-1][j-1];
  }
}
```

###### 差分+贪心
[求区间内所有值通过区间加1/减1全部相等的最小变换次数](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.acwing.com%2Fsolution%2FAcWing%2Fcontent%2F816%2F)



###### 差分+前缀和

[满足互相看见约束的序列各位置的最高可能高度](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.acwing.com%2Fsolution%2FAcWing%2Fcontent%2F817%2F)



#### 树状数组

树状数组是支持区间单点修改的前缀和

![](https://tva1.sinaimg.cn/large/00831rSTly1gchu0dzxigj30yg0hw3zs.jpg)
将上图所有区间从左至右按序排列，其区间长度的二进制表示为：
1,10,1, 100, 1,   10,  1,    1000
而图中区间标号对应的二进制表示为：
1,10,11,100,101,110,111,1000
用lowbit函数将区间标号映射为区间长度：区间长度即区间标号二进制表示从右往左出现第一个1以及这个1之后的那些0组成数的二进制对应的十进制的数。

区间标号映射区间长度

```
int lowbit(int x) {
	return x & -x;
}
```

初始化空间$O(n)$，注意下标从1到n

单点修改操作需要更新所有包含它的区间，时间复杂度$O(log n)$

```
int tr[N];

void add(int x, int c) {
	for (int i = x; i <= n; i += lowbit(i)) 
		tr[i] += c;
}
```
区间查询需要查询分支包含的所有区间，时间复杂度$O(log n)$

```
// a[1]...a[x]
int presum(int x) {
  int res = 0;
  for (int i = x; i; i -= lowbit(i))
  	res += tr[i];
}

// a[i]...a[j]
int sum(int i, int j)
  return presum(j) - presum(i-1);
```
e.x. 计算右侧小于当前元素的个数

##### 差分+树状数组

可实现区间修改、单点查询或区间修改、区间查询

##### 二维树状数组



#### 线段树

五类操作， 四倍空间，初始化复杂度$O(n)$，区间操作查询或更新复杂度均为$O(\lg n)$

pushdown操作用于区间修改时的懒标记（仅支持单点修改时不需要），在区间修改和查询需要分裂区间前调用

##### 支持单点修改的区间最大值查询

```
const int N = 200010;
struct Node {
    int l, r;
    int v;
}tr[N * 4];

void pushup(int u) {
    tr[u].v = max(tr[u << 1].v, tr[u << 1 | 1].v);
}

void build(int u, int l, int r) {
    tr[u] = {l, r};
    if (l == r) return;
    int mid = l + r >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
    // 如果初始化时对v有修改，则需要调用pushup(u);
}

// 单点修改
void modify(int u, int x, int v) {
    if (tr[u].l == x && tr[u].r == x) 
        tr[u].v = v;
    else {
        int mid = tr[u].l + tr[u].r >> 1;
        if (x <= mid) modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}

int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) 
        return tr[u].v;
    int mid = tr[u].l + tr[u].r >> 1;
    int v = 0;
    // 注意递归调用的查询区间仍然是[l, r]，使用[l, mid]可以会放大空间
    if (l <= mid) v = query(u << 1, l, r);
    if (r > mid) v = max(v, query(u << 1 | 1, l, r));
    return v;
}

int main () {
    int m, p;
    cin >> m >> p;
    int n = 0, last = 0;
    build(1, 1, m);
    for (int i = 0; i < m; i++) {
        char op;
        int x;
        cin >> op >> x;
        if (op == 'Q') {
            last = query(1, n-x+1, n);
            cout << last << endl;
        } else {
            modify(1, n+1, (x + last) % p);
            n++;
        }
    }
}
```

##### 支持区间修改的区间和查询

```
const int N = 100010;
typedef long long ll;
int a[N];
struct Node {
    int l, r;
    ll sum, add;
}tr[N * 4];

void pushup(int u) {
    tr[u].sum = tr[u<<1].sum + tr[u << 1 | 1].sum;
}

void pushdown(int u) {
    auto &root = tr[u], &left = tr[u<<1], &right = tr[u << 1 | 1];
    if (root.add) {
        left.add += root.add;
        left.sum += (ll)(left.r - left.l + 1) * root.add;
        right.add += root.add;
        right.sum += (ll)(right.r - right.l + 1) * root.add;
        root.add = 0;
    }
}

void build(int u, int l, int r) {
    if (l == r) tr[u] = {l, r, a[l], 0};
    else {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

// 区间修改
void modify(int u, int l, int r, int d) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += (ll)(tr[u].r - tr[u].l + 1) * d;
        tr[u].add += d;
    } else {
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if (l <= mid) modify(u << 1, l, r, d);
        if (r > mid) modify(u << 1 | 1, l, r, d);
        pushup(u);
    }
}

ll query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) 
        return tr[u].sum;
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    ll v = 0;
    if (l <= mid) v = query(u << 1, l, r);
    if (r > mid) v += query(u << 1 | 1, l, r);
    return v;
}

int main() {
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++) 
        cin >> a[i];

    build(1, 1, n);
    while (m--) {
        int l, r, d;
        char op;
        cin >> op >> l >> r;
        if (op == 'C') {
            cin >> d;
            modify(1, l, r, d);
        } else {
            cout << query(1, l, r) << endl;
        }
    }
}
```



#### 图论

邻接矩阵
`int G [maxv][maxv] `或` <vector<vector<int> > G`

邻接表
`vector<int> G[maxv]`

`struct edge {
  int to;
  int cost;
}`
`vector<edge> G[maxv]`

##### BFS/DFS搜索

时间复杂度均为$O(V+ E)$

##### 染色法判定二分图

```
vector<int> G[maxv];
int color[maxv];
int V;

bool dfs(int v, int c) {
	color[v] = c;
	for (int i = 0; i < G[v].size(); i++) {
		if (color[G[v][i]] == c) return false;
		if (color[G[v][i] == 0 && !dfs(G[v][i], -c)) return false;
	}
	return true;
}

bool check() {
	bool flag = true;
	for (int i = 0; i < V; i++)
		if (color[i] == 0)
			if (!dfs(i, 1)) {
				flag = false;
				break;
			}
	return flag;
}
```



##### 拓扑排序

拓扑图（可以拓扑排序的图） 等价于有向无环图DAG

将入度为0点入队，出队去边减去相关入度将入度为0点入队，队列元素即拓扑序，队列元素小于顶点数说明可能存在重边和自环，拓扑序列不存在

$O(V+E)$

```
const int N = 100001;
vector<int> G[N];
int d[N]; // 入度数
int n, m;
// 使用stl queue元素会pop出队需要另开数组单独记录结果或者直接使用数组模拟队列
vector<int> res; 

bool topsort() {
    queue<int> q;
    // 如果最终拓扑序需要按字典序输出，则将队列改为小根堆
    for (int i = 1; i <= n; i++) 
        if (!d[i]) {
            q.push(i);
            res.push_back(i);
        }
    while (q.size()) {
        int t = q.front();
        q.pop();
        for (int i = 0; i < G[t].size(); i++)
            if (--d[G[t][i]] == 0) {
                q.push(G[t][i]);
                res.push_back(G[t][i]);
            }
    }
    return res.size() == n;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        G[a].push_back(b);
        d[b]++;
    }
    
    if (!topsort()) cout << -1 << endl;
    else {
        for (int i = 0; i < n; i++) 
            cout << res[i] << " ";
        cout << endl;
    }
}
```



##### 最短路问题

###### Bellman-Ford 允许负环

每条边松弛|V| -1次（最坏情况下每次循环只松弛了一条边）之后如果存在不满足三角不等式的结点v.d > u.d + w(u,v)说明存在负权重环

时间复杂度$O(VE)$

优化 - 拓扑排序后按序松弛



###### Dijkstra 不许负权重边

维护一个已求出最短路径节点的集合S，以v.d为key构造最小堆，每次选择V-S中的最小堆顶，将其加入S并松弛所有与其相邻的边。注意第一次执行循环extract-min得到的是源点s

优先队列实现时间复杂度$O(VE)$



###### Floyd 适用负权重边，不允许存在负权重环

 时间复杂度$O(V^3)$



##### 最小生成树

Kruskal算法：集合A是森林，按权重从低到高考察每条边，如果它将两棵不同的树连接起来就加入到森林A里并完成两棵树的合并



Prim算法：集合A是一棵树，每次加入连接集合A和A之外结点的所有边中权重最小的边

用并查集和优先队列分别实现，时间复杂度均为$O(ElgV)$



#### 字符串匹配

##### 暴力

时间复杂度$O(n-m+1)*m$

```
for (int i = 1; i <= (n - m + 1); i++) {
	bool flag = true;
	for (int j = 1; j <= m; j++) 
		if (s[i + j - 1] != p[j]) {
			flag = false;
			break;
		}
}
return flag;
```



##### KMP

前缀函数 $\pi[q]$是能构成$P_q (即P[1...q])$**真**后缀的P的最长前缀长度
$\pi[q] = max(k : k < q 且 P_k \sqsupset P_q)$

next[i]表示以p[i]结尾的p的子串的前缀函数值，即next[i] = j 表示 p[1...j] == p[i - j + 1...i]



预处理阶段摊还分析，时间复杂度$\Theta(m)$，因为j最多++ m次，因此while循环最多执行m次，同理匹配阶段时间复杂度$\Theta(n)$



字符串下标从1开始，next[1] = 0

````C++
// 待匹配串s，模式串p，下标从1开始
int n+1 = s.size();
int m+1 = p.size();
// 求next数组, next[1] = 0
for (int i = 2, j = 0; i <= m; i++) {
  while(j && p[i] != p[j+1]) j = ne[j];
  if (p[i] == p[j+1]) j++;
  ne[i] = j;
}

// kmp匹配
for (int i = 1, j = 0; i <= n; i++) {
  // j表示当前模版串下一个要匹配位置的前一位
  // j == 0 表示j退回到起点
  // 如果j下一个位置不能匹配，则匹配串需要后移j-next[j]步，新的匹配末端位置即j-(j-next[j])
  while(j && s[i] != p[j+1]) j = ne[j];
  if (s[i] == p[j+1]) j++;
  if (j == m) {
    j = ne[j];
    // 匹配成功后的逻辑
		//e.x. cout << i - m << endl;
  }
}
````



字符串下标从0开始，next[0] = -1

```
ne[0] = -1;
for (int i = 1, j = -1; i < m; i ++ )
{
    while (j >= 0 && p[j + 1] != p[i]) j = ne[j];
    if (p[j + 1] == p[i]) j ++ ;
    ne[i] = j;
}

for (int i = 0, j = -1; i < n; i ++ )
{
    while (j && s[i] != p[j + 1]) j = ne[j];
    if (s[i] == p[j + 1]) j ++ ;
    if (j == m - 1) cout << i - j << ' ';
}
```



#### Trie 树

高效存储和查询字符串的集合

插入和查询时间复杂度$O(\log n)$，时间复杂度$O(n^2)$

```c++
const int N = 100010;

int son[N][26]; // Trie树每个节点的字节点，此处英文字母只包含26个小写字母
int cnt[N];// 以当前这个点结尾的单词数量
int idx; // 表示层数下标，0号既是空节点也是Trie树的根节点

void insert(string str) {
	int p = 0;
	for (int i = 0; i < str.size(); i++) {
		int u = str[i] - 'a';
		if (!son[p][u]) son[p][u] = ++idx;
    p = son[p][u];
	}
  cnt[p] ++;
}

int query(string str) {
  int p = 0; 
  for (int i = 0; i < str.size(); i++) {
    int u = str[i] - 'a';
    if (!son[p][u]) return 0;
    p = son[p][u];
  }
  return cnt[p];
}
```



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

每类物品可以使用无限次，与01背包的区别主要在于集合的划分变为$f[i, j] = f[i-1, j-v[i]*k] + k*w[i]$![image](https://tva1.sinaimg.cn/large/0082zybply1gc2qahne9fj31760a0agz.jpg)

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

