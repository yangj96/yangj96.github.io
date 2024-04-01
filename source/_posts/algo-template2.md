---
title: 算法模版-中级篇
date: 2020-02-03 19:39:44
categories: Algorithm
---

### 目录
1. 位运算
2. 初级数论(辗转相除/素数/快速幂)
3. 离散化
4. 二分判定
5. 区间合并/区间贪心调度
6. 前缀和/差分
7. 树状数组
8. 线段树
9. 图论
10. 字符串匹配
11. Trie树
12. 高精度

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

BFS优化：

##### 多源BFS

矩阵各点到多个候选起点的最短距离

可假设一虚拟源点，将其与多个起点分别相连，则转换为单源BFS的最短距离，实际实现时只需要将多个起点在第一轮都加入队列即可

##### 双端队列BFS

适用于不同边权的情况

##### BFS优化

双向广搜

A*



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

