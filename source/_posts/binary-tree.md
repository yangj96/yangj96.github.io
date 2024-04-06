---
title: 二叉树题目总结
date: 2019-10-01 12:35:21
categories: Leetcode
---

二叉树常见题目总结：

1. 二叉树遍历/分解
2. 二叉树花式遍历
3. 二叉树构造
4. 二叉树序列化
5. 二叉搜索树
   1. BST性质利用
   2. BST判定/增删改查
   3. BST构造

#### 二叉树遍历/分解

**二叉树的深度**

每个节点仅被遍历一次，所以时间复杂度是 O(n)

    int maxDepth(TreeNode* root) {
        return root ? max(maxDepth(root->left), maxDepth(root->right)) + 1 : 0;
    }

**二叉树翻转**

每个节点仅被遍历一次，所以时间复杂度是 O(n)

```
// 递归
TreeNode* invertTree(TreeNode* root) {
    if (!root) return 0;
    swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);
    return root;
}
// 分解
TreeNode* invertTree(TreeNode* root) {
    if (root == NULL) {
        return NULL;
    }
    TreeNode* left = invertTree(root->left);
    TreeNode* right = invertTree(root->right);
    root->left = right;
    root->right = left;
    return root;
}
```

**填充节点的右侧指针**

```
// 递归
Node* connect(Node* root) {
    if (root == nullptr) return nullptr;
    traverse(root->left, root->right);
    return root;
}

void traverse(Node* node1, Node* node2) {
    if (node1 == nullptr || node2 == nullptr) {
        return;
    }
    node1->next = node2;
    
    traverse(node1->left, node1->right);
    traverse(node2->left, node2->right);
    traverse(node1->right, node2->left);
}
// 层序遍历，每个节点仅会遍历一次，遍历时修改指针的时间复杂度是 O(1)，总时间复杂度是 O(n)
void connect(TreeLinkNode *root)
{
    if (!root) return;
    TreeLinkNode *last = root;
    while (last->left) // 直到遍历到叶节点
    {
        for (TreeLinkNode *p = last; p; p = p->next)
        {
            p->left->next = p->right;
            if (p->next) p->right->next = p->next->left;
            else p->right->next = 0;
        }
        last = last->left;
    }
}
// 层序遍历，非完美二叉树
Node* connect(Node *root) {
    auto head = root;
    while (root)
    {
        Node *dummy = new Node(0);
        Node *tail = dummy;
        while (root)
        {
            if (root->left)
            {
                tail->next = root->left;
                tail = tail->next;
            }
            if (root->right)
            {
                tail->next = root->right;
                tail = tail->next;
            }
            root = root->next;
        }
        root = dummy->next;
    }
    return head;
}
```



#### 二叉树花式遍历

**zigzag遍历**

```
vector<int> get_val(vector<TreeNode*> level)
{
    vector<int> res;
    for (auto &u : level)
        res.push_back(u->val);
    return res;
}

vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>>res;
    if (!root) return res;
    vector<TreeNode*>level;
    level.push_back(root);
    res.push_back(get_val(level));
    bool zigzag = true;
    while (true)
    {
        vector<TreeNode*> newLevel;
        for (auto &u : level)
        {
            if (u->left) newLevel.push_back(u->left);
            if (u->right) newLevel.push_back(u->right);
        }
        if (newLevel.size())
        {
            vector<int>temp = get_val(newLevel);
            if (zigzag)
                reverse(temp.begin(), temp.end());
            res.push_back(temp);
            level = newLevel;
        }
        else break;
        zigzag = !zigzag;
    }
    return res;
}
```



#### 二叉树构造

##### 重建二叉树

递归版本

利用哈希数组记录中序遍历中每个值对应的位置

```
    unordered_map<int,int> pos;

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
        for (int i = 0; i < n; i ++ )
            pos[inorder[i]] = i;
        return dfs(preorder, inorder, 0, n - 1, 0, n - 1);
    }

    TreeNode* dfs(vector<int>&pre, vector<int>&in, int pl, int pr, int il, int ir)
    {
        if (pl > pr) return NULL;
        int k = pos[pre[pl]] - il;
        TreeNode* root = new TreeNode(pre[pl]);
        root->left = dfs(pre, in, pl + 1, pl + k, il, il + k - 1);
        root->right = dfs(pre, in, pl + k + 1, pr, il + k + 1, ir);
        return root;
    }
```

非递归版本





##### 和为某一定值的二叉树路径

DFS

```
vector<vector<int>> ans;
vector<int> path;
vector<vector<int>> findPath(TreeNode* root, int sum) {
    dfs(root, sum);
    return ans;
}

void dfs(TreeNode* root, int sum) {
    if (!root) return;
    path.push_back(root->val);
    sum -= root->val;
    if (!root->left && !root->right && !sum) ans.push_back(path);
    dfs(root->left, sum);
    dfs(root->right, sum);
    path.pop_back();
    // sum值传递可以不恢复
}
```

##### 判断对称二叉树

- 单队列迭代，相邻两个队列成员应该一致

- 用栈模拟递归，对根节点的左子树中序遍历；对根节点的右子树反中序遍历
  则两个子树互为镜像，当且仅当同时遍历两课子树时，对应节点的值相等

```
bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    stack<TreeNode*> left, right;
    TreeNode *lc = root->left;
    TreeNode *rc = root->right;
    while(lc || rc || left.size())
    {
        while (lc && rc)
        {
            left.push(lc), right.push(rc);
            lc = lc->left, rc = rc->right;
        }
        if (lc || rc) return false;
        lc = left.top(), rc = right.top();
        left.pop(), right.pop();
        if (lc->val != rc->val) return false;
        lc = lc->right, rc = rc->left;
    }
    return true;
}
```

- 递归，除根节点外，任意两个子树互为镜像当且仅当：

两个子树的根节点值相等；
第一棵子树的左子树和第二棵子树的右子树互为镜像，且第一棵子树的右子树和第二棵子树的左子树互为镜像

    bool isSymmetric(TreeNode* root) {
        if (!root) return true;
        return dfs(root->left, root->right); 
    }
    
    bool dfs(TreeNode* l, TreeNode* r) {
        if (!l || !r) return !l && !r;
        return l->val == r->val && dfs(l->right, r->left) && dfs(l->left, r->right);
    }

##### 平衡二叉树判定

    bool ans = true;
    
    bool isBalanced(TreeNode* root) {
        dfs(root);
        return ans;
    }
    
    int dfs(TreeNode *root)
    {
        if (!root) return 0;
        int left = dfs(root->left), right = dfs(root->right);
        if (abs(left - right) > 1) ans = false;
        return max(left, right) + 1;
    }

  ##### 二叉树的下一个结点

```
TreeNode* inorderSuccessor(TreeNode* p) {
    if(!p) return  NULL;
    if (p->right) {
        p = p->right;
        while(p->left) p = p->left;
        return p;
    }
    
    while(p->father && p == p->father->right) p = p->father;
    return p->father;
}
```

##### 生成镜像二叉树

所有节点左右孩子互换

    void mirror(TreeNode* root) {
        if (!root) return;
        swap(root->left, root->right);
        mirror(root->left);
        mirror(root->right);
    }

##### 同构二叉树

两个队列边遍历边比较

##### 树的子结构

暴力匹配 找到相同的根节点 同时遍历两颗树

代码分为两个部分：

遍历树A中的所有非空节点R；
判断树A中以R为根节点的子树是不是包含和树B一样的结构，且我们从根节点开始匹配；
对于第一部分，我们直接递归遍历树A即可，遇到非空节点后，就进行第二部分的判断。

对于第二部分，我们同时从根节点开始遍历两棵子树：

如果树B中的节点为空，则表示当前分支是匹配的，返回true；
如果树A中的节点为空，但树B中的节点不为空，则说明不匹配，返回false；
如果两个节点都不为空，但数值不同，则说明不匹配，返回false；
否则说明当前这个点是匹配的，然后递归判断左子树和右子树是否分别匹配即可；

    bool hasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (!pRoot1 || !pRoot2) return false;
        if (isSame(pRoot1, pRoot2)) return true;
        return hasSubtree(pRoot1->left, pRoot2) || hasSubtree(pRoot1->right, pRoot2);
    }
    
    bool isSame(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (!pRoot2) return true;
        if (!pRoot1 || pRoot1->val != pRoot2->val) return false;
        return isSame(pRoot1->left, pRoot2->left) && isSame(pRoot1->right, pRoot2->right);
    }

##### 

##### 二叉树的花样遍历

加层数 或更改结构体加上int layer;

##### 按之字形顺序打印二叉树

将根节点插入队列中；
创建一个新队列，用来按顺序保存下一层的所有子节点；
对于当前队列中的所有节点，按顺序依次将儿子插入新队列；
按从左到右、从右到左的顺序交替保存队列中节点的值；
重复步骤2-4，直到队列为空为止。

    vector<int> get_val(vector<TreeNode*> level)
    {
        vector<int> res;
        for (auto &u : level)
            res.push_back(u->val);
        return res;
    }
    
    vector<vector<int>> printFromTopToBottom(TreeNode* root) {
        vector<vector<int>>res;
        if (!root) return res;
        vector<TreeNode*>level;
        level.push_back(root);
        res.push_back(get_val(level));
        bool zigzag = true;
        while (true)
        {
            vector<TreeNode*> newLevel;
            for (auto &u : level)
            {
                if (u->left) newLevel.push_back(u->left);
                if (u->right) newLevel.push_back(u->right);
            }
            if (newLevel.size())
            {
                vector<int>temp = get_val(newLevel);
                if (zigzag)
                    reverse(temp.begin(), temp.end());
                res.push_back(temp);
                level = newLevel;
            }
            else break;
            zigzag = !zigzag;
        }
        return res;
    }



##### 分行打印二叉树

- 滚动数组

```
vector<int> get_val(vector<TreeNode*> level)
{
    vector<int> res;
    for (auto &u : level)
        res.push_back(u->val);
    return res;
}

vector<vector<int>> printFromTopToBottom(TreeNode* root) {
    vector<vector<int>>res;
    if (!root) return res;
    vector<TreeNode*>level;
    level.push_back(root);
    res.push_back(get_val(level));
    while (true)
    {
        vector<TreeNode*> newLevel;
        for (auto &u : level)
        {
            if (u->left) newLevel.push_back(u->left);
            if (u->right) newLevel.push_back(u->right);
        }
        if (newLevel.size())
        {
            res.push_back(get_val(newLevel));
            level = newLevel;
        }
        else break;
    }
    return res;
}
```

- 在每行末尾添加null标记

```
vector<vector<int>> printFromTopToBottom(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> que;
    que.push(root);
    que.push(nullptr);
    vector<int> level;
    
    while(que.size()) {
        auto p = que.front();
        que.pop();
        if (!p) {
            if (level.empty()) break;
            res.push_back(level);
            level.clear();
            que.push(nullptr);
        } else {
            level.push_back(p->val);
            if (p->left) que.push(p->left);
            if (p->right) que.push(p->right);
        }
    }
    return res;
}
```

#### 二叉树序列化

序列化二叉树



#### 二叉搜索树





二叉搜索树与双向链表

每次递归返回一个pair<TreeNode\*, TreeNode\*>



##### BST性质利用

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

##### 二叉搜索树的第k个结点

中序遍历的第k个节点，即第k小的结点

    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
     * };
     */
    TreeNode *ans;
    
    TreeNode* kthNode(TreeNode* root, int k) {
        dfs(root, k);
        return ans;
    }
    
    void dfs(TreeNode *root, int &k)
    {
        if (!k || !root) return;
        dfs(root->left, k);
        --k;
        if (!k) ans = root;
        else dfs(root->right, k);
    }





##### BST判定

##### 判定二叉搜索树后序序列的合法性

注意dfs中[l,r]只包含一个元素的边界判断和子区间递归要剔除根节点

```
vector<int> seq;
bool verifySequenceOfBST(vector<int> sequence) {
    seq = sequence;
    if (seq.size() == 0) return true;
    return dfs(0, seq.size() - 1);
}
bool dfs(int l, int r) {
    if (l >= r) return true;
    int x = seq[r];
    int k = l;
    while (k < r && seq[k] < x) k++;
    for (int i = k; i < r; i++)
        if (seq[i] < x) return false;
    return dfs(l, k-1) && dfs(k, r-1);
}};
```

##### BST增删改查

##### 



 