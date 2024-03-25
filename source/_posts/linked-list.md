---
title: Leetcode链表题目总结
date: 2019-03-25 10:38:08
categories: Leetcode
---

Leetcode链表题常用方法总结：

1. dummy node 常用于链表的head可能被修改或删除的情况，可简化单链表没有前向指针所带来的问题，通常使用current = dummy进行遍历，最终返回 dummy->next

   

2. 链表中尽量避免new新的节点，而是在原链表上直接操作地址

   

3. 在插入和删除操作中使用临时变量来存储next指针

   

4. 反转链表通常需要使用pre指针记录前驱节点

   

5. 通过两个指针几何变换来解决链表长度、环检测等问题

   

6. 对于一些依赖后面节点才能完成的操作，通常使用递归来解决



常见题目：

##### 从尾到头打印链表

反向迭代器rbegin(), rend()，栈，递归

```
    vector<int> printListReversingly(ListNode* head) {
        vector<int> res;
        while (head) {
            res.push_back(head->val);
            head = head->next;
        }
        return vector<int>(res.rbegin(), res.rend());
    }
```



##### O(1)时间删除节点

替换下一节点的值，直接删除下一个节点

尾节点只能从头遍历

```
void deleteNode(ListNode* node) {
	node->val = node->next->val;
	node->next = node->next->next;
}
```



##### 删除重复节点

```
    ListNode* deleteDuplication(ListNode* head) {
        auto dummy = new ListNode(-1);
        dummy->next = head;

        auto p = dummy;
        while (p->next) {
            auto q = p->next;
            while (q && p->next->val == q->val) q = q->next;

            if (p->next->next == q) p = p->next;
            else p->next = q;
        }

        return dummy->next;
    }
```



##### 倒数第k个节点

```
    ListNode* findKthToTail(ListNode* head, int k) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;
        if (n < k) return nullptr;
        auto p = head;
        for (int i = 0; i < n - k; i ++ ) p = p->next;
        return p;
    }
```



##### 反转链表

```
ListNode* reverseList(ListNode* head) {
    ListNode* pre = NULL;
    ListNode* cur = head;
    while (cur) {
        ListNode* next = cur->next;
        cur->next = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}
```

##### 合并两个有序单链表

```
ListNode* merge(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(-1);
    ListNode* cur = dummy;
    while (l1 && l2) {
        if (l1->val <= l2 -> val) {
            cur->next = l1;
            cur = cur->next;
            l1 = l1->next;
        } else {
            cur->next = l2;
            cur = cur->next;
            l2 = l2->next;
        }
    }
    cur->next = (l1 == NULL ? l2 : l1);
    return dummy->next;
}  
```

##### 链表归并排序



##### 两个链表的第一个公共节点

假设公共部分长度为c，两个链表同时走a+b+c步，a + c + b = b + c + a，a走到头就转向b， b走到头转向a，则在公共部分相遇

```
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *p = headA, *q = headB;
    while (p != q) {
        if (p) p = p->next;
        else p = headB;
        if (q) q = q->next;
        else q = headA;
    }
    return p;
}
```

##### 链表环的入口

两指针一快一满，快指针以两倍速度行走，必定相遇在环内

下图x+y必定为环的长度的整数倍，因为2(x+y) = x + y + nlen

此时慢指针回到开头 两指针同时重新走x步均回到b点，即环的入口

![image-20201019130354546](/Users/jingy/Library/Application Support/typora-user-images/image-20201019130354546.png)

```
    ListNode *entryNodeOfLoop(ListNode *head) {
        if (!head || !head->next) return 0;
        ListNode *first = head, *second = head;

        while (first && second)
        {
            first = first->next;
            second = second->next;
            if (second) 
            	second = second->next;
            else 
            	return NULL; //没有环

            if (first == second)
            {
                first = head;
                while (first != second)
                {
                    first = first->next;
                    second = second->next;
                }
                return first;
            }
        }

        return 0;
    }
```

##### 复杂链表的复制

带random指针的listNode节点的复制

使用哈希表保存random指针的原节点和复制节点对应关系

在原链表上穿叉复制节点

