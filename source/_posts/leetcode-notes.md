---
title: Leetcode笔记
date: 2019-11-15 12:08:15
tags: Leetcode
---

##### LC23 合并k个有序链表

1. 分治法
2. 暴力k个指向k个链表头的指针找最小值O(KN) -> 维护k个元素的最小堆 O(nlgk)
最小堆自定义比较函数
```C++
struct Cmp {
  bool operator() (ListNode* a, ListNode* b) {
    return a->val > b->val;
  }
}
注意判读p->next不为空再加入优先队列
```



##### LC 41 缺失的第一个正整数

下标置换
让n出现在nums[n-1]的位置上，0和负数忽略，注意while循环的判断一定要思考循环是否能有效终止，第二种情况如果交换的两个数相同，则while循环会变为死循环

```C++
while (nums[i] > 0 && mums[i] <= n && nums[i] != nums[nums[i] - 1]) {
  swap(nums[i], nums[nums[i] - 1]);
}
// or
 while (nums[i] >= 0 && nums[i] < n && nums[i] != i && nums[i] != nums[nums[i]])
  swap(nums[i], nums[nums[i]]);
```
或者如果不想考虑下标和正整数元素的差1，可以先将所有元素值减1，负数不处理，但此时要注意INT_MIN不能减。



- L2 链表两数相加

记得处理最后进位>0

- L7 整数反转
- L9 回文数

注意INT溢出情况的处理



- 字符串的最大公因子

长度满足最大公约数，暴力检查a和b串或者判断a+b == b+a



- 多数元素

分治

随机化



- 最长上升子序列优化



最长上升子序列的个数



- 最长上升连续子序列



- 最长连续序列

排序

暴力枚举+哈希

并查集

- 划分和相等的k个子集

  

- 分发糖果

![左-右&右-左两次贪心](https://tva1.sinaimg.cn/large/00831rSTly1gcuq9u9ieij311o0kwq6k.jpg)





- 矩阵中的增长路径

动态规划和拓扑排序的关系



- 不同的子序列

  





