## 爬楼梯
```java
public int climbStairs(int n) {
	int[] dp = new int[n + 1];

	for (int i = 0; i <= n; i++) {
		if (i < 3) {
			dp[i] = i;
		} else {
			dp[i] = dp[i - 1] + dp[i - 2];
		}
	}
	return dp[n];
}
```

## 剪绳子

给你一根长度为n的绳子，请把绳子剪成m段 (m和n都是整数，n>1并且m>1)每段绳子的长度记为k[0],k[1],…,k[m].请问k[0]*k[1]*…*k[m]可能的最大乘积是多少？例如，当绳子的长度为8时，我们把它剪成长度分别为2,3,3的三段，此时得到的最大乘积是18.

```java
public int matProductAfterCutting(int length) {
    if (length < 2) {
        return 0;
    }
    if (length == 2) {
        return 1;
    }
    if (length == 3) {
        return 2;
    }
    // 将最优解存储在数组中
    int[] products = new int[length + 1];
    // 数组中第i个元素表示把长度为i的绳子剪成若干段之后的乘积的最大值
    products[0] = 0;
    products[1] = 1;
    products[2] = 2;
    products[3] = 3;

    for (int i = 4; i <= length; i++) {
        int max = 0;
        // 求出所有可能的f(j)*f(i-j)并比较出他们的最大值
        for (int j = 1; j <= i / 2; j++) {
            int product = products[j] * products[i - j];
            if (product > max) {
                max = product;
            }
            products[i] = max;
        }
    }
    return products[length];
}
```



## 等差数列划分

```java
public int numberOfArithmeticSlices(int[] num) {
    if (num == null || num.length <= 2) {
        return 0;
    }
    int[] dp = new int[num.length];
    int sum = 0;
    for (int i = 2; i < num.length; i++) {
        if (num[i] - num[i - 1] == num[i - 1] - num[i - 2]) {
            dp[i] = dp[i - 1] + 1;
            sum += dp[i];
        }
    }
    return sum;
}
```

## 分割等和子集

给定一个**只包含正整数**的**非空**数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```java
public boolean canPartition(int[] nums) {
    if (nums.length <= 1) return false;
    int sum = 0;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
    }
    if (sum % 2 != 0) return false;
    int capacity = sum / 2;
    //明确状态：dp[m][n] 考虑是否将第m个数字放入容量为n的背包
    boolean[][] dp = new boolean[nums.length][capacity + 1];
    //状态初始化
    for (int i = 0; i <= capacity; i++) {
        if (i != nums[0]) {
            dp[0][i] = false;
        } else {
            dp[0][i] = true;
        }
    }
    //状态转移方程：dp[m][n] = dp[m-1][n] || dp[m-1][n-nums[m]]
    for (int i = 1; i < nums.length; i++) {
        for (int j = 0; j <= capacity; j++) {
            dp[i][j] = dp[i - 1][j];
            if (nums[i] <= j) {
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
            }

        }
    }
    return dp[nums.length - 1][capacity];
}
```

## 买卖股票的最佳时机

```java
public int maxProfit(int[] prices) {
    int profit = 0;
    int min = Integer.MAX_VALUE;
    for (int i = 0; i < prices.length; i++) {
        min = prices[i] < min ? prices[i] : min;
        profit = prices[i] - min > profit ? prices[i] - min : profit;
    }
    return profit;
}
```
## 买卖股票的最佳时机Ⅱ
```java
public int maxProfit(int[] prices) {
    int profit = 0;
    for (int i = 0; i < prices.length - 1; i++) {
        int diff = prices[i+1] - prices[i];
        if (diff > 0) {
            profit += diff;
        }
    }
    return profit;
}
```

## 打劫房屋

假设你是一个专业的窃贼，准备沿着一条街打劫房屋。每个房子都存放着特定金额的钱。你面临的唯一约束条件是：相邻的房子装着相互联系的防盗系统，且 当相邻的两个房子同一天被打劫时，该系统会自动报警。

给定一个非负整数列表，表示每个房子中存放的钱， 算一算，如果今晚去打劫，你最多可以得到多少钱 在不触动报警装置的情况下。

```java	
public long houseRobber(int[] A) {
    if (A == null || A.length == 0) {
        return 0;
    } else if (A.length == 1) {
        return A[0];
    } else {
        long[] dp = new long[A.length];
        dp[0] = A[0];
        dp[1] = Math.max(dp[0], A[1]);
        for (int i = 2; i < A.length; i++) {
            dp[i] = Math.max(dp[i - 2] + A[i], dp[i - 1]);
        }
        return dp[A.length - 1];
    }
}
```

## 打劫房屋 II
在上次打劫完一条街道之后，窃贼又发现了一个新的可以打劫的地方，但这次所有的房子围成了一个圈，这就意味着第一间房子和最后一间房子是挨着的。每个房子都存放着特定金额的钱。你面临的唯一约束条件是：相邻的房子装着相互联系的防盗系统，且 当相邻的两个房子同一天被打劫时，该系统会自动报警。

给定一个非负整数列表，表示每个房子中存放的钱， 算一算，如果今晚去打劫，你最多可以得到多少钱 在不触动报警装置的情况下。
```java
public int houseRobber2(int[] nums) {
    if (nums == null || nums.length == 0)
        return 0;
    if (nums.length == 1)
        return nums[0];
    if (nums.length == 2)
        return Math.max(nums[0], nums[1]);
    if (nums.length == 3)
        return Math.max(Math.max(nums[0], nums[1]), nums[2]);
    int len = nums.length;
    int res1 = houseRobber(nums, 0, len - 2);
    int res2 = houseRobber(nums, 1, len - 1);
    return Math.max(res1, res2);
}

public int houseRobber(int[] A, int start, int end) {
    if (start == end) {
        return A[start];
    } else {
        int[] dp = new int[end - start + 1];
        dp[0] = A[start];
        dp[1] = Math.max(dp[0], A[start + 1]);
        for (int i = 2; i <= end - start; i++) {
            dp[i] = Math.max(dp[i - 2] + A[start + i], dp[i - 1]);
        }
        return dp[end - start];
    }
}
```

## 可以组成多少种二叉搜索树

给定一个整数 *n*，求以 1 ... *n* 为节点组成的二叉搜索树有多少种？

```java
/**
根据二叉搜索树的定义可以看出，根节点值不同，形成的二叉搜索树就不同，那么[1:n][1:n]范围内的n个数就有n个不同的选择。假设选取i作为根节点值，根据二叉搜索树的规则，[1:i−1][1:i−1]这i-1个数在其左子树上，[i+1:n][i+1:n]这n-i个数在其右子树上
对于由[1:i−1][1:i−1]形成的左子树，又可以采用上述方法进行分解
对于由[i+1:n][i+1:n]形成的右子树，同样可以采用上述方法进行分解
由于每个分解的范围都是连续递增的，所以无需考虑具体数值。另G(n)表示由连续的n个数形成的二叉搜索树的个数
那么G(n)为所求解，假设以i作为分界点，那么左子树为G(i-1)，右子树为G(n-i)
因为i可以取从1到n的任意一个数，所以G(n)=∑ni=1(G(i−1)∗G(n−i))G(n)=∑i=1n(G(i−1)∗G(n−i))
需要对G(0)和G(1)特殊处理，令其为1，即G(0)=G(1)=1
*/

public int numTrees(int n) {
    int[] dp = new int[n + 1];
    dp[0] = dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        for (int j = 1; j <= i; ++j) {
            dp[i] += dp[j - 1] * dp[i - j];  //G(i) += G(j - 1) * G(n - j)
        }
    }
    return dp[n];
}
```

## 动态规划题求解步骤

1. 确认原问题与子问题
2. 确认状态，即dp数组中存储的值的意义
3. 确认边界状态的值
4. 确定状态转移方程



