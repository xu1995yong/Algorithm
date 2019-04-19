## 描述：给出一个包含大小写字母的字符串。求出由这些字母构成的最长的回文串的长度是多少。	
```java
public int longestPalindrome(String s) {
    int retVal = 0;
    if (s == null | s.length() == 0) {
        return retVal;
    }
    int[] hash = new int[128];
    char[] ch = s.toCharArray();
    for (char i = 0; i < s.length(); i++) {
        int index = Integer.valueOf(ch[i]);
        hash[index]++;
    }
    // 是否有中心点
    boolean flag = false;

    for (char i = 0; i < hash.length; i++) {
        int count = hash[i];
        // 出现双数次
        if (count % 2 == 0) {
            retVal += count;
        } else {
            // 出现单数次
            retVal += (count - 1);
            flag = true;
        }
    }
    if (flag){
        retVal += 1;
    }
    return retVal;
}
```

## 给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。
```java
public String removeKdigits(String num, int k) {
    if (num.length() == 0)
        return "";
    if (num.length() <= k) {
        return "0";
    }
    while (k > 0 && num.length() != 0) {
        boolean flag = false;
        for (int i = 0; i < num.length() - 1; i++) {
            int a = Integer.valueOf(num.charAt(i));
            int b = Integer.valueOf(num.charAt(i + 1));
            if (a > b) {
                num = num.substring(0, i) + num.substring(i + 1, num.length());
                flag = true;
                break;
            }
        }
        if (!flag) {
            num = num.substring(0, num.length() - 1);

        }
        while (num.length() != 1 && num.charAt(0) == '0') {
            num = num.substring(1);
        }
        k--;
    }
    return num;
}
```



## 11. 盛最多水的容器

```java
public int maxArea(int[] height) {
//容器的盛水量取决于容器的底和容器较短的那条高。可见只有较短边会对盛水量造成影响，因此
//移动较短边的指针，并比较当前盛水量和当前最大盛水量。直至左右指针相遇。可以证明，如果
//移动较高的边，则盛水量只会变少；移动较低的边，则可以遍历到最大的情况。
    int left = 0;
    int right = height.length - 1;
    int max = 0;
    while (left < right) {
        max = Math.max(max, Math.min(height[left], height[right]) * (right - left));
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    return max;
}
```



## 43. 

```java
public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0")) {
        return "0";
    }
    char[] nums1 = num1.toCharArray();
    char[] nums2 = num2.toCharArray();
    int[] val = new int[nums1.length + nums2.length];

    for (int j = nums2.length - 1; j > -1; j--) {
        int v1 = nums2[j] - '0';
        for (int i = nums1.length - 1; i > -1; i--) {
            int v2 = nums1[i] - '0';
            long temp = v1 * v2;
            int k = i + j + 1;
            while (temp != 0) {
                val[k] += temp % 10;
                temp /= 10;
                k--;
            }
        }
    }
    StringBuffer sb = new StringBuffer();
    for (int i = val.length - 1; i > -1; i--) {
        if (val[i] > 9) {
            val[i - 1] += val[i] / 10;
            val[i] %= 10;
        }
    }

    for (int i = 0; i < val.length; i++) {
        if ((i == 0 && val[0] != 0) || i != 0) {
            sb.append(val[i]);
        }
    }
    return sb.toString();
}
```



 

## [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

```java
public boolean canJump(int[] nums) {
    int right = nums.length - 1;
    int i = right;
    while (i >= 0) {
        if (nums[i] >= right - i) {    //如果当前下标可以跳转到right，就更新right的值
            right = i;
        }
        i--;
    }
    if (right != 0) return false;    //当整个循环结束时，right没有到达0,说明不可抵达
    return true;
}
```



## 670. 最大交换

```java
public int maximumSwap(int num) {
    char[] digits = Integer.toString(num).toCharArray();
    if(digits.length == 1) return num;

    // 寻找不符合非递增顺序的分界线
    int split = 0;
    for (int i = 0; i < digits.length-1; i++){
        if (digits[i] < digits[i+1]){
            split = i+1;
            break;
        }
    }

    // 在分界线后面的部分寻找最大的值max
    char max = digits[split];
    int index1 = split;
    for (int j = split+1; j < digits.length; j++){
        if (digits[j] >= max){
            max = digits[j];
            index1 = j;
        }
    }

    // 在分界线前面的部分向前寻找小于max的最大值
    int index2 = split;
    for (int i = split-1; i >= 0; i--){
        if (digits[i] >= max){
            break;
        }
        index2--;
    }

    //交换两位找到的char
    char temp = digits[index1];
    digits[index1] = digits[index2];
    digits[index2] = temp;

    return Integer.valueOf(new String(digits));
}
```

