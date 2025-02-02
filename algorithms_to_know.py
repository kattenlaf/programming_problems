from typing import List
import math

def binary_search(nums: List[int], target) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = math.floor((right + left) / 2)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

print(binary_search([5, 7, 7, 8, 9, 10], 10))

# [1, 2, 3] [5, 10, 20] 5
def knapsack_0_1(weights: List[int], values: List[int], maxweight: int) -> int:
    N = len(weights)
    W = maxweight
    dp = [[0 for _ in range(W + 1)] for _ in range(N + 1)]

    for i in range(N + 1):
        for w in range(maxweight + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            # This means we can include the weight
            elif weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            # This means we cannot include the weight
            else:
                dp[i][w] = dp[i-1][w]

    return dp[N][W]



print(knapsack_0_1([1,2,3], [5, 10, 20], 5))