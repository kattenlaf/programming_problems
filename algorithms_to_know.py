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