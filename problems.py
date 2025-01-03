#You are given an m x n binary matrix grid. An island is a group of 1('s (representing land) connected 4-directionally (horizontal or vertical.) '
#'You may assume all four edges of the grid are surrounded by water.)

# The area of an island is the number of cells with a value 1 in the island.

# Return the maximum area of an island in grid. If there is no island, return 0.
import math

from typing import List
from typing import Optional
from collections import defaultdict, deque

def solution(arr):
    i,j = 0, 0
    islands = set()

    def exploreIsland(arr, x0, y0, i, j, v):
        arr[i][j] = 2
        v.append((i - x0, j - y0))

        if i > 0 and arr[i-1][j] == 1:
            exploreIsland(arr, x0, y0, i-1, j, v)
        if i < len(arr) - 1 and arr[i+1][j] == 1:
            exploreIsland(arr, x0, y0, i+1, j, v)
        if j > 0 and arr[i][j-1] == 1:
            exploreIsland(arr, x0, y0, i, j-1, v)
        if j < len(arr[i]) - 1 and arr[i][j+1]:
            exploreIsland(arr, x0, y0, i, j+1, v)

    count = 0
    while i < len(arr):
        j = 0
        while j < len(arr[i]):
            if arr[i][j] == 1:
                v = []
                exploreIsland(arr, i, j, i, j, v)
                islands.add(tuple(v))
            j += 1
        i += 1

    return len(islands)

input_list = [
        [1, 1, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 1]
        ]


class Solution:
    def longestPalindrome(self, s: str) -> str:
        solution = ""
        solutionLen = 0

        for i in range(len(s)):
            left, right = i, i
            while left >= 0 and right < len(s) and s[left] == s[right]:
                if len(s[left:right + 1]) > solutionLen:
                    solution = s[left:right + 1]
                    solutionLen = len(solution)

                left -= 1
                right += 1

            # evaluate where string is even length
            left, right = i, i + 1
            while left >= 0 and right < len(s) and s[left] == s[right]:
                if len(s[left:right + 1]) > solutionLen:
                    solution = s[left:right + 1]
                    solutionLen = len(solution)

                left -= 1
                right += 1

        return solution

    def constructRectangle(area: int) -> List[int]:
        # get square root to determine starting point to iterate backwards
        # because we know, closest LxW can be is when W = L ie rectangle is a square
        # iterate backwards from root area, and return on the first scenario of an answer which will be where L and W are closest
        root = int(area ** 0.5)

        for x in range(root, 0, -1):
            if area % x == 0:
                return [area // x, x]

    def findTargetSumWays(nums: List[int], target: int) -> int:
        lookup = {}

        def backtrack(index, current_sum):
            if (index, current_sum) in lookup:
                return lookup[(index, current_sum)]

            if index == len(nums):
                return 1 if current_sum == target else 0
            lookup[(index, current_sum)] = (
                    backtrack(index + 1, current_sum + nums[index]) +
                    backtrack(index + 1, current_sum - nums[index])
            )
            return lookup[(index, current_sum)]

        return backtrack(0, 0)

    def findTargetSumWays2(nums: List[int], target: int) -> int:
        dp = [defaultdict(int) for _ in range(len(nums) + 1)]

        dp[0][0] = 1  # (0 only used 0 elements, current sum is 0 and there is one way to do it)

        for i in range(len(nums)):
            for cur_sum, count in dp[i].items():
                dp[i + 1][cur_sum + nums[i]] += count
                dp[i + 1][cur_sum - nums[i]] += count

        return dp[len(nums)][target]

    def findTargetSumWays3(nums: List[int], target: int) -> int:
        dp = defaultdict(int)

        dp[0] = 1  # (0 only used 0 elements, current sum is 0 and there is one way to do it)

        for i in range(len(nums)):
            new_dp = defaultdict(int)
            for cur_sum, count in dp.items():
                new_dp[cur_sum + nums[i]] += count
                new_dp[cur_sum - nums[i]] += count
            dp = new_dp

        return dp[target]

    def convert(self, s: str, numRows: int) -> str:
        row = 0
        step = 1
        solutionList = [""] * numRows
        idx = 0
        if (numRows == 1):
            return s
        while idx < len(s):
            solutionList[row] += s[idx]
            row += step
            if (row == numRows-1):
                step = -1
            elif (row == 0):
                step = 1

            idx += 1

        solution = ""
        for string in solutionList:
            solution += string

        return solution

    def longestSubstringSize(self, s: str):
        char_pos = {}
        l, r = 0, 0
        sol = 0
        for r in range(len(s)):
            if s[r] in char_pos and char_pos[s[r]] >= l:
                l = char_pos[s[r]] + 1
            char_pos[s[r]] = r

            sol = max(sol, r - l + 1)

        return sol

    def reverse(self, x: int) -> int:
        num = []
        isNegative = True if x < 0 else False
        x = abs(x)
        while x != 0:
            num.append(x % 10)
            x //= 10

        position = 0
        sol = 0
        for i in range(len(num) - 1, -1, -1):
            sol += num[i] * (10 ** position)
            position += 1

        if sol > (2 ** 31) - 1:
            return 0

        if isNegative:
            sol *= -1
        return sol

    def myAtoi(self, string: str) -> int:
        string = string.strip(" ") # Remove leading white space
        num = []
        valid_nums = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
        idx = 0
        negative = False
        if (string[idx] == "-"):
            negative = True
            idx = 1
        elif (string[idx] == "+"):
            idx = 1

        # iterate while the current position in the string is a number and append to list
        while idx < len(string) and string[idx] in valid_nums:
            if len(num) == 0 and string[idx] == "0":
                # Skip when list empty and the current digit is 0
                idx += 1
            else:
                num.append(string[idx])
                idx += 1

        position = 0
        sol = 0
        for pos in range(len(num) -1, -1, -1):
            sol += int(num[pos]) * (10 ** position)
            position += 1

        if sol > (2 ** 31) - 1:
            if negative:
                sol = (2 ** 31)
            else:
                sol = (2 ** 31) - 1

        if negative:
            sol *= -1
        return sol

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        i = 0
        nums = sorted(nums)
        solution_arr = []
        # leave space for possible j and k values
        while i < len(nums) - 2:
            j, k = i + 1, len(nums) - 1
            current_sum = 0
            while j < k:
                current_sum = nums[i] + nums[j] + nums[k]
                if current_sum == 0:
                    solution_arr.append([nums[i], nums[j], nums[k]])
                if current_sum <= 0:
                    j_temp = nums[j]
                    while j < k and nums[j] == j_temp:
                        j += 1
                else:
                    k_temp = nums[k]
                    while k > j and nums[k] == k_temp:
                        k -= 1

            # Assuming all the current sums have been found for nums[i], now we adjust nums[i]
            # to be a value we have not used already
            i_temp = nums[i]
            while i < len(nums) - 2 and nums[i] == i_temp:
                i += 1

        return solution_arr

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        min_sum = math.inf
        nums = sorted(nums)
        i = 0
        while i < len(nums) - 2:
            j, k = i + 1, len(nums) - 1
            while j < k:
                current_sum = nums[i] + nums[j] + nums[k]
                min_sum = current_sum if abs(target - current_sum) < abs(target - min_sum) else min_sum
                if current_sum == target:
                    return current_sum
                elif current_sum < target:
                    j_temp = nums[j]
                    while j < k and nums[j] == j_temp:
                        j += 1
                else:
                    k_temp = nums[k]
                    while k > j and nums[k] == k_temp:
                        k -= 1

            i_temp = nums[i]
            while i < len(nums) - 2 and nums[i] == i_temp:
                i += 1

        return min_sum

    # Time complexity O(n * m) = n being the target and m being the number of coins
    def min_coin_change(self, coins: List[int], target: int) -> int:
        dp = [(target+1) for i in range(target+1)] # similar to [target+1] * (target+1)

        # base case
        dp[0] = 0
        for current_value in range(1, target + 1):
            for coin in coins:
                if current_value - coin >= 0:
                    # building dp table bottom up, min value for current total is minimum of what's currently stored
                    # and what's stored in dp for the total of this total - current coin
                    # 1 indicates the current coin we are using
                    dp[current_value] = min(dp[current_value], 1 + dp[current_value - coin])

        # This would indicate a minimum could not be found
        if dp[target] == target + 1:
            return -1

        return dp[target]

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        n = len(days)
        left7 = 0
        left30 = 0
        dp = [0] * n

        for right in range(n):
            while days[right] - days[left7] >= 7:
                left7 += 1
            while days[right] - days[left30] >= 30:
                left30 += 1

            cost1 = (dp[right - 1] if right > 0 else 0) + costs[0]
            cost7 = (dp[left7 - 1] if left7 > 0 else 0) + costs[1]
            cost30 = (dp[left30 - 1] if left30 > 0 else 0) + costs[2]

            dp[right] = min(cost1, cost7, cost30)

        print(dp)

        return dp[n - 1]

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        i = 0
        seen_pos = {}
        while i < len(nums):
            if nums[i] not in seen_pos:
                new_set = set()
                new_set.add(i)
                seen_pos[nums[i]] = new_set
            else:
                for num in seen_pos[nums[i]]:
                    if num <= k:
                        return True
                seen_pos[nums[i]].add(i)

            i += 1

        return False

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        print(visited)

        sol = []
        nums.sort()

        i = 0
        while i < len(nums) - 3:
            j = i + 1
            while j < len(nums) - 2:
                k, l = j + 1, len(nums) - 1
                while k < l:
                    current_sum = nums[i] + nums[j] + nums[k] + nums[l]
                    if current_sum == target:
                        sol.append([nums[i], nums[j], nums[k], nums[l]])
                        k += 1
                        l -= 1
                        # Removing any duplicates
                        while k < l and nums[k] == nums[k - 1]:
                            k += 1
                        while l > k and nums[l] == nums[l + 1]:
                            l -= 1
                    elif current_sum < target:
                        k += 1
                    else:
                        l -= 1

                # Exclude duplications
                j += 1
                while j < len(nums) - 2 and nums[j] == nums[j - 1]:
                    j += 1

            # Exclude duplications
            i += 1
            while i < len(nums) - 3 and nums[i] == nums[i - 1]:
                i += 1

        return sol

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def swap_nodes(current_node):
            if current_node == None:
                return
            temp = current_node.left
            current_node.left = current_node.right
            current_node.right = temp
            swap_nodes(current_node.left)
            swap_nodes(current_node.right)

        swap_nodes(root)
        return root

    def invertTreeIterative(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root == None:
            return None

        q = deque([root])
        while q:
            current_node = q.popleft()
            current_node.left, current_node.right = current_node.right, current_node.left

            if current_node.left != None:
                q.append(current_node.left)

            if current_node.right != None:
                q.append(current_node.right)


        return root


solutions = Solution()

nums = ['XXXX' , 'XX' , 'XXXXXX']
visited = []
for i in range(len(nums)):
    visited.append([False] * len(nums[i]))

print(visited)
# solutions.reverse(1000000003)
# print(solutions.fourSum([1,0,-1,0,-2,2], 0))
# print(solutions.threeSum([0,0,0]))
# print(solutions.threeSumClosest([4,0,5,-5,3,3,0,-4,-5], -2))
# print(solutions.convert("PAYPALISHIRING", 3))