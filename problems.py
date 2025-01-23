import math
from typing import List
from typing import Optional
from collections import defaultdict, deque
import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

"""
You are given an m x n binary matrix grid. An island is a group of 1('s (representing land) connected 4-directionally (horizontal or vertical.) '
You may assume all four edges of the grid are surrounded by water.)
The area of an island is the number of cells with a value 1 on the island.
Return the maximum area of an island in grid. If there is no island, return 0. 
"""
def solution_exploreIslands(arr):
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

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
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

    def findTargetSumWays2(self, nums: List[int], target: int) -> int:
        dp = [defaultdict(int) for _ in range(len(nums) + 1)]

        dp[0][0] = 1  # (0 only used 0 elements, current sum is 0 and there is one way to do it)

        for i in range(len(nums)):
            for cur_sum, count in dp[i].items():
                dp[i + 1][cur_sum + nums[i]] += count
                dp[i + 1][cur_sum - nums[i]] += count

        return dp[len(nums)][target]

    def findTargetSumWays3(self, nums: List[int], target: int) -> int:
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

    def intToRoman(self, num: int) -> str:
        solution = []
        places = [["I", "V", "X"], ["X", "L", "C"], ["C", "D", "M"], ["M"]]
        places_pos = 0
        while num != 0:
            current_str = ""
            digit = num % 10
            num //= 10
            if digit == 4 or digit == 9:
                # handle differently
                current_str += places[places_pos][0]
                current_str += places[places_pos][1] if digit == 4 else places[places_pos][2]
            else:
                # the digit is not a 4 or 9, check if greater than 5, if yes we append the 5 value at that places
                # position first, then add the number of 10s place values
                if digit >= 5:
                    digit -= 5
                    current_str += places[places_pos][1]
                for j in range(digit):
                    current_str += places[places_pos][0]

            solution.append(current_str)
            places_pos += 1

        solution_string = ""
        k = len(solution) - 1
        while k >= 0:
            solution_string += solution[k]
            k -= 1

        return solution_string

            # with the digit map it to the value we expect to append to the string

    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []

        num_possible_combinations = 1
        i = 0
        while i < len(digits):
            # if it is a 7 or 9 we multiply by 4, else 3
            if digits[i] == "7" or digits[i] == "9":
                num_possible_combinations *= 4
            else:
                num_possible_combinations *= 3
            i += 1

        answer = ["" for p in range(num_possible_combinations)]
        numpad = {"2": "abc", "3": "def",
                  "4": "ghi", "5": "jkl", "6": "mno",
                  "7": "pqrs", "8": "tuv", "9": "wxyz"}

        i = 0
        # Check each digit
        while i < len(digits):
            current_keys = numpad[digits[i]]
            num_possible_combinations //= len(current_keys)
            j = 0
            while j < len(answer):
                # Place the expected key in solution
                for k in range(len(current_keys)):
                    for x in range(num_possible_combinations):
                        answer[j] += current_keys[k]
                        j += 1
            i += 1

        return answer

    def divide(self, dividend: int, divisor: int) -> int:
        if divisor == 0:
            return 2**31 - 1

        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1

        sign = 1
        if dividend < 0:
            dividend = -dividend
            sign = -sign
        if divisor < 0:
            divisor = -divisor
            sign = -sign

        multiple = 1
        while dividend >= (divisor << 1):
            divisor <<= 1
            multiple <<= 1

        quotient = 0
        while multiple > 0:
            if dividend >= divisor:
                dividend -= divisor
                quotient += multiple
            divisor >>= 1
            multiple >>= 1

        return sign * quotient

    def stringMatching(self, words: List[str]) -> List[str]:
        solution = []
        word_string = " ".join(words)
        for word in words:
            # if the number of substrings is at least 2 we count it
            # 2 because a word is a substring of itself
            if word_string.count(word) >= 2:
                solution.append(word)
        return solution

    def checkIfExist(self, arr: List[int]) -> bool:
        lookup = defaultdict(list)
        i = 0
        for i in range(len(arr)):
            lookup[arr[i]].append(i)

        for i in range(len(arr)):
            if (arr[i] * 2) in lookup:
                for idx in lookup[arr[i]]:
                    if idx != i:
                        return True

        return False

    def searchRange(self, nums: List[int], target: int) -> List[int]:

        def binarySearch(nums: List[int], target: int, l: int, r: int) -> int:
            while l <= r:
                mid = (r + l) // 2
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1

            return -1

        known_pos = binarySearch(nums, target, 0, len(nums) - 1)
        start_idx = known_pos
        end_idx = known_pos
        # if the element exists in the array, keep binary searching sub arrays until final element is found
        if start_idx != -1:
            new_idx = 0
            while new_idx != -1 and start_idx > 0:
                # binary Search left subarray, keep going left until can't find element
                new_idx = binarySearch(nums, target, 0, start_idx - 1)
                if new_idx == start_idx:
                    new_idx = -1
                if new_idx != -1:
                    start_idx = new_idx
            new_idx = 0
            while new_idx != -1 and end_idx < len(nums) - 1:
                new_idx = binarySearch(nums, target, end_idx+1, len(nums) - 1)
                # This means no more to find
                if new_idx == end_idx:
                    new_idx = -1
                if new_idx != -1:
                    end_idx = new_idx

        ret = []
        ret.append(start_idx)
        ret.append(end_idx)
        return ret

    def fizzbuzz(self, target):
        for i in range(1, target + 1):
            prompt = ""
            if i % 3 == 0:
                prompt += "fizz"
            if i % 5 == 0:
                prompt += "buzz"
            if prompt == "":
                prompt = i
            print(prompt)

    def fizzbuzzTwo(self, target):
        for i in range(1, 101):
            print("Fizz" * (i%3 == 0) + (i%5 == 0) * "Buzz" or i)

    def findAllValidParenthesis(self, n:int) -> List[int]:
        sol = []
        count_open = count_close = 0
        def dfs(ct_open, ct_close, current_string):
            # this means we have built a full string with the total number of parenthesis' available
            # validate and append to the list
            if ct_open == ct_close and ct_open + ct_close == (n * 2):
                sol.append(current_string)
                return

            if ct_open < n:
                dfs(ct_open + 1, ct_close, current_string + "(")
            # Can only add a closing parenthesis if the total number of opened ones are less than it
            if ct_close < ct_open:
                dfs(ct_open, ct_close + 1, current_string + ")")

        dfs(count_open,count_close,"")
        return sol

    def maxSubArray(self, nums: List[int]) -> int:
        answer = nums[0]
        current_sum = 0
        i = 0
        while i < len(nums):
            # if it is the first number, we include in the subarray
            # or if num[i] is > current sum plus num[i]
            if i == 0 or nums[i] > current_sum + nums[i]:
                # restart subarray basically
                current_sum = nums[i]
            else:
                current_sum += nums[i]

            answer = max(answer, current_sum)
            i += 1

        return answer

    def maxSubArrayDivideAndConquer(self, nums: List[int]) -> int:
        def maxSubArray(nums):
            if len(nums) == 1:
                return nums[0]

            mid = len(nums) // 2
            left_max = maxSubArray(nums[:mid])
            right_max = maxSubArray(nums[mid:])

            # Find the maximum sum subarray that crosses the middle element
            left_sum = float('-inf')
            sum = 0
            for i in range(mid - 1, -1, -1):
                sum += nums[i]
                left_sum = max(left_sum, sum)

            right_sum = float('-inf')
            sum = 0
            for i in range(mid, len(nums)):
                sum += nums[i]
                right_sum = max(right_sum, sum)

            cross_max = left_sum + right_sum

            return max(left_max, right_max, cross_max)

        return maxSubArray(nums)

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if len(lists) == 0:
            return None

        while len(lists) > 1:
            mergedLists = []
            # Merge each list in pairs
            for i in range(0, len(lists), 2):
                list1 = lists[i]
                list2 = lists[i+1] if (i+1) < len(lists) else None
                mergedList = self.mergeLinkedLists(list1, list2)
                mergedLists.append(mergedList)
            lists = mergedLists

        return lists[0]

    def mergeLinkedLists(self, list1, list2):
        dummy_node = ListNode()
        current_node = dummy_node
        while list1 and list2:
            if list1.val < list2.val:
                current_node.next = list1
                list1 = list1.next
            else:
                current_node.next = list2
                list2 = list2.next
            current_node = current_node.next

        if list1:
            current_node.next = list1
        if list2:
            current_node.next = list2

        return dummy_node.next

    def mergeKListsHeap(self, lists: Optional[ListNode]) -> Optional[ListNode]:
        minHeap = []
        for i, node in enumerate(lists):
            if node is not None:
                heapq.heappush(minHeap, (node.val, i, node))

        dummy = current = ListNode()
        while minHeap:
            # Pop the smallest item off the heap
            val, i, node = heapq.heappop(minHeap)
            current.next = node
            current = current.next
            if node.next is not None:
                heapq.heappush(minHeap, (node.next.val, i, node.next))

        return dummy.next

    def minimumLength(self, s: str) -> int:
        counter = defaultdict(int)
        i = 0
        length = 0
        while i < len(s):
            counter[s[i]] += 1
            i += 1
        for key in counter:
            num_characters = counter[key]
            if num_characters >= 3:
                if num_characters % 2 == 0:
                    length += 2
                else:
                    length += 1
            else:
                length += num_characters

        return length

    def makeGood(self, s: str) -> str:
        i = 0
        str_len = len(s)
        stack = []
        while i < str_len:
            if stack and stack[-1] != s[i]:
                # if it is a bad pair
                if stack[-1] == s[i].lower() or stack[-1] == s[i].upper():
                    stack.pop()
                    first_half = s[:i-1]
                    second_half = s[i+1:] if (i + 1) < str_len else ""
                    s = first_half + second_half
                    i -= 2
                    str_len -= 2
                else:
                    stack.append(s[i])
            else:
                stack.append(s[i])
            i += 1
        return s

    def canConstruct(self, ransomNote: str, magazine: str):
        # optimize
        if len(ransomNote) > len(magazine):
            return False

        mag_char_count = defaultdict(int)
        i = 0
        while i < len(magazine):
            mag_char_count[magazine[i]] += 1
            i += 1

        i = 0
        while i < len(ransomNote):
            if mag_char_count[ransomNote[i]] == 0:
                return False
            else:
                mag_char_count[ransomNote[i]] -= 1
            i += 1
        return True

    def reverseVowels(self, s: str) -> str:
        reversed_string = ["" for i in range(len(s))]
        for i in range(len(s)):
            reversed_string[i] = s[i]

        l, r = 0, len(s) - 1
        vowels = set(["a", "e", "i", "o", "u"])
        while l < r:
            if reversed_string[l].lower() in vowels:
                # iterate r to the first vowel while also staying > l
                while r > l:
                    if reversed_string[r].lower() in vowels:
                        # do the swap
                        reversed_string[l], reversed_string[r] = reversed_string[r], reversed_string[l]
                        l += 1
                        r -= 1
                        break
                    else:
                        r -= 1
            else:
                l += 1
        return "".join(reversed_string)

    def findMaxAverage(self, nums: List[int], k: int) -> float:
        i = j = 0
        current_max = 0
        for j in range(k):
            current_max += nums[j]
        j = k
        i += 1
        current_sum = current_max
        while j < len(nums):
            current_sum = current_sum - nums[i-1] + nums[j]
            current_max = max(current_max, current_sum)
            j += 1
            i += 1

        return current_max / k

    def findKthLargest(self, nums: List[int], k: int) -> int:
        maxHeap = []
        for num in nums:
            heapq.heappush(maxHeap, -num)

        answer = 0
        for i in range(k):
            answer = heapq.heappop(maxHeap)

        return -answer

    def thirdMax(self, nums: List[int]) -> int:
        maximums = [float('-inf') for i in range(3)]
        maximums[0] = nums[0]
        for i in range(len(nums)):
            if nums[i] > maximums[0]:
                maximums[0], maximums[1], maximums[2] = nums[i], maximums[0], maximums[1]
            elif nums[i] < maximums[0]:
                if maximums[1] == "" or nums[i] > maximums[1]:
                    maximums[1], maximums[2] = nums[i], maximums[1]
                elif nums[i] < maximums[1]:
                    if maximums[2] == "" or nums[i] > maximums[2]:
                        maximums[2] = nums[i]
        return maximums[2] if maximums[2] != "" else maximums[0]

    def removeElement(self, nums: List[int], val: int) -> int:
        j = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[j] = nums[i]
                j += 1

        return j

    def wordPattern(self, pattern: str, s: str) -> bool:
        s_list = s.split(" ")
        hashmap = {}
        hashset = set()
        if len(s) != len(s_list):
            return False
        for i in range(len(pattern)):
            if pattern[i] not in hashmap:
                # If the current letter has not been seen but the word has return false, that means multiple mappings
                if s_list[i] in hashset:
                    return False
                hashmap[pattern[i]] = s_list[i]
                hashset.add(s_list[i])
            else:
                if hashmap[pattern[i]] != s_list[i]:
                    return False

        return True

    def findPeakElement(self, nums: List[int]) -> int:
        # can solve using binary search
        l, r = 0, len(nums) - 1
        while l < r:
            # if mid is > index to the right, we know peak will be over there otherwise peak will be to the left
            #O(logn)
            mid = (r + l) // 2
            if nums[mid] > nums[mid + 1]:
                r = mid
            else:
                l = mid + 1

        return l

    def is_pangram(self, st):
        solution = [False for i in range(26)]
        for i in range(len(st)):
            position = ord(st[i].lower()) - ord("a")
            if position >= 0 and position < 26:
                solution[position] = True
        for i in range(len(solution)):
            if solution[i] == False:
                return False
        return True

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        solution = [1] * len(nums)
        left = 1
        # iterate upwards, storing the product of the entire array on left side of nums[i] in solution [i] position
        for i in range(len(nums)):
            solution[i] = left
            left *= nums[i]
        right = 1
        for i in range(len(nums) - 1, -1, -1):
            solution[i] *= right
            right *= nums[i]
        return solution

    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            # Find the position in the array where nums[i] > nums[i+1] if it exists
            i -= 1

        # i will be -1 if there is no permutation after this current list and we will just reverse the list at the end
        if i >= 0:
            j = len(nums) - 1

            # find the number to swap position i with, which will be the next greatest number in the list, nums[i] will
            # take nums[j] place to uphold the rule of lexicographical order
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]

        nums[i + 1:] = reversed(nums[i + 1:])

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        heap = []
        counter = defaultdict(int)
        # increment count, of each element
        for n in nums:
            counter[n] += 1

        # to know the top k elements, put the count of each element on a max heap
        for key, count in counter.items():
            heapq.heappush(heap, (-count, key))

        solution = []
        # pop off k elements to know the top k elements
        for i in range(k):
            elem = heapq.heappop(heap)
            solution.append(elem[1])

        return solution

    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (r + l) // 2
            if nums[mid] == target:
                return mid
            # if the middle number is > but nums[l] is also greater we want to binary search the right
            elif nums[mid] >= nums[l]:
                if nums[l] <= target <= nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] <= target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1


        return -1

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = defaultdict(set)
        cols = defaultdict(set)
        sub_boxes = defaultdict(set)

        # iterate over entire board,

        for row in range(9):
            for col in range(9):
                if board[row][col] == ".":
                    continue
                sb_r = row // 3
                sb_c = col // 3
                if board[row][col] in rows[row] or board[row][col] in cols[col] or board[row][col] in sub_boxes[
                    (sb_r, sb_c)]:
                    return False
                else:
                    rows[row].add(board[row][col])
                    cols[col].add(board[row][col])
                    sub_boxes[(sb_r, sb_c)].add(board[row][col])

        return True

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        answer = []
        def appendSolution(index, possible_candidates, remaining_target):
            if remaining_target == 0:
                answer.append(possible_candidates.copy())
                return
            if remaining_target < 0 or index == len(candidates):
                return

            # do dfs with current index
            possible_candidates(candidates[index])
            appendSolution(index + 1, possible_candidates, remaining_target - candidates[index])
            possible_candidates.pop()

            # skip the current candidate and any indexes that have the same value
            while (index+1) < len(candidates) and candidates[index] == candidates[index+1]:
                index += 1
            appendSolution(index+1, possible_candidates, remaining_target - candidates[index])

        appendSolution(0, [], target)

        return answer

    def orangesRotting(self, grid: List[List[int]]) -> int:
        # do a multi source bfs, do the following:
        # first start with all rotten oranges in the queue
        # then from each cell/node check the surrounding cells if they are fresh oranges
        # if fresh, set it to rotten and then append to new queue
        # ignore cell if it is already rotten or 0
        # increment minutes when queue is empty, update queue to have the new rotten oranges
        # and then continue, stop iterating when queue is empty
        # do final check of grid for any still fresh oranges,
        # if any is fresh return -1
        minutes = 0
        rotten = []
        fresh = set()
        # row
        for r in range(len(grid)):
            # col
            for c in range(len(grid[r])):
                if grid[r][c] == 2:
                    # tuple representing coordinate of rotten orange
                    rotten.append((r, c))
                if grid[r][c] == 1:
                    fresh.add((r, c))

        new_rotten = []

        def bfs(r, c, new_rotten):
            # if outside the grid return
            if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[r]):
                return

            if grid[r][c] == 1:
                grid[r][c] = 2
                fresh.remove((r, c))
                new_rotten.append((r, c))

        while rotten or new_rotten:
            node = rotten.pop()
            r, c = node[0], node[1]
            bfs(r - 1, c, new_rotten)
            bfs(r + 1, c, new_rotten)
            bfs(r, c - 1, new_rotten)
            bfs(r, c + 1, new_rotten)

            if len(rotten) == 0 and len(new_rotten) != 0:
                minutes += 1
                rotten = new_rotten
                new_rotten = []

        if len(fresh) > 0:
            return -1
        else:
            return minutes

    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"

        def convertToInt(num):
            places = len(num) - 1
            i = 0
            digit = 0
            result = 0
            while i < len(num):
                digit = ord(num[i]) - ord("0")
                result += digit * (10 ** places)
                places -= 1
                i += 1

            return result

        intNum1 = convertToInt(num1)
        intNum2 = convertToInt(num2)
        solution = intNum1 * intNum2
        return str(solution)

    def jump(self, nums: List[int]) -> int:
        start = end = num_jumps = 0
        while end < len(nums) - 1:
            farest = 0
            for i in range(start, end + 1):
                farest = max(farest, i + nums[i])

            start = end + 1
            end = farest
            num_jumps += 1

        return num_jumps


def buildList(nums):
    dummy = ListNode()
    current = dummy
    for num in nums:
        current.next = ListNode(num)
        current = current.next
    return dummy.next

def printList(list: Optional[ListNode]):
    current = list
    while current:
        print(current.val, end="->") if current.next is not None else print(current.val, end="")
        current = current.next

solutions = Solution()
# print(solutions.orangesRotting([[2,1,1],[1,1,0],[0,1,1]]))
print(solutions.jump([2,3,1,1,4]))

# print(solutions.minimumLength("ucvbutgkohgbcobqeyqwppbxqoynxeuuzouyvmydfhrprdbuzwqebwuiejoxsxdhbmuaiscalnteocghnlisxxawxgcjloevrdcj"))
# print(solutions.canConstruct("aa", "aab"))
# print(solutions.findMaxAverage([1,12,-5,-6,50,3], 4))
# print(solutions.findKthLargest([3,2,1,5,6,4], 2))
# (solutions.findPeakElement([1,2,1,3,5,6,4]))
# print(solutions.search([5,1,2,3,4], 1))

"""
list1 = buildList([1,4,5])
list2 = buildList([1,3,4])
list3 = buildList([2,6])
merged = solutions.mergeKListsHeap([list1, list2, list3])
printList(merged)
"""

