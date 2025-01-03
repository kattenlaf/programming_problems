from collections import deque

def reachTheEnd(grid, maxTime):
    # Write your code here
    visited = []
    for i in range(len(grid)):
        visited.append([False] * len(grid[i]))

    # Use bfs to check from starting node 0,0 shortest distance to get to target node x,y
    # bfs
    que = deque()
    visited[0][0] = True
    # Append the list representing the coordinate
    que.append([0, 0])
    current_time = maxTime
    while que:
        time_taken = 0
        current_node = que.popleft()
        # check nodes adjacent to current_node, if valid add to the queue
        x, y = current_node[0], current_node[1]
        time_taken = abs(x) + abs(y)
        print(que)
        print(visited)

        # If x and y ever equal to the bottom right just return here
        if x == len(grid) - 1 and y == len(grid[x]) - 1:
            return 'Yes'

        if (maxTime - time_taken) >= 0:
            if x > 0:
                # check the top and if it is not a visited node
                if not visited[x - 1][y] and grid[x - 1][y] != '#':
                    que.append([x - 1, y])
                    visited[x - 1][y] = True

            if x < len(grid) - 1:
                # check the bottom if it is not a visited node
                if not visited[x + 1][y] and grid[x + 1][y] != "#":
                    print('foo')
                    que.append([x + 1, y])
                    visited[x + 1][y] = True

            if y > 0:
                # check the left and if it is not a visited node
                if not visited[x][y - 1] and grid[x][y - 1] != '#':
                    que.append([x, y - 1])
                    visited[x][y - 1] = True

            if y < len(grid[x]) - 1:
                # check the right and if it is not a visited node
                if not visited[x][y + 1] and grid[x][y + 1] != '#':
                    que.append([x, y + 1])
                    visited[x][y + 1] = True

    return 'No'


def journey(path, maxStep):
    # Write your code here
    # dynamic programming problem

    if len(path) == 1:
        return path[0]

    dp = []
    dp.append(path[0])
    for i in range(1, len(path)):
        bound = i - maxStep
        if bound < 0:
            bound = 0
        current_max = dp[i-1] + path[i]
        for step in range(i, bound, -1):
            current_max = max(current_max, dp[step-1] + path[i])
        dp.append(current_max)

    print(dp)

    return dp[len(path) - 1]



print(journey([10, 2, -10, 5, 20, 100, 20, -300, 2], 2))