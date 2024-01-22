# For Problem 2
#
# we will implement the MDE algorithm in dynamic programming
#
# DO NOT modify any function definitions or return types,
# as we will use these to grade your work.
# However, feel free to add new functions to the file to avoid redundant code.
#


def med_naive(str1, str2):
    def med_naive_helper(str1, str2, m, n):
        if m == 0: 
            return n
        if n == 0: 
            return m
        if str1[m - 1] == str2[n - 1]: 
            return med_naive_helper(str1, str2, m - 1, n - 1)

        return min(
            1 + med_naive_helper(str1, str2, m, n - 1),  # Insert
            1 + med_naive_helper(str1, str2, m - 1, n),  # Remove
            2 + med_naive_helper(str1, str2, m - 1, n - 1)  # Replace
        )

    return med_naive_helper(str1, str2, len(str1), len(str2))


def med_dp_top_down(str1, str2):
    d = {}
    def med_dp_helper(str1, str2, m, n):
        key = m, n
        
        if m == 0: 
            return n
        if n == 0: 
            return m
        if key in d: 
            return d[key]
        if str1[m - 1] == str2[n - 1]: 
            return med_dp_helper(str1, str2, m-1, n-1)

        d[key] = min(med_dp_helper(str1, str2, m, n-1) + 1, med_dp_helper(str1, str2, m-1, n) + 1, med_dp_helper(str1, str2, m-1, n-1) + 2)
        return d[key]

    return med_dp_helper(str1, str2, len(str1), len(str2))
        


def med_dp_bottom_up(str1, str2):
    #######################################
    """
    Time Complexity = O(M.N)
    Space Complexity = O(M.N)

    len1 = len(str1)
    len2 = len(str2)

    DP = [[0 for _ in range(len2 + 1)] for __ in range(len1 + 1)]

    for _ in range(len1 + 1):
        for __ in range(len2 + 1):
           
            if _ == 0:
                DP[_][__] = __
            elif __ == 0:
                DP[_][__] = _
            elif str1[_-1] == str2[__-1]:
                DP[_][__] = DP[_-1][__-1]
            else:
                DP[_][__] = min((DP[_][__-1] + 1), (DP[_-1][__] + 1), (DP[_-1][__-1] + 2))
    
    return DP[len1][len2]
    """
    #######################################
    
    # Time Complexity = O(M.N)
    # Space Complexity = O(2N)

    len1 = len(str1)
    len2 = len(str2)

    DP = [[_ for _ in range(len2 + 1)],[0 for _ in range(len2 + 1)]]

    for _ in range(1, len1+1):
        DP[1][0] = _
        for __ in range(1, len2+1):
            if str1[_-1] == str2[__-1]:
                DP[1][__] = DP[0][__-1]
            else:
                DP[1][__] = min((1 + DP[0][__]), (1 + DP[1][__-1]), (2 + DP[0][__-1]))
        DP[0] = DP[1].copy()

    return DP[0][len2]


