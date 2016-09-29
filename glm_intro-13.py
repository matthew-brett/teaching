berkeley_indicator = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
stanford_indicator = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
mit_indicator      = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
X = np.column_stack((berkeley_indicator,
                    stanford_indicator,
                    mit_indicator))
X
# array([[1, 0, 0],
# [1, 0, 0],
# [1, 0, 0],
# [1, 0, 0],
# [0, 1, 0],
# [0, 1, 0],
# [0, 1, 0],
# [0, 1, 0],
# [0, 0, 1],
# [0, 0, 1],
# [0, 0, 1],
# [0, 0, 1]])
