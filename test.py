import matplotlib.cm as cm
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
])


# Yayma (broadcast) işlemi
color_exp = color[np.newaxis, :, :]     # (1, 3, 3)
b_exp = b[:, :, np.newaxis]     # (5, 3, 1)

# Eleman-eleman çarpım
result = a_exp * b_exp          # (5, 3, 3)

# Ortadaki eksen boyunca topla (axis=1)
summed_result = np.sum(result, axis=1)   # shape: (5, 3)

print(summed_result)