from bioptim import Mapping, BiMapping
import numpy as np

a = np.array((0, 0, 0, 10, -9))
b = np.array((10, 9))

b_from_a = Mapping([3, -4])
# to get b
b_from_a.map(a)

# equivalent to get b
A_ba = np.ndarray((len(b), len(a)))
A_ba[:, :] = 0
A_ba[0, 3] = 1
A_ba[1, 4] = -1
np.dot(A_ba, a)

#better
L = max(len(b), len(a))
bool_map = np.zeros(L)
bool_map[3] = 1
bool_map[4] = -1

A_ba_2 = np.zeros((L, L))
A_ba_2[bool_map > 0, bool_map > 0] = 1
A_ba_2[bool_map < 0, bool_map < 0] = -1
A_ba_2 = np.delete(A_ba_2, np.sum(A_ba_2, axis=1) == 0, 0)
np.dot(A_ba_2, a)

# Bimapping
Vec_map = BiMapping([None, None, None, 0, -1],[3, -4])
Vec_map.to_first.map(a)
Vec_map.to_second.map(b)

np.dot(A_ba, a)
np.dot(A_ba.T, b)
np.dot(A_ba_2.T, b)
