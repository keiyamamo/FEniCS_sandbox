import numpy as np

a = [0, 2, 1, 3]
b = [3, 2, 0, 1]

# I can not directly use sort on a and b 
# somehow I get the map index for a and b
map_index_a = [0, 2, 1, 3]
map_index_b = [2, 3, 1, 0]

a = np.array(a)
b = np.array(b)

a = a[map_index_a]
b = b[map_index_b]

print(a)
print(b)

# Basically, I have many a, b vectors and want to sort all of them

