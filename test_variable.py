import numpy as np
from variable import Variable

data = np.array([1, 2])
x = Variable(data)
print(x.data)