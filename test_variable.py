import numpy as np
from basics import Variable, Square


def testVariable():
    data = np.array([1, 2])
    x = Variable(data)
    print(x.data)


def testSquare():
    data = np.array([1, 2])
    x = Variable(data)
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)


if __name__ == "__main__":
    testSquare()