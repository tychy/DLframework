import numpy as np
from basics import Variable, Square, Exp


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


def testExp():
    data = np.array([1, 2])
    x = Variable(data)
    f = Exp()
    y = f(x)
    print(type(y))
    print(y.data)


def testconcat():
    data = np.array([1, 2])
    x = Variable(data)
    a = Square()
    b = Exp()
    y = a(b(x))
    print(type(y))
    print(y.data)


if __name__ == "__main__":
    testSquare()
    testExp()
    testconcat()