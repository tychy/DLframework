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


def testbackward():
    data = np.array([1, 2])
    x = Variable(data)
    A = Square()
    B = Exp()
    a = A(x)
    y = B(a)
    y.grad = np.array([1, 2])
    a.grad = B.backward(y.grad)
    x.grad = A.backward(a.grad)

    print(x.grad)


def testconttection():
    data = np.array([1, 2])
    x = Variable(data)
    A = Square()
    B = Exp()
    a = A(x)
    y = B(a)
    assert y.creator == B
    assert y.creator.input == a
    assert y.creator.input.creator == A
    assert y.creator.input.creator.input == x


if __name__ == "__main__":
    testSquare()
    testExp()
    testconcat()
    testbackward()
    testconttection()