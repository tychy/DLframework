import numpy as np
from dezero import Variable, Square, Exp, square, exp, as_array


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
    y.grad = np.array([1, 1])
    a.grad = B.backward(y.grad)
    x.grad = A.backward(a.grad)

    return x.grad


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


def testautobackward():
    data = np.array([1, 2])
    x = Variable(data)
    A = Square()
    B = Exp()
    a = A(x)
    y = B(a)
    y.backward()
    return x.grad


def test_wrapper():
    x = Variable(np.array([1, 2]))
    y = exp(square(x))
    y.backward()
    return x.grad


def test_grad():
    assert np.array_equal(testbackward(), testautobackward())
    assert np.array_equal(testbackward(), test_wrapper())


def test_asarray():
    a = np.array(1)
    b = a ** 2
    assert np.isscalar(b)
    assert not np.isscalar(as_array(b))


if __name__ == "__main__":
    # testSquare()
    # testExp()
    # testconcat()
    testconttection()
    test_grad()
    test_asarray()
