import spacejam as sj
import pytest
import numpy as np

print(sj)


def test_Dual():

    z = sj.Dual(1.0, 2.0)
    assert z.r == 1.0
    assert z.d == 2.0


def test_add():
    x = sj.Dual(1.0, 2.0)
    y = sj.Dual(3.0, 4.0)
    z = x + y
    assert z.r == 4.00
    assert z.d == 6.00


def test_divide():
    # real = (self.r*other)/other**2
    #    Dual = (self.d*other)/other**2
    z = sj.Dual(1.0, 4.0)
    result = z / 2.0
    assert result.r == 0.5
    assert result.d == 2.00


def test_multiply():
    x = sj.Dual(2.0, 4.0)
    y = sj.Dual(1.0, 1.0)
    z = x * y
    assert z.r == 2.00
    assert z.d == 6.00


def test_sin():
    x = sj.Dual(3.0, 4.0)
    z = x.sin()
    assert z.r == np.sin(3.0)
    assert z.d == 4 * np.cos(3.0)


def test_cos():
    x = sj.Dual(3.0, 4.0)
    z = x.cos()
    assert z.r == np.cos(3.0)
    assert z.d == -4.0 * np.sin(3.0)


# def test_pow():
# 	x=sj.Dual(1.0,2.0)
# 	y=2
# 	with pytest.raises(AttributeError):
# 		x**(y)


def test_pow_negative_case():
    x = sj.Dual(-1.0, 2.0)
    y = sj.Dual(3.0, 4.0)
    with pytest.raises(Exception):
        x ** (y)


def test_sqrt():
    x = sj.Dual(4.0, 1.0)
    z = x.sqrt()
    assert z.r == 2.0
    assert z.d == 0.25


def test_repr():
    x = sj.Dual([1, 2], [3, 4])
    z = x.__repr__()
    assert z == "[1.00 2.00] + eps [3.00 4.00]"


def test_repr2():
    x = sj.Dual(-1, -2)
    z = x.__repr__()
    assert z == "-1.00 - eps 2.00"


def test_repr3():
    x = sj.Dual(1, 2)
    z = x.__repr__()
    assert z == "1.00 + eps 2.00"
