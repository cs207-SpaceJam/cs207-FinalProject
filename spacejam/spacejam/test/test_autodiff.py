
import spacejam as sj
import pytest
import numpy as np 

def test_autodiff():
    def f(x):
        return np.array([x**3])
    p=np.array([5])
    ad=sj.AutoDiff(f, p)
    assert ad.r==[125.00]
    assert ad.d==[75.00]

def test_vector_p():
    def f(x_1, x_2, x_3):
        return np.array([x_1 + 2*x_2 + 3*x_3])
    p=np.array([1,2,3])
    ad=sj.AutoDiff(f,p)
    assert ad.r==[14.00]
    assert ad.d[0]==1.00 and ad.d[1]==2.00 and ad.d[2]==3.00

def test_vector_F():
    def F(x_1, x_2):
        f_1=1/x_1
        f_2=2/x_2
        f_3=1/(x_1+x_2)
        return np.array([f_1, f_2, f_3])
    p=np.array([1,2])
    z=sj.AutoDiff(F,p)
    zr_test = np.array([[1.00],[1.00],[0.33]])
    zd_test = np.array([[-1.00,-0.00], [-0.00,-0.50], [-0.11, -0.11]])
    assert repr(zr_test) == repr(z.r)
    assert repr(zd_test) == repr(z.d)
