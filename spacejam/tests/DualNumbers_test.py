import pytest
import spacejam.Dual as Dual
import numpy as np 

class TestComplex():
	def test_dual(self):
		z= Dual.Dual(1.0,2.0)
		assert z.r=1.0
		assert z.d=2.0
    def test_add(self):
        
        z= Dual.Dual(1.0, 2.0) + Dual.Dual(3.0, 4.0)
        assert z.r == 3.0
        assert z.d == 6.0

    def test_divide(self,other):
        z = Dual.Dual(2.0, 4.0) 
        result=z.trudiv(2.0)
        assert z.r== 1.0
        assert z.d== 2.0 

    def test_multiply(self,other):
    	x=Dual.Dual(2.0, 4.0) 
    	y=Dual.Dual(1.0,1.0)
    	z=x.mul(y)
    	assert z.r==2.0
    	assert z.d==6.0


    def test_sin(self):
    	x=Dual.Dual(3.0, 4.0)
        z = x.sin()
        assert z.r==np.sin(3.0)
        assert z.d==4*np.cos(3.0)

    def test_cos(self):
    	x=Dual.Dual(3.0, 4.0)
    	z=x.cos()
    	assert z.r==np.cos(3.0)
    	assert z.d==-4.0*np.sin(3.0)

    def test_tan(self):
    	x=Dual.Dual(3.0, 4.0)
    	z= x.sin()/x.cos()
    	assert x.tan()==Dual.Dual(z.r, z.d)


