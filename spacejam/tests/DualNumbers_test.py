import pytest
import spacejam.Dual as sp
import numpy as np 


def test_dual():

	z= sp.Dual(1.0,2.0)
	assert z.r==1.0
	assert z.d==2.0

def test_add():
	x=sp.Dual(1.0, 2.0); 
	y=sp.Dual(3.0, 4.0)
	z= x+y
	assert z.r == 4.00
	assert z.d == 6.00

def test_divide():
	# real = (self.r*other)/other**2
 #    dual = (self.d*other)/other**2
	z = sp.Dual(1.0, 4.0) 
	result=z/2.0
	assert result.r==0.5
	assert result.d==2.00


def test_multiply():
	x=sp.Dual(2.0, 4.0) 
	y=sp.Dual(1.0,1.0)
	z=x * y 	 
	assert z.r==2.00
	assert z.d==6.00


def test_sin():
	x=sp.Dual(3.0, 4.0)
	z = x.sin()
	assert z.r==np.sin(3.0)
	assert z.d==4*np.cos(3.0)

def test_cos():
	x=sp.Dual(3.0, 4.0)
	z=x.cos()
	assert z.r==np.cos(3.0)
	assert z.d==-4.0*np.sin(3.0)

	



