import spacejam as sj
import pytest
import numpy as np
import sys


h = 0.01
N = 500
T = h*N
X = np.zeros(N)
X[0] = 1
t = np.arange(0, T, h)

# toy function
def f(x):
    f1 = -x
    return np.array([f1])

# analytic solution
def analytic(t): #t=np.arange(0, T=5, h=0.01)
    x = np.exp(-t)
    return x

# the X_test is the analytical result
X_test = analytic(t)

# test simulation for Adams-Moulton order 0 with a toy example
def test_amso():
	# Initial conditions
	
	for n in range(N-1):
	    X[n+1] = sj.integrators.amso(f, np.array([X[n]]), h=h, X_tol=1E-4)

	tol_0 = 0.0032 
	diff_0 = np.abs(X[-1] - X_test[-1]) 

	assert diff_0 < tol_0

# test simulation for Adams-Moulton order 1 with a toy example
def test_amsi():
	X = np.zeros(N); X[0] = 1; t = np.arange(0, T, h)

	for n in range(N-1):
	    X[n+1] = sj.integrators.amsi(f, np.array([X[n]]), h=h, X_tol=1E-4)

	tol_1 = 0.0032 
	diff_1 = np.abs(X[-1] - X_test[-1]) 
	assert diff_1 < tol_1


# test simulation for Adams-Moulton order 2 with a toy example
def test_amsii():
	X = np.zeros(N); X[0] = 1; t = np.arange(0, T, h)

	for n in range(N-1):
	    X[n+1] = sj.integrators.amsii(f, np.array([X[n]]), np.array([X[n-1]]),h=h, X_tol=1E-4)

	tol_2 = 0.00315
	diff_2 = np.abs(X[-1] - X_test[-1]) 
	assert diff_2 < tol_2




# def test_amso_diverge(capsys):
# 	with self.assertRaises(SystemExit) as cm:
# 	# with pytest.raises(SystemExit):
# 		for n in range(N-1):
# 			X[n+1] = sj.integrators.amso(f, np.array([X[n]]), h=h, X_tol=1E-6, i_tol=0)

# 	self.assertEqual(cm.exception.code, 1)
# 	# out, err = capsys.readouterr()
# 	# assert err == 1

#def test_amso_diverge():
	# with pytest.raises(SystemExit) as pytest_wrapped_e:
	# 	for n in range(N-1):
	# 		X[n+1] = sj.integrators.amso(f, np.array([X[n]]), h=h, X_tol=1E-6, i_tol=0)
	# assert pytest_wrapped_e == SystemExit
	# assert pytest_wrapped_e.value.code==1
	
