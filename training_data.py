
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project, Dx, Array, FunctionSpace, dx
import numpy as np
from sympy import symbols, cos, sin, exp, lambdify,log, integrate
import sympy as sp
import matplotlib.pyplot as plt
from sympy import diff, Symbol


x=Symbol('x')

# given the space and the expansion coefficients this function assembles the
# spectral method solution for the PDE
# args: type(space)= FunctionSpace, function space
#       type(coefficients)=np.array[float], expansion coefficients
def assemble_expansion(space, coefficients):
    expansion=[coefficients[i]*space.basis_function(i,x=x) for i in range(space.N)]
    return sum(expansion)

# estimate the global error
# the estimation is based on two spectral method solutions u_N and u_K
# it has the form \int_\Omega \grad (u_N-u_K)**2
# args: type(spaces)=list[FunctionSpace], the spaces u_N and u_K
#       type(coefficients)= list[np.array[float]], expansion coefficients
#       a,b are domain boundaries
def compute_error_estimation(spaces, coefficients, a:int, b:int):
    exp1 = assemble_expansion(spaces[0], coefficients[0])

    exp2 = assemble_expansion(spaces[1], coefficients[1])

    #print((exp2 - exp1).diff(x, 1) ** 2)
    return integrate(((exp2 - exp1).diff(x, 1)) ** 2, (x, a, b))