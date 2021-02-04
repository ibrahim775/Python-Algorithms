from math import sqrt
from math import pi
from math import sin
from math import pow
from matplotlib import pyplot

R = 1
g = 9.81
T_ln = 2*pi*sqrt(R/g)
f_ln = 1/T_ln

phi = []
for i in range(158):
    phi += [i/100]


def double_factorial(r):
    if r <= 1:
        return 1
    else:
        return r*double_factorial(r-2)


def CEIFK(k):
    n = 300
    result = 0
    for j in range(n):
        o1 = double_factorial(2*j-1)
        o2 = double_factorial(2*j)
        result += pow(k, 2*j)*pow(o1/o2, 2)
    result = result*pi/2
    return result


def T_nln(initial_angle):
    parameter = sin(initial_angle)
    result = 4*sqrt(R/g)*CEIFK(parameter)
    return result


# Until this point, the calculations are awesome!!
# Now I want to visualize that!!
def linear_transformation(r):
    k = []
    for s in range(len(r)):
        k.append(T_ln)
    return k


def nonlinear_transformation(r):
    k = []
    for s in range(len(r)):
        k.append(T_nln(r[s]))
    return k


linear_time = linear_transformation(phi)
nonlinear_time = nonlinear_transformation(phi)

pyplot.scatter(phi, linear_time)
pyplot.title('Linear Period time of the Pendulum Vs. Initial Angle')
pyplot.xlabel('Initial Angle')
pyplot.ylabel('Linear Period time')
pyplot.show()

pyplot.scatter(phi, nonlinear_time)
pyplot.title('Non-Linear Period time of the Pendulum Vs. Initial Angle')
pyplot.xlabel('Initial Angle')
pyplot.ylabel('Non-Linear Period time')
pyplot.show()

pyplot.plot(phi, linear_time)
pyplot.plot(phi, nonlinear_time)
pyplot.legend(['Linear Periodic Time', 'Non-Linear Periodic Time'])
pyplot.show()
