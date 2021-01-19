import pandas


data = pandas.read_csv('Britian Lorenz Curve Data.csv', sep=';')
CW = list(data['CW'])
CHH = list(data['CHH'])
# @CW : Cumulative Wealth
# @CHH : Cumulative HouseHolds
# SCALE:
p = 1


def scale_by_k(r, k):
    L = []
    for l in r:
        L.append(l*k)
    return L


def remove_redundancy(x, y):
    for i in range(len(x) - 1):
        if x[i] == x[i+1]:
            x[i] = None
            y[i] = None
    while True:
        try:
            x.remove(None)
        except ValueError:
            break
    while True:
        try:
            y.remove(None)
        except ValueError:
            break
    return [x, y]


CHH = remove_redundancy(CHH, CW)[0]
CW = remove_redundancy(CHH, CW)[1]

CW = scale_by_k(CW, p)
CHH = scale_by_k(CHH, p)


# function to test the methods
def transformation(r):
    k = []
    for i in range(len(r)):
        k.append(4*r[i]*r[i]*r[i] + 3*r[i]*r[i] - 2*r[i] + 4)
    return k


# Trapezoidal method
def evaluate_trapezoidal(x, y):
    if len(x) == len(y):
        n = len(x)
        sum_trap = 0
        for i in range(n-1):
            del_x_i = x[i+1] - x[i]
            sum_trap += del_x_i*(y[i] + y[i+1])/2
        return sum_trap
    else:
        return 'error, x is ' + str(len(x)) + ' and y is ' + str(len(y))


# Simpson's Method
def evaluate_simpson(x, y):
    if len(x) == len(y):
        n = len(x) - 1
        sum_simps = 0
        for i in range(0, n, 2):
            s = x[i+1] - x[i]
            h = x[i+2] - x[i+1]
            B = (s*s*(y[i+2] - y[i+1]) + h*h*(y[i+1] - y[i]))/(h*s*s + s*h*h)
            sum_simps += (h*(B*h + 4*y[i+1] + 2*y[i+2]) + s*(2*y[i] + 4*y[i+1] - B*s))/6
        return sum_simps
    else:
        return 'error, x is ' + str(len(x)) + ' and y is ' + str(len(y))


Gini_Index_Trap = 1 - 2*evaluate_trapezoidal(CHH, CW)/(10000*p*p)
Gini_Index_Simps = 1 - 2*evaluate_simpson(CHH, CW)/(10000*p*p)
print('The Gini Index of Britain is :\nBy Trapezoidal Rule : ' + str(Gini_Index_Trap) +
      '\nBy Simpson\'S Rule: ' + str(Gini_Index_Simps))
