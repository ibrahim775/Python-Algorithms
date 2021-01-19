import math
# @x is the independent variable I am testing with functions @Exponential and
# @Polynomial
def Exponential (b, p) : 
    result = math.exp(p*math.log(b, math.exp(1)))
    return result
# Now I have the exponential function of base @base and power @power
def Polynomial (b, p) :
    if (p > 1):
        result = b*Polynomial(b, p-1)
    elif (p == 1) :
        result = b
    return result
for n in range (20, 30) :
    print ('for n = ' + str(n))
    for x in range(1, 100):
        ratio = Polynomial (x, n)/ Exponential (n, x)
        print ('      for x = ' + str(x) + ' the ratio is ' + str(ratio))


# The ratio is approaching zero, which means that the Polynomial grows slower
# than the exponential
