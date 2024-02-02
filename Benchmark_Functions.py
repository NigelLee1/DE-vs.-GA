import math
import numpy as np

def high_conditioned_elliptic(variable):
    dim = len(variable)
    ans = 0
    for i in range(dim):
        ans += pow(pow(10,6),i/(dim-1))* variable[i] ** 2
    return ans

def bent_cigar(variable):
    dim = len(variable)
    t1 = variable[0]**2
    t2 = 0
    for i in range(1,dim):
        t2 += variable[i] ** 2
    ans = t1 + pow(10,6) * t2
    return ans

def discus(variable):
    dim = len(variable)
    t1 = pow(10,6) * variable[0]**2
    t2 = 0
    for i in range(1,dim):
        t2 += variable[i] ** 2
    ans = t1 + t2
    return ans

def rosen_brock(variable):
    dim = len(variable)
    ans = 0
    for i in range(0,dim-1):
        ans += 100 * (variable[i] ** 2 - variable[i + 1]) ** 2 + (variable[i] - 1)**2
    return ans

def ackley(variable):
    dim=len(variable)
    t1=0
    t2=0
    for i in range(dim):
        t1+=variable[i]**2
        t2+=math.cos(2*math.pi*variable[i])
    ans=20+math.e-20*math.exp((t1/dim)*-0.2)-math.exp(t2/dim)
    return ans

def weierstrass(variable):
    dim=len(variable)
    a=0.5
    b=3
    ans=0
    for i in range(dim):
        t1=0
        for k in range (0,21):
            t1+=(a**k)*math.cos((2*math.pi*(b**k))*(variable[i]+0.5))
            # t1 += np.power(a, k) * math.cos((2 * math.pi * np.power(b, k)) * (variable[i] + 0.5))
        ans+=t1
    t2=0
    for k in range (0,21):
        t2+=(a**k)*math.cos(2*math.pi*(b**k)*0.5)
        # t2 += np.power(a, k) * math.cos(2 * math.pi * np.power(b, k) * 0.5)
    ans-=dim*t2
    return ans

def griewank(variable):
    dim=len(variable)
    t1 = 0
    t2 = 1
    for i in range(dim):
        t1 += variable[i]**2
        t2 *= math.cos(variable[i] / math.sqrt(i + 1))
    ans = 1 + t1 / 4000 - t2
    return ans

def rastrigin(variable):
    dim=len(variable)
    ans=0
    for i in range(dim):
        ans+=(variable[i]**2)-10*math.cos(2*math.pi*variable[i])+10
    return ans