import numpy as np
import matplotlib.pyplot as plt
x = np.array(np.arange(0,np.pi,0.05))
z = np.sin(x)
plt.plot(z)
plt.show()
# Factral Dimension Method
s = 100
def L_s(x):
    Ls = 0
    ls = np.zeros_like(x)
    for i in range(len(x)-1):
        l_s = np.sum(np.sqrt((x[i+1]-x[i])**2+(s**2)*((i+1)-i)**2))
        ls[i] = l_s
    Ls = np.sum(ls)
    return Ls
def D_s(x):
    Ds = 0
    ds = np.zeros_like(x)
    for i in range(len(x)-1):
        d_s = (np.sqrt((x[i+1]-x[i])**2+(s**2)*((i+1)-i)**2))
        ds[i] = d_s
    Ds = np.max(ds)
    return Ds
     
def GFactualDist(x):
    n = len(x)
    m = 4   
    GFD = np.zeros([int(x.shape[0]-m)])
    for i in range(len(x)-m):
        ds = D_s(x[i:i+m]) 
        ls = L_s(x[i:i+m-1])        
        gfd = (np.log10(n))/(np.log10(ds/ls)+np.log10(n))  
        #print(gfd)
        GFD[i] = gfd
    return GFD


Z = GFactualDist(z)
plt.plot(Z)
plt.show()