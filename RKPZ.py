import numpy as np
import tqdm
from scipy import stats
import matplotlib.pyplot as plt


def Glt(height):
    L = len(height)
    l = L//2
    Gl = np.zeros(l)
    for i in range(l):
        Gx = np.zeros(L-l)
        for j in range(L-l):
            Gx[j] = (height[j+i]-height[j])**2
        Gl[i] = np.sum(Gx)/(L-l)
    Gl = Gl/L
    return Gl

def mainloop(L0, L, t, a, b, r0, dx, dt):
    interface = L0
    interface[0] = interface[0] * r0
    for j in range(0, t - 1):
        # print(interface)
        ri = np.copy(interface[j])
        ri_1 = np.roll(ri, -1)
        ri1 = np.roll(ri, 1)

        #P = np.random.binomial(1, 1)

        linear = (a / (ri * ri)) * ((ri1 + ri_1 - 2 * ri) / dx ** 2)
        lateral = b*(1+ ( ((ri1-ri)**2+(ri-ri_1)**2)+(ri1-ri)*(ri-ri_1)/(3*dx**2) )/(2*ri**2))
        #lateral = b * (1 + ( (ri1-ri_1)/(2*dx) ) / (2 * ri ** 2))
        noise = np.random.randint(-1, 2, L)*1
        interface[j+1] = np.round((linear + lateral + noise) * dt + ri,6)

        # print(linear)
        # print(interface[j+1])
        # print('-------------------------------------------------------------------------------------------------------')
    return interface


Interface_length = 256
time = 1000
D = 1
F = 0
R0 = 25
dtheta = 2*np.pi/Interface_length
dtime = 0.001
repeat = 1024
Gs = np.zeros(128)
W = np.zeros(time)
hight = np.array([])
for re in tqdm.trange(repeat):
    L0 = np.zeros((time, Interface_length))
    L0[0] = L0[0] + 1
    data = mainloop(L0, Interface_length, time, D, F, R0, dtheta, dtime)
    hight = np.append(hight, data[-1])
    w = np.std(data, axis=1)
    W = W + w
    #Gs = Gs + Glt(data[-1])
    L0 = 256
W = W/repeat

hight  = np.round(hight,6)

radii = hight[0:time]
# radii = np.ones(256)
theta = np.linspace(0,360,len(radii))
#ax = plt.subplot(111, projection='polar')
#ax.plot(theta,radii)
#ax.grid(False)
#plt.show()

print(stats.skew(hight))
print(stats.kurtosis(hight))

x, y = np.log10(np.arange(1,time+1)), np.log10(W)

res = stats.linregress(x[500:700], y[500:700])
Beta = round(res.slope,5)
print(round(Beta,4))

beff = []
gap = 10
for k in range(0,time-gap):
    resgap = stats.linregress(x[k:k+gap], y[k:k+gap])
    betagap= round(resgap.slope,5)
    beff.append(betagap)
print(round(np.mean(beff[200:300]),4))

data2 = data
for i in range(time-1):
    data2[i+1] = data2[i+1]-(np.mean(data2[i+1])-np.mean(data2[i])-0.000001*i)

np.savetxt("./data/temp/temp1.txt",x)
np.savetxt("./data/temp/temp2.txt",y)
np.savetxt("./data/temp/temp3.txt",data2.T)
np.savetxt("./data/temp/temp4.txt",theta)

fig1 = plt.figure()
plt.plot(x, y)
fig2 = plt.figure()
plt.plot(range(0,time-gap), beff)

logl , logG = np.log2(range(1,L0//2 + 1)), np.log2(Gs/repeat)
# np.savetxt("./data/temp/temp1.txt",logl)
#np.savetxt("./data/temp/temp1.txt",logG)
'''
fig3 = plt.figure()
plt.scatter(logl, logG)

fig4 = plt.figure()
beff = []
gap = 10
for k in range(0,128-gap):
    resgap = stats.linregress(logl[k:k+gap], logG[k:k+gap])
    betagap= round(resgap.slope,5)
    beff.append(betagap)
beff = np.array(beff)
plt.plot(np.log2(range(0,128-gap)), beff/2)
'''
plt.show()
