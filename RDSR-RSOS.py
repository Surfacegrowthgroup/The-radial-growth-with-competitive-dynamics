import numpy as np
import tqdm
from scipy import stats
import matplotlib.pyplot as plt


def Glt(height):
    L = len(height)
    l = L//2
    Gl = np.zeros(l)
    for i in range(l):
        G1 = np.zeros(L-l)
        for j in range(L-l):
            G1[j] = (height[j+i]-height[j])**2
        Gl[i] = np.sum(G1)/L
    return Gl


def growth(N0, T, p):
    Sub = np.zeros(N0)
    w = []
    SubT = []
    #plt.figure(figsize=(20,20))
    for t in range(T):
        if np.random.binomial(1, p) == 0:
            N = len(Sub)
            noise = np.random.randint(0, 2, N)
            site = np.where(noise == 1)[0]
            for i in site:
                if i == N - 1:
                    nearest = np.array((Sub[i], Sub[i - 1], Sub[0]))
                    lowest = np.where(nearest == np.min(nearest))[0]
                    if lowest[0] == 0:
                        Sub[i] += 1
                    else:
                        test = np.sum(lowest)
                        if test == 1:
                            Sub[i - 1] += 1
                        elif test == 2:
                            Sub[0] += 1
                        elif test == 3:
                            temp = np.random.randint(0, 2)
                            if temp == 0:
                                Sub[i - 1] += 1
                            elif temp == 1:
                                Sub[0] += 1
                else:
                    nearest = np.array((Sub[i], Sub[i - 1], Sub[i + 1]))
                    lowest = np.where(nearest == np.min(nearest))[0]
                    if lowest[0] == 0:
                        Sub[i] += 1
                    elif lowest[0] != 0:
                        test = np.sum(lowest)
                        if test == 1:
                            Sub[i - 1] += 1
                        elif test == 2:
                            Sub[i + 1] += 1
                        elif test == 3:
                            temp = np.random.randint(0, 2)
                            if temp == 0:
                                Sub[i - 1] += 1
                            elif temp == 1:
                                Sub[i + 1] += 1
        else:
            N = len(Sub)
            noise = np.ones(N)
            # site = np.where(noise == 1)[0]
            for j in range(N):
                if noise[j] == 1:
                    i = np.random.randint(0, N)
                    if i == N - 1:
                        l, m, r = Sub[i - 1], Sub[i] + 1, Sub[0]
                    else:
                        l, m, r = Sub[i - 1], Sub[i] + 1, Sub[i + 1]
                    if ((m - l) <= 1) and ((m - r) <= 1):
                        Sub[i] = Sub[i] + 1
                    else:
                        continue
                else:
                    continue

        '''
        degree = np.linspace(0, 2 * np.pi, N)
        plt.polar(degree, Sub, marker='o', ls='none', markersize=2.5, color='#f38181')
        plt.draw()
        '''
        w.append(np.std(Sub))

        Sub = list(Sub)
        for k in range(3):
            L = len(Sub)
            add_site = np.random.randint(0, L)
            temp = np.random.randint(0,2)
            if add_site == L-1:
                lll, rrr = Sub[add_site - 1], Sub[0]
                Sub.insert(add_site, rrr)
                Sub.insert(add_site, lll)
            else:
                lll, rrr = Sub[add_site - 1], Sub[add_site + 1]
                Sub.insert(add_site, rrr)
                Sub.insert(add_site, lll)
        Sub = np.array(Sub)

        SubT.append(Sub)

        # print(Sub)
        # print(t)
    #plt.axis('off')
    #plt.grid(False)
    #plt.savefig('E:\Science\Radial\paper_2nd\Figures\Atemp.png',dpi=200,bbox_inches='tight',pad_inches=0,transparent=False)
    #plt.show()

    return w,Sub,SubT


probility = 0.5
SL = 6
Time = 1000
Repeat = 512
W = np.zeros((Repeat, Time))
AllSub = np.array([])

Gs1 = np.zeros(123)
Gs2 = np.zeros(243)
Gs3 = np.zeros(483)
Gs4 = np.zeros(963)
Gs5 = np.zeros(1923)
Gs6 = np.zeros(3003)
for re in tqdm.tqdm(range(Repeat)):
    Val = growth(SL, Time, probility)
    W[re] = Val[0]
    AllSub = np.append(AllSub, Val[1])
    Gs1 = Gs1 + Glt(Val[2][39])
    Gs2 = Gs2 + Glt(Val[2][79])
    Gs3 = Gs3 + Glt(Val[2][159])
    Gs4 = Gs4 + Glt(Val[2][319])
    Gs5 = Gs5 + Glt(Val[2][639])
    Gs6 = Gs6 + Glt(Val[2][999])
    L1 = len(Val[2][39])
    L2 = len(Val[2][79])
    L3 = len(Val[2][159])
    L4 = len(Val[2][319])
    L5 = len(Val[2][639])
    L6 = len(Val[2][999])

logW, logt = np.log10(np.mean(W, axis=0)), np.log10(range(1, Time + 1))
S = stats.skew(AllSub)
K = stats.kurtosis(AllSub)
print(S)
print(K)
res = stats.linregress(logt[100:1000], logW[100:1000])
Beta = round(res.slope,5)
print(Beta-0.01)

Xl = list(AllSub)
X = list(set(Xl))
P = []
Lm = len(AllSub)
for xx in X:
    P.append(Xl.count(xx) / Lm)
X = (np.array(X) - np.mean(X)) / np.std(X)
# print(AllSub.shape)
# print(len(Xl))
# np.savetxt("./data/temp/temp1.txt",logW)
# np.savetxt("./data/temp/temp2.txt",logt)
#np.savetxt("./data/temp/temp3.txt", Xl)
#np.savetxt("./data/temp/temp4.txt", P)
logl1 , logG1 = np.log2(range(1,L1//2 + 1)), np.log2(Gs1/Repeat)
logl2 , logG2 = np.log2(range(1,L2//2 + 1)), np.log2(Gs2/Repeat)
logl3 , logG3 = np.log2(range(1,L3//2 + 1)), np.log2(Gs3/Repeat)
logl4 , logG4 = np.log2(range(1,L4//2 + 1)), np.log2(Gs4/Repeat)
logl5 , logG5 = np.log2(range(1,L5//2 + 1)), np.log2(Gs5/Repeat)
logl6 , logG6 = np.log2(range(1,L6//2 + 1)), np.log2(Gs6/Repeat)
np.savetxt("./data/temp/temp1.txt",logG1)
np.savetxt("./data/temp/temp2.txt",logG2)
np.savetxt("./data/temp/temp3.txt",logG3)
np.savetxt("./data/temp/temp4.txt",logG4)
np.savetxt("./data/temp/temp5.txt",logG5)
np.savetxt("./data/temp/temp6.txt",logG6)
print("Finish")




'''
fig1 = plt.figure()
plt.scatter(logl, logG)

fig2 = plt.figure()
beff = []
gap = 10
for k in range(0,3003-gap):
    resgap = stats.linregress(logl[k:k+gap], logG[k:k+gap])
    betagap= round(resgap.slope,5)
    beff.append(betagap)
beff = np.array(beff)
plt.plot(np.log2(range(0,3003-gap)), beff/2)
'''
plt.show()
