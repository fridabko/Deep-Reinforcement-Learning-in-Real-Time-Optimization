import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import make_interp_spline, BSpline
import seaborn as sns

#Plotting settings
sns.set()
sns.set_style("whitegrid")


FList = []
TList = []

filename ="RLRTO_DDPG" #RLRTODDPG #"TD3_5k"

with open("Fb"+filename+".txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",") #returns an array of 1 line
        for j in range(len(currentline)-1):
            if j==0:
                currentline[j] = float(currentline[j][1:])
            elif j<5:
                currentline[j] = float(currentline[j])
        currentline = currentline[:5]
        FList.append(currentline)
with open("T"+filename+".txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",") #returns an array of 1 line
        for j in range(len(currentline)-1):
            if j==0:
                currentline[j] = float(currentline[j][1:])
            elif j<5:
                currentline[j] = float(currentline[j])
        currentline = currentline[:5]
        TList.append(currentline)
FList = np.array(FList)
TList = np.array(TList)

meanFList = []
meanTList = []
FStdList = []
TStdList = []

for j in range(len(FList[0])):
    c11F = FList[:,j] #column
    c11Fmeanj = np.mean(c11F)
    c11FStd = np.std(c11F)
    meanFList.append(c11Fmeanj)
    FStdList.append(c11FStd)

    c11T = TList[:,j] #column
    c11Tmeanj = np.mean(c11T)
    c11TStd = np.std(c11T)
    meanTList.append(c11Tmeanj)
    TStdList.append(c11TStd)

meanFList = np.array(meanFList)
meanTList = np.array(meanTList)
FStdList = np.array(FStdList)
TStdList = np.array(TStdList)

#Constraint bounds
G1_Fb = [4,4.125,4.3,4.39,4.45,4.5,4.75,5,5.25,5.36,5.5,5.75,6,6.1]
G1_Tr = [84,82.65,81.2,80.5,80,79.5,77.6,76,74.5,73.9,73.2,71.85,70.5,70]
a_BSpline = make_interp_spline(G1_Fb, G1_Tr)
G1_Tr_new = a_BSpline(G1_Fb)
# 300 represents number of points to make between x.min and x.max
G1_Fb_new = np.linspace(4, 6.1, 300) 

spl = make_interp_spline(G1_Fb, G1_Tr, k=3)  # type: BSpline
G1_Tr_smooth = spl(G1_Fb_new)

G2_Fb = [4,4.3,4.39,4.5,4.7,5.05,5.25,5.5,7]
G2_Tr = [77.7,80,80.5,81.5,82.5,85,86,88,98.5]
G2_Fb = [4,7]
G2_Tr = [77.7,99.1]
G2_Tr1 = np.linspace(77.7,99.1,600)

x = np.append(G1_Fb_new,np.linspace(6.1,7,100))
G1_Tr_add = np.ones(100)*70
G1_Tr_fill = np.append(G1_Tr_smooth, G1_Tr_add)
y = np.linspace(10,98.5,300)
midTr = 80.4948*np.ones(400)
midTr2 = 80.4948*np.ones(600)
x2 = np.linspace(4,7,600)

xopt = 4.3894
yopt = 80.4948



#####TR(FB) plot#########
plt.figure(figsize=(8,6))
#plt.title('DDPG-RTO simulation')
plt.plot(G1_Fb_new,G1_Tr_smooth,'r',label = '$g_1$')
plt.plot(G2_Fb,G2_Tr,color='darkred',label = '$g_2$')
plt.fill_between(x,midTr, G1_Tr_fill, where = (x>4.3894),color='lightblue')
plt.fill_between(x2,midTr2, G2_Tr1, color='lightblue',where = (x2>4.3894))
plt.fill_between(meanFList, meanTList-TStdList, meanTList+TStdList,alpha=0.5, facecolor='#FF9848', edgecolor ='#FF9848')

plt.plot(meanFList,meanTList,'--')
plt.plot(meanFList,meanTList,'*', label='$u_k$',color='tab:blue')
plt.plot(meanFList[-1],meanTList[-1],'r*',label='$u_k^f$')
plt.axis([4,7, 70, 100])
plt.grid(color='0.95')
plt.xlabel("$F_b$ [kg/s]")
plt.ylabel("$T_R$ [°C]")
plt.legend()
plt.savefig(filename+'.png')
plt.show()