import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

def f1(x,y):
    return 6*x/y + y/x

def f2(x,y):
    return 3*(x**2) - 4*y

case = 1 #1 or 2

if case == 1:
    X_1 = [1.4507593574119826, 1.3507486189061431, 1.3437418445036584, 1.3996234304570205, 1.3979518828892759, 1.3945465639823558, 1.3893781212050336, 1.3782470988467692, 1.3648270165860439, 1.3479365311663805, 1.3285108377900048, 1.3120951990134058, 1.2868375658377489, 1.2656330355743055, 1.2405363570128518, 1.2203868288824882, 1.1945421583475919, 1.1740931868275142, 1.1457783211309374, 1.1247194811397321, 1.0940924653259601, 1.0740971555978447, 1.046665207920993, 1.0312717130314046, 1.0142926928778742, 1.00675530607401, 1.0018964610836723]
    Y_1 = [1.2746203212940088, 1.3639448189323231, 1.3710893232307095, 1.4289557868431355, 1.4306623653652537, 1.4341480429624689, 1.4394642565204157, 1.451025238640514, 1.4651772646109298, 1.4835195450758685, 1.5049548604941887, 1.5240408416391231, 1.5534554696305445, 1.5797955767210212, 1.6112417762895508, 1.6383765888916266, 1.6732018884304294, 1.702924636387104, 1.7441888280507536, 1.7775956790981118, 1.8263521662204718, 1.8613769128483673, 1.9094365785178182, 1.9389131820227419, 1.9712756615641267, 1.9864663621785372, 1.9961669981835553]
    Z_1 = [f1(X_1[i],Y_1[i]) for i in range(len(X_1))]

    X_list = np.arange(0.2, 1.6, .005)
    Y_list = np.arange(1.2, 2.1, .005)
    X, Y = np.meshgrid(X_list, Y_list)
    Z = []
    for i in range(X.shape[0]):
        temp = []
        for j in range(X.shape[1]):
            temp.append(f1(X[i][j],Y[i][j]))
        Z.append(temp)
    Z = np.array(Z)

    c = np.arange(1, 8, .1)

    fig = plt.figure()
    cp = plt.contour(X, Y, Z,c, alpha=0.6)
    plt.scatter(X_1,Y_1,s=10,color='black',marker='x',alpha=1)
    plt.scatter(X_1[-1],Y_1[-1],s=100,color='red',marker='o',alpha=0.5)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title('Contour Plot for f(x) = 6*x/y + y/x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

elif case == 2:
    X_1 = [-18.005857674048542, -17.892804554750331, -6.1140432780749148, -1.3268403020185504, -1.0182491323328142, -1.0000635975161252, -1.0000635975161252]
    Y_1 = [82.983229092913945, 39.78560271430662, 16.228086556149837, 6.6536806040371008, 6.0364982646656289, 6.000127195032249, 6.000127195032249]
    Z_1 = [f2(X_1[i],Y_1[i]) for i in range(len(X_1))]
    X_list = np.arange(-20, 20, .1)
    Y_list = np.arange(-10, 90, .1)
    X, Y = np.meshgrid(X_list, Y_list)
    Z = []
    for i in range(X.shape[0]):
        temp = []
        for j in range(X.shape[1]):
            temp.append(f2(X[i][j],Y[i][j]))
        Z.append(temp)
    Z = np.array(Z)
    c = np.arange(-1000, 1000, 10)

    fig = plt.figure()
    cp = plt.contour(X, Y, Z,c, alpha=0.6)
    # cp = plt.contour(X, Y, Z, alpha=0.6)
    plt.scatter(X_1,Y_1,s=30,color='black',marker='x',alpha=1)
    plt.scatter(X_1[-1],Y_1[-1],s=300,color='red',marker='o',alpha=0.5)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title('Contour Plot for f(x) = 3x^2 - 4y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-20,20])
    plt.ylim([-10,90])
    plt.show()
