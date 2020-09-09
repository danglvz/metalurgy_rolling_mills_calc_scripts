import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp1d

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def profilvalka(x_valka):
    r1 = 292/2
    r2 = 300/2
    r3 = 265/2

    ang1 = 1.77*math.pi/180
    ang2 = 11*math.pi/180
    ang3 = 13*math.pi/180

    if 0 <= x_valka <= 130:
        return r1 + x_valka*math.tan(ang1)

    if 130 < x_valka <= 220:
        return r2 - (x_valka-130)*math.tan(ang2)

    if 220 < x_valka <= 270:
        return r3 - (x_valka-220)*math.tan(ang3)


alpha = 12*math.pi/180
betta = 15*math.pi/180


a = np.array([0,130,220,270])
#a = np.linspace(0,270,100)
b = np.linspace(2.7,3.7,1000)

u, x =np.meshgrid(b,a)

#x , u = np.mgrid[0:271:3j, 0:2*np.pi:100j]

profilvalka1 = np.vectorize(profilvalka)
print(x[0][0])
y = profilvalka1(x)*np.sin(u)
z = profilvalka1(x)*np.cos(u)
x-=130
z = z + profilvalka(130)

for i in range(len(x)):
    for k in range(len(x[i])):
        a1 = x[i][k]*math.cos(alpha)+z[i][k]*math.sin(alpha)
        b1 = x[i][k]*(-1)*math.sin(alpha)+ z[i][k]*math.cos(alpha)
        x[i][k] = a1
        z[i][k] = b1


for i in range(len(x)):
    for k in range(len(x[i])):
        a1 = x[i][k]*math.cos(betta)-y[i][k]*math.sin(betta)
        b1 = x[i][k]*math.sin(betta)+y[i][k]*math.cos(betta)
        x[i][k] = a1
        y[i][k] = b1


z+=105

short_radius = np.sqrt(z**2+y**2)


short_lenght = np.argmin(short_radius, axis = 1)

xs = []
ys = []
zs = []

for i in range(len(short_lenght)):
    xs.append(x[i][short_lenght[i]])
    ys.append(y[i][short_lenght[i]])
    zs.append(z[i][short_lenght[i]])

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)


minline_y = interp1d(xs,ys)
minline_z = interp1d(xs,zs)


minline_x = np.arange(xs[0],xs[-1], 10)

xnradmin = minline_x[0]
xnn1 = minline_x[0]*math.cos(betta*(0))-minline_y(minline_x[0])*math.sin(betta*(0))
ynn2 = minline_x[0]*math.sin(betta*(0))+minline_y(minline_x[0])*math.cos(betta*(0))
znn1 = minline_z(minline_x[0])-105

xnn2 = xnn1*math.cos(alpha*(-1))+znn1*math.sin(alpha*(-1))
znn2 = (-1)*xnn1*math.sin(alpha*(-1))+znn1*math.cos(alpha*(-1))
print(xnn2)
print('fjfksfjs')
print(np.arctan(ynn2 / znn2)+np.pi)

cv = 120*math.pi/180


#for i in range(len(y)):
    #for k in range(len(y[i])):
        #a1 = y[i][k]*math.cos(cv)-z[i][k]*math.sin(cv)
        #b1 = y[i][k]*math.sin(cv)+z[i][k]*math.cos(cv)
        #y[i][k] = a1
        #z[i][k] = b1



print(profilvalka((xnn2+130)))
print(int(xnn2+130))

surfacespeed = 3.665191429188*profilvalka1(xnn2+130)*np.cos(betta)
vx1 = 3.0

bettax = np.arctan(vx1/surfacespeed)
print(bettax*180/np.pi)



short_radius = np.sqrt((np.sqrt(z**2+y**2)-profilvalka1((xnn2+130)))**2)
#print(np.amin(short_radius, axis=1))
short_lenght = np.argmin(short_radius, axis = 1)
xn = []
yn = []
zn = []
for i in range(len(x)):
    xn.append(x[i][short_lenght[i]])
    yn.append(y[i][short_lenght[i]])
    zn.append(z[i][short_lenght[i]])


#print(xn)
#print(yn)
#print(zn)

int1 = interp1d(xn,yn)
int2= interp1d(xn,zn)




fig = plt.figure()







axes = Axes3D(fig)

bn = np.linspace(xn[0],xn[-1],100)
axes.plot_surface(x, y, z, rstride=1, cstride=1)
axes.plot(xs, ys, zs, label='parametric curve')
#axes.plot(bn, int1(bn), int2(bn), label='parametric curve')
plt.show()

