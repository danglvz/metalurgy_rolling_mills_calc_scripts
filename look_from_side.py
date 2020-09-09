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






x , u = np.mgrid[0:271:3j, 0:2*np.pi:100j]

profilvalka1 = np.vectorize(profilvalka)

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




ud = np.linspace(0,2*np.pi,36)
xd = np.arange(0,len(xs),5)
udv, xd = np.meshgrid(ud, xd)

def radd(l):
    return short_radius[int(l)][short_lenght[int(l)]]

radd = np.vectorize(radd)

yd = np.sin(udv)*radd(xd)
zd = np.cos(udv)*radd(xd)
xd = xs[xd]


fig = plt.figure()


print(np.nanmax((np.sqrt(yd**2+zd**2))))

print(np.nanmin((np.sqrt(yd**2+zd**2))))


cv = 90*math.pi/180


for i in range(len(y)):
    for k in range(len(y[i])):
        a1 = y[i][k]*math.cos(cv)-z[i][k]*math.sin(cv)
        b1 = y[i][k]*math.sin(cv)+z[i][k]*math.cos(cv)
        y[i][k] = a1
        z[i][k] = b1




print(x[2])




axes = Axes3D(fig)

axes.plot_surface(xd, yd, zd, rstride=1, cstride=1)
axes.plot_surface(x, y, z, rstride=1, cstride=1)
axes.plot(xs, ys, zs, label='parametric curve')

plt.show()

