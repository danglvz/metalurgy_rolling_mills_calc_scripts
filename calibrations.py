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


alpha = 10*math.pi/180
betta = 21*math.pi/180


a = np.array([0,130,220,270])
#a = np.linspace(0,270,100)
b = np.linspace(0,2*np.pi,100)

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


z+=52.5

a10 = np.array([0,130,220,270])
#a = np.linspace(0,270,100)
b10 = np.linspace(0,2*np.pi,100)

u10, x10 =np.meshgrid(b10,a10)

#x , u = np.mgrid[0:271:3j, 0:2*np.pi:100j]

profilvalka1 = np.vectorize(profilvalka)
print(x[0][0])
y10 = profilvalka1(x10)*np.sin(u10)
z10 = profilvalka1(x10)*np.cos(u10)
x10-=130
z10 = z10 + profilvalka(130)

for i in range(len(x10)):
    for k in range(len(x10[i])):
        a1 = x10[i][k]*math.cos(alpha)+z10[i][k]*math.sin(alpha)
        b1 = x10[i][k]*(-1)*math.sin(alpha)+ z10[i][k]*math.cos(alpha)
        x10[i][k] = a1
        z10[i][k] = b1


for i in range(len(x10)):
    for k in range(len(x10[i])):
        a1 = x10[i][k]*math.cos(betta)-y10[i][k]*math.sin(betta)
        b1 = x10[i][k]*math.sin(betta)+y10[i][k]*math.cos(betta)
        x10[i][k] = a1
        y10[i][k] = b1


z10+=52.5

cv = 240*math.pi/180


for i in range(len(y10)):
    for k in range(len(y10[i])):
        a1 = y10[i][k]*math.cos(cv)-z10[i][k]*math.sin(cv)
        b1 = y10[i][k]*math.sin(cv)+z10[i][k]*math.cos(cv)
        y10[i][k] = a1
        z10[i][k] = b1

a20 = np.array([0,130,220,270])
#a = np.linspace(0,270,100)
b20 = np.linspace(0,2*np.pi,100)

u20, x20 =np.meshgrid(b20,a20)

#x , u = np.mgrid[0:271:3j, 0:2*np.pi:100j]

profilvalka1 = np.vectorize(profilvalka)
print(x[0][0])
y20 = profilvalka1(x20)*np.sin(u20)
z20 = profilvalka1(x20)*np.cos(u20)
x20-=130
z20 = z20 + profilvalka(130)

for i in range(len(x20)):
    for k in range(len(x20[i])):
        a1 = x20[i][k]*math.cos(alpha)+z20[i][k]*math.sin(alpha)
        b1 = x20[i][k]*(-1)*math.sin(alpha)+ z20[i][k]*math.cos(alpha)
        x20[i][k] = a1
        z20[i][k] = b1


for i in range(len(x20)):
    for k in range(len(x20[i])):
        a1 = x20[i][k]*math.cos(betta)-y20[i][k]*math.sin(betta)
        b1 = x20[i][k]*math.sin(betta)+y20[i][k]*math.cos(betta)
        x20[i][k] = a1
        y20[i][k] = b1


z20+=52.5

cv = 120*math.pi/180


for i in range(len(y20)):
    for k in range(len(y20[i])):
        a1 = y20[i][k]*math.cos(cv)-z20[i][k]*math.sin(cv)
        b1 = y20[i][k]*math.sin(cv)+z20[i][k]*math.cos(cv)
        y20[i][k] = a1
        z20[i][k] = b1



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

rads = []
by1 = []
by2 = []

for i in range(len(xs)):
    rads.append(np.sqrt(ys[i]**2+zs[i]**2))
    by2.append(105)


int1 = interp1d(xs,rads)

xs1 = np.linspace(xs[0],0,10)
for i in range(len(xs1)):
    by1.append(65)
by1 = np.array(by1)
by3 = np.minimum(by1,int1(xs1))

a2 = xs1
b2 = np.linspace(0,2*np.pi,100)

u2, x2 =np.meshgrid(b2,a2)

#x , u = np.mgrid[0:271:3j, 0:2*np.pi:100j]

def whereradd(xss):
    for i in range(len(by3)):
        if xss == xs1[i]:
            return by3[i]

whereradd = np.vectorize(whereradd)


y22 = whereradd(x2)*np.sin(u2)
z2 = whereradd(x2)*np.cos(u2)


a33 = np.linspace(0, 400,2)
b33 = np.linspace(0,2*np.pi,100)

u33, x33 =np.meshgrid(b33,a33)

y33 = 52.5*np.sin(u33)
z33 = 52.5*np.cos(u33)

a44 = np.linspace(-400, xs1[0],2)
b44 = np.linspace(0,2*np.pi,100)

u44, x44 =np.meshgrid(b44,a44)

y44 = 65*np.sin(u33)
z44 = 65*np.cos(u33)


fig = plt.figure()







axes = Axes3D(fig)
#vb = np.linspace(xs[0],xs[-1], 10)


#axes.plot(xs, ys, zs, label='parametric curve')
axes.plot_surface(x2, y22, z2, rstride=1, cstride=1, color='orange')
axes.plot_surface(x33, y33, z33, rstride=1, cstride=1, color='orange')
axes.plot_surface(x44, y44, z44, rstride=1, cstride=1, color='orange')
axes.plot_surface(x10, y10, z10, rstride=1, cstride=1, color='blue')
axes.plot_surface(x20, y20, z20, rstride=1, cstride=1,color='blue')
axes.plot_surface(x, y, z, rstride=1, cstride=1, color='blue')
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')
plt.show()
