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

a1 = 292
b1 = 300
c1 = 272.8377777777778
d1 = 241.925

def profilvalka(x_valka, a1,b1,c1,d1):
    r1 = a1/2
    r2 = b1/2
    r3 = c1/2
    r4 = d1/2

    #ang1 = 1.77*math.pi/180

    if r1 > r2:
        ang1 = math.atan((r1 - r2) / 130)
    else:
        ang1 = math.atan((r2 - r1) / 130)

    if r2 > r3:
        ang2 = math.atan((r2 - r3) / 90)
    else:
        ang2 = math.atan((r2 - r3) / 90)
    if r3>r4:
        ang3 = math.atan((r3-r4)/50)
    else:
        ang3 = math.atan((r4 - r3) / 50)

    if 0 <= x_valka <= 130:
        if r1>r2:
            return float(r1 - x_valka*math.tan(ang1))
        else:
            return float(r1 + x_valka * math.tan(ang1))

    if 130 < x_valka <= 220:
        if r2>r3:
            return float(r2 - (x_valka-130)*math.tan(ang2))
        else:
            return float(r2 + (x_valka - 130) * math.tan(ang2))

    if 220 < x_valka <= 270:
        if r3>r4:
            return float(r3 - (x_valka-220)*math.tan(ang3))
        else:
            return float(r3 + (x_valka - 220) * math.tan(ang3))


alpha = 10*math.pi/180
betta = 18*math.pi/180


#a = np.array([0,130,220,270])

#x , u = np.mgrid[0:271:3j, 0:2*np.pi:100j]
for b in range(100):
    space = np.linspace(25,30,100)
    a11 = 292
    b11 = 300
    c1 = 272.8377777777778
    d1 = 261.9910660660661
    a11 = a11+space[b]
    print(space[b])
    profilvalka1 = np.vectorize(profilvalka)
    a = np.array([0, 130, 220, 270])
    #a = np.linspace(0, 270, 100)
    b = np.linspace(2.7, 3.7, 1000)

    u, x = np.meshgrid(b, a)
    print(x[0][0])

    y = profilvalka1(x,a11,b11,c1,d1)*np.sin(u)
    z = profilvalka1(x,a11,b11,c1,d1)*np.cos(u)
    x-=130
    z = z + profilvalka(130,a11,b11,c1,d1)

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
    for i in range(len(xs)):
        rads.append(np.sqrt(ys[i] ** 2 + zs[i] ** 2))
    print(rads[0])
    print(a11)
    if float(rads[0])>float(69.5) and float(rads[0]<float(70)):
        print(a11)
        break

by1 = []
by2 = []
rads = []
for i in range(len(xs)):
    rads.append(np.sqrt(ys[i]**2+zs[i]**2))
    by2.append(105)

fig, ax = plt.subplots()

by2 = np.array(by2)
rads = np.array(rads)

int1 = interp1d(xs,rads)
xs1 = np.linspace(xs[0],0,50)
for i in range(len(xs1)):
    by1.append(65)
by1 = np.array(by1)
by3 = np.minimum(by1,int1(xs1))

for i in range(len(by3)):
    if by3[i]!=65:
        print(xs1[i])
        break

ax.fill_between(xs1,by3,color='green', alpha='0.7')
xy4 = np.linspace(0,150,2)
by4 = []
for i in range(len(xy4)):
    by4.append(52.5)
by4 = np.array(by4)
ax.fill_between(xy4,by4,color='green', alpha='0.7')
#fig.set_facecolor('floralwhite')

plt.plot(xs,np.array(rads))
plt.grid()
plt.yticks(np.arange(0,90,10))
plt.xticks(np.arange(-130,150,26))
ax.set_xlabel('X (мм)')
ax.set_ylabel('Расстояние до оси прокатки (мм)')


#fig = plt.figure()







#axes = Axes3D(fig)
#vb = np.linspace(xs[0],xs[-1], 10)

#axes.plot_surface(x, y, z, rstride=1, cstride=1)
#axes.plot(xs, ys, zs, label='parametric curve')

plt.show()
