import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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


alpha = 45*math.pi/180
betta = 45*math.pi/180

def proektsia_na_os(x_valka, ang_valka):
    ang_valka = ang_valka*math.pi/180
    r_valka = profilvalka(x_valka)
    x0 = x_valka*math.cos(alpha)*math.cos(betta)
    xkontakta0 = 130*math.cos(alpha)*math.cos(betta)-profilvalka(130)*math.sin(alpha)*math.cos(betta)
    xkontakta = x0 + r_valka*math.cos(ang_valka)*math.sin(alpha)*math.cos(betta)-r_valka*math.sin(ang_valka)*math.sin(betta)
    ykontakta = r_valka*math.sin(ang_valka)*math.cos(betta)+((r_valka-r_valka*math.cos(ang_valka))+r_valka)*math.sin(alpha)*math.sin(betta)-(xkontakta0-(x0-r_valka*math.sin(alpha)*math.cos(betta)))*math.tan(betta)
    zkontakta = r_valka*math.cos(ang_valka)*math.cos(alpha)+(130-x_valka)*math.sin(alpha)-r_valka*math.cos(alpha)+profilvalka(130)*math.cos(alpha)

    return xkontakta, ykontakta, zkontakta

v = list(range(0, 361, 10))

x = []
y = []
z = []

for k in range(0, 271, 10):
    tapex = []
    tapey = []
    tapez = []
    for l in v:
        xn, yn, zn = proektsia_na_os(k, l)
        print(zn)
        tapex.append(xn)
        tapey.append(yn)
        tapez.append(zn)

    x.append(tapex)
    y.append(tapey)
    z.append(tapez)
x = np.array([np.array(i) for i in x])
y = np.array([np.array(i) for i in y])
z = np.array([np.array(i) for i in z])




fig = plt.figure()
axes = Axes3D(fig)
axes.plot_surface(x, y, z, rstride=1, cstride=1)

plt.show()
