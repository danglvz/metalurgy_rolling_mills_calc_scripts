import xlrd, xlwt
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#открываем файл
rb = xlrd.open_workbook('15.xls',formatting_info=True)

#выбираем активный лист
sheet = rb.sheet_by_index(0)

#print(sheet.row_values(15)[1])

x = []
y = []
z = []

vel = []
x1 = []
y1 = []
z1 = []

vel1 =[]
for i in range(15,180):
    a = str(sheet.row_values(i)[7])
    n = ""
    n = a[:-5]

    y.append(float(n)*float(10**float(a[-4:]))-150)
    a = str(sheet.row_values(i)[8])
    n = ""
    n = a[:-5]

    z.append(float(n)*float(10**float(a[-4:])))
    a = str(sheet.row_values(i)[9])
    n = ""
    n = a[:-5]

    x.append((float(n)*float(10**float(a[-4:]))-140)*(-1))
    a = str(sheet.row_values(i)[64])
    n = ""
    n = a[:-5]

    vel.append(float(n)*float(10**float(a[-4:]))*(-1))

for i in range(180, 344):
    a = str(sheet.row_values(i)[7])
    n = ""
    n = a[:-5]

    y1.append(float(n) * float(10 ** float(a[-4:])) - 150)
    a = str(sheet.row_values(i)[8])
    n = ""
    n = a[:-5]

    z1.append(float(n) * float(10 ** float(a[-4:])))
    a = str(sheet.row_values(i)[9])
    n = ""
    n = a[:-5]

    x1.append((float(n) * float(10 ** float(a[-4:])) - 140)*(-1))
    a = str(sheet.row_values(i)[64])
    n = ""
    n = a[:-5]

    vel1.append(float(n) * float(10 ** float(a[-4:]))*(-1))

print(sheet.nrows)
betta = []
#for i in range(1,len(x)-1):
   # betta.append((math.acos(math.sqrt((y[i]-x[i-1])**2+(y[i]-y[i-1])**2)/(math.sqrt((x[i]-x[i-1])**2+(y[i]-y[i-1])**2+(z[i]-z[i-1])**2))))*180/math.pi)
radius = []
for i in  range(len(x)):
    radius.append(math.sqrt(y[i]**2+z[i]**2))

radius = np.array(radius)
radius = radius

print(x)
print(y)
print(z)



print(radius)


xmin = []
ymin = []
zmin = []

n = []

for i in range(len(x)):
    n.append(math.sqrt(x[i]**2+y[i]**2))



x = np.array(x)
y = np.array(y)
z = np.array(z)
vel = np.array(vel)
x1 = np.array(x1)
y1 = np.array(y1)
z1 = np.array(z1)
vel1 = np.array(vel1)


#plt.plot(x,vel)
#plt.plot(x1,vel1)
plt.grid()
# plot the data itself
plt.plot(x,vel,'o', label='продольная скорость точки на поверхности')

# calc the trendline (it is simply a linear fitting)
zzz = np.polyfit(x, vel, 1.5)
ppp = np.poly1d(zzz)

# plot the data itself
plt.plot(x1,vel1,'o',label='продольная скорость точки в центре прутка')

# calc the trendline (it is simply a linear fitting)
zzz1 = np.polyfit(x1, vel1, 1.5)
ppp1 = np.poly1d(zzz1)
plt.plot(x1,ppp1(x1),'r--', color='red',label='линия тренда для точки в центре прутка')
plt.plot(x,ppp(x),'r--',color='blue',label='линия тренда для точки на поверхности')

plt.legend()
plt.xlabel('X (мм)')
plt.ylabel('скорость мм/с')
#fig = plt.figure()
#axes = Axes3D(fig)

#axes.plot(x, y, z, label='parametric curve')
#axes.set_xlabel('X')
#axes.set_ylabel('Y')
#axes.set_zlabel('Z')
plt.show()