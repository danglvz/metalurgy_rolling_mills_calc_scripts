import xlrd, xlwt
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp1d

class coordinats(object):
    def __init__(self, number):
        self.number = int(number)
        self.x = []
        self.y = []
        self.z = []
        self.radius = []
        self.velocity_z = []
        self.velocity_radial = []
        self.velocity_axis = []


    def get_coords(self, number_cell, x_cell, y_cell, z_cell, velocity_z_cell, velocity_x_cell, velocity_y_cell):
        if int(number_cell) == self.number:
            a = str(x_cell)
            n = ""
            n = n + a[:-5]

            self.x.append(float(n) * float(10 ** float(a[-4:])))
            a = str(y_cell)
            n = ""
            n = n + a[:-5]

            self.y.append(float(n) *float(10 ** float(a[-4:])))
            a = str(z_cell)
            n = ""
            n = n + a[:-5]

            self.z.append(float(n) * float(10 ** float(a[-4:])))

            a = str(velocity_z_cell)
            n = ""
            n = n + a[:-5]

            self.velocity_z.append(float(n) * float(10 ** float(a[-4:])))

            a1 =  str(velocity_x_cell)
            n1 = ""
            n1 = n1 + a1[:-5]

            a2 = str(velocity_y_cell)
            n2 = ""
            n2 = n2 + a2[:-5]

            self.velocity_radial.append(math.sqrt((float(n) * float(10 ** float(a[-4:])))**2+(float(n) * float(10 ** float(a[-4:])))**2))

            self.radius.append(math.sqrt(self.x[-1]**2+self.y[-1]**2))
            self.velocity_axis.append(self.velocity_radial[-1]/self.radius[-1])

    def getresultradius(self):
        return self.radius





#открываем файл
rb = xlrd.open_workbook('DEF_PTRnewlast122.xls',formatting_info=True)

#выбираем активный лист
sheet = rb.sheet_by_index(0)

x1 = []
y1 = []
z1 = []
x2 = []
y2 = []
z2 = []
x3 = []
y3 = []
z3 = []
x4 = []
y4 = []
z4  = []
c1 = coordinats(1)
c2 = coordinats(2)
c3 = coordinats(3)
c4 = coordinats(4)
c5 = coordinats(5)
c6 = coordinats(6)
for i in range(15,3866):
    c1.get_coords(sheet.row_values(i)[0],sheet.row_values(i)[7],sheet.row_values(i)[8],sheet.row_values(i)[9],sheet.row_values(i)[64],sheet.row_values(i)[62],sheet.row_values(i)[63])
    c2.get_coords(sheet.row_values(i)[0],sheet.row_values(i)[7],sheet.row_values(i)[8],sheet.row_values(i)[9],sheet.row_values(i)[64],sheet.row_values(i)[62],sheet.row_values(i)[63])
    c3.get_coords(sheet.row_values(i)[0],sheet.row_values(i)[7],sheet.row_values(i)[8],sheet.row_values(i)[9],sheet.row_values(i)[64],sheet.row_values(i)[62],sheet.row_values(i)[63])
    c4.get_coords(sheet.row_values(i)[0],sheet.row_values(i)[7],sheet.row_values(i)[8],sheet.row_values(i)[9],sheet.row_values(i)[64],sheet.row_values(i)[62],sheet.row_values(i)[63])
    c5.get_coords(sheet.row_values(i)[0],sheet.row_values(i)[7],sheet.row_values(i)[8],sheet.row_values(i)[9],sheet.row_values(i)[64],sheet.row_values(i)[62],sheet.row_values(i)[63])
    c6.get_coords(sheet.row_values(i)[0],sheet.row_values(i)[7],sheet.row_values(i)[8],sheet.row_values(i)[9],sheet.row_values(i)[64],sheet.row_values(i)[62],sheet.row_values(i)[63])







x1 = np.array(c1.x)
y1 = np.array(c1.y)
z1 = np.array(c1.z)


y2 = np.array(y2)
z2 = np.array(z2)

x3 = np.array(x3)
y3 = np.array(y3)
z3 = np.array(z3)

x4 = np.array(x4)
y4 = np.array(y4)
z4 = np.array(z4)
print(c1.radius)
print(c1.velocity_axis)

velocity_z_raspr_z = []
velocity_z_raspr_x = []
velocity_z_raspr_y = []

n = []

velocity_z_raspr_z.append(c1.z)
velocity_z_raspr_z.append(c2.z)
velocity_z_raspr_z.append(c3.z)
velocity_z_raspr_z.append(c4.z)
velocity_z_raspr_z.append(c5.z)
velocity_z_raspr_z.append(c6.z)

velocity_z_raspr_x.append(c1.getresultradius())
velocity_z_raspr_x.append(c2.getresultradius())
velocity_z_raspr_x.append(c3.getresultradius())
velocity_z_raspr_x.append(c4.getresultradius())
velocity_z_raspr_x.append(c5.getresultradius())
velocity_z_raspr_x.append(c6.getresultradius())

velocity_z_raspr_y.append(c1.velocity_z)
velocity_z_raspr_y.append(c2.velocity_z)
velocity_z_raspr_y.append(c3.velocity_z)
velocity_z_raspr_y.append(c4.velocity_z)
velocity_z_raspr_y.append(c5.velocity_z)
velocity_z_raspr_y.append(c6.velocity_z)


betta = []
for i in range(1,len(c1.x)):
    betta.append((math.acos(math.sqrt((c1.x[i]-c1.x[i-1])**2+(c1.y[i]-c1.y[i-1])**2)/(math.sqrt((c1.x[i]-c1.x[i-1])**2+(c1.y[i]-c1.y[i-1])**2+(c1.z[i]-c1.z[i-1])**2))))*180/math.pi)

f1 = interp1d(c1.z,c1.velocity_z, kind='cubic')
f2 = interp1d(c6.z,c6.velocity_z, kind='cubic')
f3 = interp1d(c1.z,c1.radius, kind='cubic')
f4 = interp1d(c1.z, c1.x, kind='cubic')
f5 = interp1d(c2.z, c2.x, kind='cubic')
f6 = interp1d(c3.z, c3.x, kind='cubic')
f7 = interp1d(c4.z,c4.x,kind='cubic')
f8 = interp1d(c5.z, c5.x, kind='cubic')
f9 = interp1d(c6.z,c6.x, kind='cubic')
f10 = interp1d(c1.z, c1.velocity_axis ,kind='cubic')
fig = plt.figure()
axes = Axes3D(fig)

axes.plot(np.array(c1.z), np.array(c1.x), np.array(c1.y), label='parametric curve1')
#axes.plot(np.array(c2.x), np.array(c2.y), np.array(c2.z), label='parametric curve')
#axes.plot(np.array(c3.x), np.array(c3.y), np.array(c3.z), label='parametric curv3e')
#axes.plot(np.array(c4.x), np.array(c4.y), np.array(c4.z), label='parametricdd curve')
#.plot(np.array(c5.x), np.array(c5.y), np.array(c5.z), label='kp')




plt.xlabel('продольная координата очага деформации')

#ffg = np.linspace(c1.z[0], c1.z[-1],10)
#plt.plot(ffg, f10(ffg))
plt.show()

print(betta)
