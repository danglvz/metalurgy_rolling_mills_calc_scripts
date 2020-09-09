for i in y:
    for k in range(len(i)):
        i[k] = i[k]*0+i[k]*1+i[k]*0

for i in z:
    for k in range(len(i)):
        i[k] = i[k]*(math.sin(alpha))*(-1)+i[k]*0+i[k]*math.cos(alpha)
for i in range(len(x)):
    for k in range(len(x[i])):
        x[i][k] = np.sqrt(profilvalka1(x[i][k])*profilvalka1(x[i][k])- (y[i][k]-0)*(y[i][k]-0)-(z[i][k]+x[i][k]*np.sin(alpha))*(z[i][k]+x[i][k]*np.sin(alpha))) + (x[i][k])*np.cos(alpha)
