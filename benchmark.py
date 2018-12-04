from math import sin, pi

x1 = [8*sin(i*pi/32) for i in range(0,7)]
x2 = [16*sin(i*pi/32) for i in range(25,32)]
x3 = [x1[i]+x2[i] for i in range(len(x1))]
x4 = [x3[i]-x2[i] for i in range(len(x3))]
print(x1)
print(x2)
print(x3)
print(x4)