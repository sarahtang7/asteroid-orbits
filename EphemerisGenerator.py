# Ephemeris Generator - takes in 6 orbital elements and returns RA and DEC of object (at time-specific moment)
# Sarah Tang
# Created 7/4/2019

import math
from math import *

e = 0.6587595515873473 #eccentricity
a = 3.092704185336301 #semi-major axis (AU)
I = radians(11.74759129647092) #inclination (radians)
capOmega = radians(82.15763948051409) #radians
lowOmega = radians(356.34109239) #radians
Moriginal = radians(0.01246738682149958) #degrees

#calculate mean motion (n)
sqrtmu = 0.01720209895
mu = sqrtmu**2
n = math.sqrt(mu/a**3)

#calculate M
t = 2458668.5 #JD
toriginal = 2458465.5 #JD
M = Moriginal + n*(t-toriginal)

#calculate E - converge on a value
def kepler(M):
    E = M
    Mguess = E - e*sin(E)
    Mreal = M
    while abs(Mguess - Mreal) > 1e-004:
        Mguess = E - e*sin(E)
        E = E - (Mreal - (E - e*sin(E))) / (e*cos(E)-1)
    return E

#calculate x and y (physics coordinates)
x = a*cos(kepler(M)) - e*a
y = (a*sin(kepler(M)))*(sqrt(1-e**2))
r = sqrt(x**2 + y**2)
##nu = acos(x/r)
sinnu = y/r
cosnu = x/r
nu = atan2(sinnu, cosnu)

#calculate ecliptic coordinates
eclipticX = r*(cos(nu+lowOmega)*cos(capOmega) - cos(I)*sin(nu+lowOmega)*sin(capOmega))
eclipticY = r*(cos(I)*cos(capOmega)*sin(nu+lowOmega) + cos(nu+lowOmega)*sin(capOmega))
eclipticZ = r*sin(I)*sin(nu+lowOmega)

#get ecliptic Earth to Asteroid vectors
sunX = -2.027873566936922e-01
sunY = 9.963238789875005e-01
sunZ = -4.453100906916791e-05
rowXecliptic = eclipticX + sunX
rowYecliptic = eclipticY + sunY
rowZecliptic = eclipticZ + sunZ

#convert to equatorial coordinates
ob = radians(23.4358) #obliquity
equatorialX = (rowXecliptic)
equatorialY = rowYecliptic*cos(ob) - rowZecliptic*sin(ob)
equatorialZ = rowYecliptic*sin(ob) + rowZecliptic*cos(ob)
magnitude = sqrt(equatorialX**2 + equatorialY**2 + equatorialZ**2)
normEqX = equatorialX / magnitude
normEqY = equatorialY / magnitude
normEqZ = equatorialZ / magnitude

#calculate RA and DEC in degrees
dec = asin(normEqZ) #radians
sinra = normEqY/cos(dec)
cosra = normEqX/cos(dec)
ra = atan2(sinra, cosra)
ra = ra/pi*180

#convert RA from decimal degrees to hh:mm:ss.ss
RAobjecthr = ra/360*24
RAobjectmin = (RAobjecthr % int(RAobjecthr))*60
RAobjectsec = (RAobjectmin % int(RAobjectmin))*60

#convert DECobject from decimal degrees to dd:mm:ss.s
DECobjectDeg = dec/pi*180
DECobjectArcMin = (DECobjectDeg % int(DECobjectDeg))*60
DECobjectArcSec = (DECobjectArcMin % int(DECobjectArcMin))*60

print("RA: ", int(RAobjecthr), ":", int(RAobjectmin), ":", RAobjectsec, "hh:mm:ss.ss")
print("DEC: ", int(DECobjectDeg), ":", int(DECobjectArcMin), ":", DECobjectArcSec, "dd:mm:ss.ss")
