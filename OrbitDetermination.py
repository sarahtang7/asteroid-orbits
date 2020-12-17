#ORBIT DETERMINATION CODE WITH MONTE CARLO & VPYTHON SIMULATION (w/ additional Gaussian Distribution Histograms)
#SARAH TANG
#SATURDAY, JULY 13, 2019

import astropy as astropy
from astropy.io import ascii
import numpy as np
import math
from math import *
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as ss
import vpython
from vpython import vector,color
from vpython import canvas

#read in the text file
data = np.loadtxt('Tang.txt')

#ask user for center data point - for a file with 4 data points
print('Number of data points: ', len(data))
center = int(input('Type the central data point you would like to use (2, 3): ')) #when you use this, subtract 1 to get index
RAerror = [0.6, 0.5, 0.6, 0.4] #in arcseconds
DECerror = [0.6, 0.6, 0.7, 0.7] #in arcseconds
print()
monteCarlo = input('Apply Monte Carlo? Type \'yes\' or \'no\'. ') #to not print all values out when doing Monte Carlo
if monteCarlo=="yes":
    monteCarlo=True
elif monteCarlo=="no":
    monteCarlo=False
indices = [0, center-1, 3]
print()
vPythonVis = input('Simulate the orbit of this object? Type \'yes\' or \'no\'. ')
print()
if vPythonVis=="yes":
    vPythonVis=True
elif vPythonVis=="no":
    vPythonVis=False

#make new data arrays using user-input center (3 lines)
year = []
month = []
day = []
hour = []
mins = []
sec = []
RAhh = []
RAmm = []
RAss = []
DECdd = []
DECam = []
DECas = []
sunX = []
sunY = []
sunZ = []

for i in indices:
    year.append(data[i,0])
    month.append(data[i,1])
    day.append(data[i,2])
    hour.append(data[i,3])
    mins.append(data[i,4])
    sec.append(data[i,5])
    RAhh.append(data[i,6])
    RAmm.append(data[i,7])
    RAss.append(data[i,8])
    DECdd.append(data[i,9])
    DECam.append(data[i,10])
    DECas.append(data[i,11])
    sunX.append(data[i,12])
    sunY.append(data[i,13])
    sunZ.append(data[i,14])

#convert time of observation from hh:mm:ss to JD
timeJD = []
for row in range(len(year)):
    a1 = int(((month[row])+9)/12)
    a2 = (year[row] + a1)*(7/4)
    term2 = int(a2)
    term3 = int(275*month[row]/9)
    J0 = (367*year[row]) - term2 + term3 + int(day[row]) + 1721013.5
    UThours = hour[row]
    UTmins = mins[row]
    UTsec = sec[row]
    UThrdd = UThours + UTmins/60 + UTsec/3600 #fractional hour
    JD = J0 + UThrdd/24
    timeJD.append(JD)

#convert RA and DEC from degrees to radians
DECradList = []
RAradList = []
eclip = radians(23.4358) #rad
for row in range(len(year)):
    RArad = radians((RAss[row]/3600 + RAmm[row]/60 + RAhh[row])*15)
    DECrad = radians(DECdd[row] + DECam[row]/60 + DECas[row]/3600)
    RAradList.append(RArad)
    DECradList.append(DECrad)

#convert sun vectors from equatorial to ecliptic
obmatrix = np.array([[1, 0, 0],
                     [0, cos(eclip), sin(eclip)],
                     [0, -sin(eclip), cos(eclip)]])
sun1MatrixEQ = np.array([[sunX[0]], #for obs1
                       [sunY[0]],
                       [sunZ[0]]])
sun2MatrixEQ = np.array([[sunX[1]], #for obs2
                       [sunY[1]],
                       [sunZ[1]]])
sun3MatrixEQ = np.array([[sunX[2]], #for obs3
                       [sunY[2]],
                       [sunZ[2]]])
obs1sunEC = np.squeeze(np.matmul(obmatrix, sun1MatrixEQ)) #ECLIPTIC SUN VECTORS
obs2sunEC = np.squeeze(np.matmul(obmatrix, sun2MatrixEQ))
obs3sunEC = np.squeeze(np.matmul(obmatrix, sun3MatrixEQ))
    
#calculate A1, D21, D22, A3, D23, B1, B3, D0 - for A and B 
k = 0.01720209895
T3 = k*(timeJD[2]-timeJD[1])
T1 = k*(timeJD[0]-timeJD[1])
T = T3 - T1
A1 = T3/T 
A3 = -T1/T 
B1 = (1/6)*A1*(T**2 - T3**2)
B3 = (1/6)*A3*(T**2 - T1**2)

#initialize lists used in Monte Carlo to keep track of tweaked 6 orbital elements
mcSemMaj = []
mcEcc = []
mcInc = []
mcLAN = []
mcPer = []
mcMA = []
mcOP = []
mcE = []
mcmm = []
mcT = []

goodSemMaj = []
goodEcc = []
goodInc = []
goodLAN = []
goodPer = []
goodMA = []
goodOP = []
goodEA = []
goodmm = []
goodT = []
goodr2Pos = []
goodr2Vel = []

#orbital elements from JPL to calculate percent uncertainties
##for central observation 2 at UT 06:20:04 on 6/23/2019
aJPL2 = 1.541850293016061 #semi-major axis in AU
eJPL2 = 4.060232942826339e-01 #eccentricity
iJPL2 = 2.452648721537227e+01 #inclination in degrees
longJPL2 = 2.207453277800303E+02 #longitude of acsending node in degrees
maJPL2 = 3.674634977296745E+01 #mean anomaly in degrees
mmJPL2 = 5.148026364980952E-01*pi/180 #mean motion in radians/day
TJPL2 = 2458586.384398638271 #time of last perihelion passage (periapsis)
opJPL2 = 6.992971179185715E+02/365.25 #orbital period in years
##for central observation 3 at UT 07:40:27 on 7/05/2019
aJPL3 = 1.541852818474723 #semi-major axis in AU
eJPL3 = 4.060258137235424E-01 #eccentricity
iJPL3 = 2.452630543952999E+01 #inclination in degrees
longJPL3 = 2.207448929673566E+02 #longitude of acsending node in degrees
maJPL3 = 4.295244207321888E+01 #mean anomaly in degrees
mmJPL3 = 5.148013716767426E-01*pi/180 #mean motion in radians/day
TJPL3 = 2458586.384464593139 #time of last perihelion passage (periapsis)
opJPL3 = 6.992988360296241E+02/365.25 #orbital period in years

#START MONTE CARLO LOOP - using 1000 iterations
iteration = 1
while iteration<=1000:

    if monteCarlo==True:
        #tweaked RA and DEC in radians
        RArad1 = np.random.normal(RAradList[0], RAerror[0]/3600*pi/180, 1) #first observation new RA
        RArad2 = np.random.normal(RAradList[1], RAerror[1]/3600*pi/180, 1) #center observation new RA
        RArad3 = np.random.normal(RAradList[2], RAerror[2]/3600*pi/180, 1) #last observation new RA
        DECrad1 = np.random.normal(DECradList[0], DECerror[0]/3600*pi/180, 1) #first observation new DEC
        DECrad2 = np.random.normal(DECradList[1], DECerror[1]/3600*pi/180, 1) #center observation new DEC
        DECrad3 = np.random.normal(DECradList[2], DECerror[2]/3600*pi/180, 1) #last observation new DEC

    if monteCarlo==False:
        RArad1 = RAradList[0]
        RArad2 = RAradList[1]
        RArad3 = RAradList[2]
        DECrad1 = DECradList[0]
        DECrad2 = DECradList[1]
        DECrad3 = DECradList[2]

    #calculate rho hats - these are based on tweaked RA and DEC values for Monte Carlo
    rhoHat1 = np.squeeze(np.array([[cos(RArad1)*cos(DECrad1)],
                        [sin(RArad1)*cos(DECrad1)],
                        [sin(DECrad1)]]))
    rhoHat2 = np.squeeze(np.array([[cos(RArad2)*cos(DECrad2)],
                        [sin(RArad2)*cos(DECrad2)],
                        [sin(DECrad2)]]))
    rhoHat3 = np.squeeze(np.array([[cos(RArad3)*cos(DECrad3)],
                        [sin(RArad3)*cos(DECrad3)],
                        [sin(DECrad3)]]))

    #convert rho hats from equatorial to ecliptic
    oblMatrix = np.array([[1, 0, 0],
                         [0, cos(eclip), sin(eclip)],
                         [0, -sin(eclip), cos(eclip)]])
    rhoHat1EC = np.matmul(oblMatrix, rhoHat1) #for observation 1
    rhoHat2EC = np.matmul(oblMatrix, rhoHat2) #for observation 2
    rhoHat3EC = np.matmul(oblMatrix, rhoHat3) #for observation 3

    D11 = np.dot(np.cross(obs1sunEC, rhoHat2EC), rhoHat3EC)
    D12 = np.dot(np.cross(obs2sunEC, rhoHat2EC), rhoHat3EC)
    D13 = np.dot(np.cross(obs3sunEC, rhoHat2EC), rhoHat3EC)

    D21 = np.dot(np.cross(rhoHat1EC, obs1sunEC), rhoHat3EC)
    D22 = np.dot(np.cross(rhoHat1EC, obs2sunEC), rhoHat3EC)
    D23 = np.dot(np.cross(rhoHat1EC, obs3sunEC), rhoHat3EC)

    D31 = np.dot(rhoHat1EC, np.cross(rhoHat2EC, obs1sunEC))
    D32 = np.dot(rhoHat1EC, np.cross(rhoHat2EC, obs2sunEC))
    D33 = np.dot(rhoHat1EC, np.cross(rhoHat2EC, obs3sunEC))

    D0 = np.dot(rhoHat1EC, np.cross(rhoHat2EC, rhoHat3EC))

    #calculate A, B, E, and F - for r2 position vector
    A = (A1*D21 - D22 + A3*D23)/-D0
    B = (B1*D21 + B3*D23)/-D0
    E = -2*np.dot(rhoHat2EC, obs2sunEC)
    F = float(obs2sunEC[0]**2 + obs2sunEC[1]**2 + obs2sunEC[2]**2)

    #calculate a, b, c - for r2 position vector
    mu = 1 #mu is equal to 1 in gaussian time
    a = -1*(A**2 + A*E + F)
    b = -1*(2*A*B + B*E) #mu=1
    c = -1*(B**2) #mu=1 

    #plug a, b, and c into the scalar equation of Lagrange to find all real roots
    coeffArray = [c, 0, 0, b, 0, 0, a, 0, 1]
    r2roots = np.polynomial.polynomial.polyroots(coeffArray)

    #get all valid roots
    rho2Test = []
    r2ValidRoots = []
    for i in range(len(r2roots)):
        rho2forTest = A + ((1*B)/r2roots[i]**3)
        if rho2forTest > 0:
            if np.imag(r2roots[i])==0:
                if r2roots[i]>0:
                    r2RealValid = np.real(r2roots[i])
                    r2ValidRoots.append(r2RealValid)

    #ask user for root to iterate || for Monte Carlo, use the first root
    if monteCarlo==False:
        print('Valid Roots for r2: ', r2ValidRoots)
        root = input('Type the index of the root you would like to use (0, 1, 2): ')
    if monteCarlo==True:
        root = 0
    
    r2guess = r2ValidRoots[int(root)]
    r2compare = [0,0,0] #initialize loop for the convergence loop

    #solve for initial f1, f3, g1, g3 
    u = 1 / r2guess**3
    f1 = 1 - (0.5*u*(T1**2))
    f3 = 1 - (0.5*u*(T3**2))
    g1 = T1-((u/6)*(T1**3))
    g3 = T3-((u/6)*(T3**3))

    #solve for initial c1, c2, c3
    c1 = g3 / (f1*g3 - g1*f3)
    c2 = -1
    c3 = -g1 / (f1*g3 - g1*f3)

    #use c1, c2, c3 to get rho1, rho2, rho3
    rho1 = (c1*D11 + c2*D12 + c3*D13)/(c1*D0)
    rho2 = (c1*D21 + c2*D22 + c3*D23)/(c2*D0)
    rho3 = (c1*D31 + c2*D32 + c3*D33)/(c3*D0)

    #calculate first position vectors & first velocity vector
    r1PosVect = rho1*rhoHat1EC - obs1sunEC
    r2PosVect = rho2*rhoHat2EC - obs2sunEC
    r3PosVect = rho3*rhoHat3EC - obs3sunEC
    d1 = -f3 / (f1*g3 - g1*f3)
    d3 = f1 / (f1*g3 - g1*f3)
    r2VelVect = d1*r1PosVect + d3*r3PosVect

    #print out initial position and velocity vectors
    if monteCarlo==False:
        print()
        print('INITIAL POSITION AND VELOCITY VECTORS FOR THE CENTRAL OBSERVATION - NO MONTE CARLO:')
        print('Initial r2 Position Vector: ', r2PosVect)
        print('Initial r2 Velocity Vector: ', r2VelVect)
        print()

    #start iteration to converge on r2 position and r2 velocity vectors
    while abs(r2compare[0]-r2PosVect[0])>1e-5:
        r2compare = r2PosVect
        
        #calculate new Tau values - with light correction
        c = 3e8 * 86400 / 1.496e11 #speed of light in AU/day
        newTimes = [0, 0, 0]
        newTimes[0] = timeJD[0] - rho1/c #perform light correction on original times
        newTimes[1] = timeJD[1] - rho2/c #perform light correction on original times
        newTimes[2] = timeJD[2] - rho3/c #perform light correction on original times
        T1 = k*(newTimes[0]-newTimes[1])
        T3 = k*(newTimes[2]-newTimes[1])
        
        #use 4th power Taylor series expansion to get better f1, f3, g1, g3
        r2PosMag = sqrt(r2PosVect[0]**2 + r2PosVect[1]**2 + r2PosVect[2]**2)
        u = 1/(r2PosMag**3)
        z = np.dot(r2PosVect, r2VelVect) / r2PosMag**2
        q = (np.dot(r2VelVect, r2VelVect)/(r2PosMag**2)) - u
        f1 = 1-(u/2)*T1**2+((u*z)/2)*T1**3+((3*u*q-15*u*z**2+u**2)/24)*T1**4
        f3 = 1-(u/2)*T3**2+((u*z)/2)*T3**3+((3*u*q-15*u*z**2+u**2)/24)*T3**4
        g1 = T1-(u/6)*T1**3+((u*z)/4)*T1**4
        g3 = T3-(u/6)*T3**3+((u*z)/4)*T3**4

        #get better c1, c2, c3 from scalar range equations
        c1 = g3 / (f1*g3 - g1*f3)
        c2 = -1
        c3 = -g1 / (f1*g3 - g1*f3)

        #use c1, c2, c3 to get better rho1, rho2, rho3
        rho1 = (c1*D11 + c2*D12 + c3*D13)/(c1*D0)
        rho2 = (c1*D21 + c2*D22 + c3*D23)/(c2*D0)
        rho3 = (c1*D31 + c2*D32 + c3*D33)/(c3*D0)

        #calculate better position and velocity vectors
        r1PosVect = rho1*rhoHat1EC - obs1sunEC
        r2PosVect = rho2*rhoHat2EC - obs2sunEC
        r3PosVect = rho3*rhoHat3EC - obs3sunEC
        d1 = -f3 / (f1*g3 - g1*f3)
        d3 = f1 / (f1*g3 - g1*f3)
        r2VelVect = d1*r1PosVect + d3*r3PosVect

    #OUTPUT: position and velocity vectors for central observation in ecliptic rectangular coordinates
    #OUTPUT: range to the asteroid
    astMag = sqrt(r2PosVect[0]**2 + r2PosVect[1]**2 + r2PosVect[2]**2) 
    if monteCarlo==False:
        rho2forRange = A + ((mu*B)/astMag**3)
        print('ASTEROID\'S FINAL POSITION AND VELOCITY VECTORS - NO MONTE CARLO:')
        print('Position vector of central observation (ecliptic, AU): ', r2PosVect)
        print('Velocity vector of central observation (ecliptic, AU/day): ', r2VelVect)
        print()
        print('RANGE TO THE ASTEROID - NO MONTE CARLO')
        print('Range to the asteroid of central observation (AU):', rho2forRange)
        
    #OUTPUT: 6 orbital elements with respect to ecliptic plane
    time = timeJD[1] #use the time of the central observation
    ##calculate semi-major axis (a)
    magPosVector = sqrt(r2PosVect[0]**2 + r2PosVect[1]**2 + r2PosVect[2]**2)
    magVelVector = sqrt(r2VelVect[0]**2 + r2VelVect[1]**2 + r2VelVect[2]**2)
    mu = 1
    a = 1 / ( (2/magPosVector) - (np.dot(magVelVector, magVelVector)/mu) )
    ##calculate eccentricity (e)
    numer = np.cross(r2PosVect, r2VelVect)
    numerMag = sqrt(numer[0]**2 + numer[1]**2 + numer[2]**2) **2
    denom = mu*a
    e = sqrt(1 - (numerMag/denom))
    ##calculate inclination (i) 
    h = np.cross(r2PosVect, r2VelVect)
    hMag = sqrt(h[0]**2 + h[1]**2 + h[2]**2)
    i = acos(h[2] / hMag)
    i = i/pi*180
    ##calculate longitude of asceding node (Omega)
    sinLAN = h[0]/(hMag*sin(radians(i)))
    cosLAN = -h[1]/(hMag*sin(radians(i)))
    LAN = atan2(sinLAN, cosLAN)
    LAN = LAN/pi*180
    if LAN < 0:
        LAN += 360
    ##calculate argument of perihelion (omega)
    Unumer = r2PosVect[0]*cos(radians(LAN)) + r2PosVect[1]*sin(radians(LAN)) #radians
    cosU = Unumer/astMag #radians
    sinU = r2PosVect[2] / (astMag*sin(radians(i))) #radians
    U = atan2(sinU, cosU)
    dotU = np.dot(r2PosVect, r2VelVect)
    esinv = ((a*(1-e**2))/hMag) * (dotU/astMag)
    sinv = esinv / e
    v = asin(sinv) #gives degrees
    perihelion = U - v
    perihelion = perihelion/pi*180
    if perihelion<0:
        perihelion = 360+perihelion
    ##calculate mean motion (n)
    mu2 = 0.01720209895
    n = mu2 * sqrt(1/a**3)
    ##calculate mean anomaly (M) precessed to July 21, 2019 6:00 UT
    cosE = (1/e) * (1-(astMag/a))
    E = acos(cosE)
    M = E - e*sin(E)
    M = M/pi*180
    if v < 0:
        M = 360-M
    M0 = M + n*(2458685.75-timeJD[1])
    maJPLpre2 = maJPL2 + n*(2458685.75-timeJD[1]) #precess mean anomaly for JPL value - centered on obs2
    maJPLpre3 = maJPL3 + n*(2458685.75-timeJD[1]) #precess mean anomaly for JPL value - centered on obs3
    ##calculate eccentric anomaly (E)
    Edeg = E/pi*180
    if v<0:
        Edeg = 360-Edeg
    ##calculate time of last perihelion passage (T)
    if v<0:
        M = 360 - M
        M = M*pi/180
        T = time - (((2*pi)-M)/n)
    else:
        M = M*pi/180
        T = time - (M/n)
    ##calculate orbital period (P)
    k = 0.01720209895
    P = (2*pi*a**(3/2)) / k #days
    P = P / 365.25

    #print orbital elements - no Monte Carlo
    if monteCarlo==False:
        print()
        print('ORBITAL ELEMENTS - NO MONTE CARLO:')
        print("Semi-major Axis (a) = %0.6f AU" %a)
        print("Eccentricity (e) = %0.6f" %e)
        print("Inclination (i) = %0.6f degrees" %i)
        print("Longitude of Ascending Node (Omega) = %0.6f degrees" %LAN)
        print("Argument of Perihelion (omega) = %0.6f degrees" %perihelion)
        print("Mean Anomaly (M) = %0.6f degrees" %M0)
        print("Mean Motion (n) = %0.6f radians/day" %n)
        print("Eccentric Anomaly (E) = %0.6f degrees" %Edeg)
        print("Time of Last Perihelion Passage (T) = %0.5f" %T)
        print("Orbital Period (P) = %0.6f years" %P)
        
    #orbital elements from JPL - for percent uncertainties
    if monteCarlo==False:
        #for data points (1, 2, 4)
        if center==2: #central observation at UT 06:20:04 on 6/23/2019
            print()
            print('PERCENT UNCERTAINTY ON ORBITAL ELEMENTS - NO MONTE CARLO:')
            print('Percent uncertainty of semi-major axis (AU): %0.2f' %((abs(aJPL2-a)/aJPL2)*100), '%')
            print('Percent uncertainty of eccentricity: %0.2f' %((abs(eJPL2-e)/eJPL2)*100), '%')
            print('Percent uncertainty of inclination (degrees): %0.2f' %((abs(iJPL2-i)/iJPL2)*100), '%')
            print('Percent uncertainty of longitude of acsending node (degrees): %0.2f' %((abs(longJPL2-LAN)/longJPL2)*100), '%')
            print('Percent uncertainty of mean anomaly (degrees): %0.2f' %((abs(maJPLpre2-M0)/maJPLpre2)*100), '%') #precess mean anomaly
            print('Percent uncertainty of mean motion (radians/day): %0.2f' %((abs(mmJPL2-n)/mmJPL2)*100), '%')
            print('Percent uncertainty of time of last periapsis (JD): %0.6f' %((abs(TJPL2-T)/TJPL2)*100), '%')
            print('Percent uncertainty of orbital period (years): %0.2f' %((abs(opJPL2-P)/opJPL2)*100), '%')
        #for data points (1, 3, 4)
        if center==3: #central observation at UT 07:40:27 on 7/05/2019
            print()
            print('PERCENT UNCERTAINTY ON ORBITAL ELEMENTS - NO MONTE CARLO:')
            print('Percent uncertainty of semi-major axis (AU): %0.2f' %((abs(aJPL3-a)/aJPL3)*100), '%')
            print('Percent uncertainty of eccentricity: %0.2f' %((abs(eJPL3-e)/eJPL3)*100), '%')
            print('Percent uncertainty of inclination (degrees): %0.2f' %((abs(iJPL3-i)/iJPL3)*100), '%')
            print('Percent uncertainty of longitude of acsending node (degrees): %0.2f' %((abs(longJPL3-LAN)/longJPL3)*100), '%')
            print('Percent uncertainty of mean anomaly (degrees): %0.2f' %((abs(maJPLpre3-M0)/maJPLpre3)*100), '%') #precess mean anomaly
            print('Percent uncertainty of mean motion (radians/day): %0.2f' %((abs(mmJPL3-n)/mmJPL3)*100), '%')
            print('Percent uncertainty of time of last periapsis (JD): %0.6f' %((abs(TJPL3-T)/TJPL3)*100), '%')
            print('Percent uncertainty of orbital period (years): %0.2f' %((abs(opJPL3-P)/opJPL3)*100), '%')

        #append orbital elements to lists for vpython simulation without Monte Carlo
        goodSemMaj.append(a)
        goodEcc.append(e)
        goodInc.append(i)
        goodLAN.append(LAN)
        goodPer.append(perihelion)
        goodMA.append(M0)
        goodOP.append(P)
        goodEA.append(radians(Edeg))
        goodmm.append(n)
        goodT.append(T)
        break

    #apend all orbital element calculations to lists for Monte Carlo
    if monteCarlo==True:
        mcSemMaj.append(a)
        mcEcc.append(e)
        mcInc.append(i)
        mcLAN.append(LAN)
        mcPer.append(perihelion)
        mcMA.append(M0)
        mcOP.append(P) #orbital period for vpython
        mcE.append(Edeg) #Eccentric Anomaly in degrees for vpython
        mcmm.append(n)
        mcT.append(T)

    #FOR MONTE CARLO - use ephemeris generator to calculate error for orbital elements
    if monteCarlo==True:
        differences = [100] #100 is arbitrary
        ##calculate x and y (physics coordinates)
        x = a*cos(E) - e*a
        y = (a*sin(E))*(sqrt(1-e**2))
        r = sqrt(x**2 + y**2)
        sinnu = y/r
        cosnu = x/r
        nu = atan2(sinnu, cosnu)
        ##calculate ecliptic coordinates
        eclipticX = r*(cos(nu+perihelion)*cos(LAN) - cos(i)*sin(nu+perihelion)*sin(LAN))
        eclipticY = r*(cos(i)*cos(LAN)*sin(nu+perihelion) + cos(nu+perihelion)*sin(LAN))
        eclipticZ = r*sin(i)*sin(nu+perihelion)
        ##get ecliptic Earth to Asteroid vectors
        sunX = -2.027873566936922e-01
        sunY = 9.963238789875005e-01
        sunZ = -4.453100906916791e-05
        rowXecliptic = eclipticX + sunX
        rowYecliptic = eclipticY + sunY
        rowZecliptic = eclipticZ + sunZ
        ##convert to equatorial coordinates
        ob = radians(23.4358) #obliquity
        equatorialX = (rowXecliptic)
        equatorialY = rowYecliptic*cos(ob) - rowZecliptic*sin(ob)
        equatorialZ = rowYecliptic*sin(ob) + rowZecliptic*cos(ob)
        magnitude = sqrt(equatorialX**2 + equatorialY**2 + equatorialZ**2)
        normEqX = equatorialX / magnitude
        normEqY = equatorialY / magnitude
        normEqZ = equatorialZ / magnitude
        ##calculate RA and DEC in degrees
        dec = asin(normEqZ)/pi*180 #degrees
        sinra = normEqY/cos(dec)
        cosra = normEqX/cos(dec)
        ra = atan2(sinra, cosra)
        ra = ra/pi*180 #degrees

        diffyay = abs(ra-(RAradList[1]/pi*180))+abs(dec-(DECradList[1]/pi*180))
        if diffyay < np.min(differences):
            differences.append(diffyay)
            goodr2Pos.append(r2PosVect)
            goodr2Vel.append(r2VelVect)
            goodSemMaj.append(a)
            goodEcc.append(e)
            goodInc.append(i)
            goodLAN.append(LAN)
            goodPer.append(perihelion)
            goodMA.append(M0)
            goodOP.append(P)
            goodEA.append(radians(Edeg))
            goodmm.append(n)
            goodT.append(T)

    print('Monte Carlo Iteration (of 1000): ', iteration)
    iteration+=1 #add 1 to iteration

#print "best" orbital elements given by Monte Carlo
if monteCarlo==True:
    #print r2 position and r2 velocity vectors
    astMag = sqrt((goodr2Pos[-1])[0]**2 + (goodr2Pos[-1])[1]**2 + (goodr2Pos[-1])[2]**2) #rho2
    rho2forRange = A + ((mu*B)/astMag**3)
    print()
    print('ASTEROID\'S POSITION AND VELOCITY VECTORS - WITH MONTE CARLO:')
    print('Position vector of central observation (ecliptic, AU): ', goodr2Pos[-1])
    print('Velocity vector of central observation (ecliptic, AU/day): ', goodr2Vel[-1])
    print('Velocity vector of central observation (ecliptic, AU/year): ', goodr2Vel[-1]*365.25)
    print()
    print('RANGE TO THE ASTEROID - WITH MONTE CARLO')
    print('Range to the asteroid of central observation (AU):', rho2forRange)
    print()
    print('ORBITAL ELEMENTS - WITH MONTE CARLO:')
    print("Semi-major Axis (a) = %0.2f AU" %goodSemMaj[-1])
    print("Eccentricity (e) = %0.2f" %goodEcc[-1])
    print("Inclination (i) = %0.1f degrees" %goodInc[-1])
    print("Longitude of Ascending Node (Omega) = %0.1f degrees" %goodLAN[-1])
    print("Argument of Perihelion (omega) = %0.1f degrees" %goodPer[-1])
    print("Mean Anomaly (M) = %0.0f degrees" %goodMA[-1])

    ##calculate error of orbital elements (from monte carlo) for histogram -- standard deviation & full width half max
    stdSemMaj = np.std(mcSemMaj)
    stdEcc = np.std(mcEcc)
    stdInc = np.std(mcInc)
    stdLAN = np.std(mcLAN)
    stdPer = np.std(mcPer)
    stdMA = np.std(mcMA)

    fwhmSemMaj = 2.355*stdSemMaj #this is error range
    fwhmEcc = 2.355*stdEcc
    fwhmInc = 2.355*stdInc
    fwhmLAN = 2.355*stdLAN
    fwhmPer = 2.355*stdPer
    fwhmMA = 2.355*stdMA

    #print uncertainty of orbital elements given by monte carlo
    print()
    print('UNCERTAINTY OF ORBITAL ELEMENTS - WITH MONTE CARLO: ')
    print('Semi-major Axis (a): +/- %0.2f AU' %(fwhmSemMaj/2))
    print('Eccentricity (e): +/- %0.2f'%(fwhmEcc/2))
    print('Inclination (i): +/- %0.1f degrees' %(fwhmInc/2))
    print('Longitude of Ascending Node (Omega): +/- %0.1f degrees' %(fwhmLAN/2))
    print('Argument of Perihelion (omega): +/- %0.1f degrees' %(fwhmPer/2))
    print('Mean Anomaly (M): +/- %0.0f degrees' %(fwhmMA/2))

    #print percent uncertainty of orbital elements given through Monte Carlo
    #for data points (1, 2, 4)
    if center==2: #central observation at UT 06:20:04 on 6/23/2019
        print()
        print('PERCENT UNCERTAINTY OF ORBITAL ELEMENTS - WITH MONTE CARLO:')
        print('Percent uncertainty of semi-major axis (AU): %0.2f' %((abs(aJPL2-goodSemMaj[-1])/aJPL2)*100), '%')
        print('Percent uncertainty of eccentricity: %0.2f' %((abs(eJPL2-goodEcc[-1])/eJPL2)*100), '%')
        print('Percent uncertainty of inclination (degrees): %0.2f' %((abs(iJPL2-goodInc[-1])/iJPL2)*100), '%')
        print('Percent uncertainty of longitude of acsending node (degrees): %0.2f' %((abs(longJPL2-goodLAN[-1])/longJPL2)*100), '%')
        print('Percent uncertainty of mean anomaly (degrees): %0.2f' %((abs(maJPLpre2-goodMA[-1])/maJPLpre2)*100), '%') #precess mean anomaly
        print('Percent uncertainty of mean motion (radians/day): %0.2f' %((abs(mmJPL2-goodmm[-1])/mmJPL2)*100), '%')
        print('Percent uncertainty of time of last periapsis (JD): %0.6f' %((abs(TJPL2-goodT[-1])/TJPL2)*100), '%')
        print('Percent uncertainty of orbital period (years): %0.2f' %((abs(opJPL2-goodOP[-1])/opJPL2)*100), '%')
    #for data points (1, 3, 4)
    if center==3: #central observation at UT 07:40:27 on 7/05/2019
        print()
        print('PERCENT UNCERTAINTY OF ORBITAL ELEMENTS - WITH MONTE CARLO:')
        print('Percent uncertainty of semi-major axis (AU): %0.2f' %((abs(aJPL3-goodSemMaj[-1])/aJPL3)*100), '%')
        print('Percent uncertainty of eccentricity: %0.2f' %((abs(eJPL3-goodEcc[-1])/eJPL3)*100), '%')
        print('Percent uncertainty of inclination (degrees): %0.2f' %((abs(iJPL3-goodInc[-1])/iJPL3)*100), '%')
        print('Percent uncertainty of longitude of acsending node (degrees): %0.2f' %((abs(longJPL3-goodLAN[-1])/longJPL3)*100), '%')
        print('Percent uncertainty of mean anomaly (degrees): %0.2f' %((abs(maJPLpre3-goodMA[-1])/maJPLpre3)*100), '%') #precess mean anomaly
        print('Percent uncertainty of mean motion (radians/day): %0.2f' %((abs(mmJPL3-goodmm[-1])/mmJPL3)*100), '%')
        print('Percent uncertainty of time of last periapsis (JD): %0.6f' %((abs(TJPL3-goodT[-1])/TJPL3)*100), '%')
        print('Percent uncertainty of orbital period (years): %0.2f' %((abs(opJPL3-goodOP[-1])/opJPL3)*100), '%')

#SIMULATE ORBIT OF THE ASTEROID USING VPYTHON
if vPythonVis==True:
    print()
    print('The simulated orbits of asteroid 1998 OH (white), Earth (green), and Mars (red) will appear shortly!')
    sun = vpython.sphere(pos=vector(0,0,0),color=color.yellow,radius=(50))
    r1ecliptic = vector(0,0,0)
    Earthecliptic = vector(-5, 0, 0)
    Marsecliptic = vector(-2, 0, 0)
    earth = vpython.sphere(pos=Earthecliptic*50, color=color.green, radius=(15))
    earth.trail = vpython.curve(color=color.green)
    mars = vpython.sphere(pos=Marsecliptic*150, color=color.red, radius=(10))
    mars.trail = vpython.curve(color=color.red)
    asteroid = vpython.sphere(pos=r1ecliptic*150, radius=(5), color=color.white)
    asteroid.trail = vpython.curve(color=color.white)
    time = 0
    sqrtmu = 0.01720209895
    mu = sqrtmu**2
    period = sqrt(4*pi**2*goodSemMaj[-1]**3/mu)
    Eperiod = sqrt(4*pi**2*1**3/mu)
    Mperiod = sqrt(4*pi**2*1.52366231**3/mu)
    def solvekep(M0):
        Eguess = M0
        Mguess = Eguess - e*sin(Eguess)
        Mtrue = M0
        while abs(Mguess - Mtrue) > 1e-004:
            Mguess = Eguess - e*sin(Eguess)
            Eguess = Eguess - (Mtrue - (Eguess - e*sin(Eguess))) / (e*cos(Eguess)-1)
        return Eguess
    
    while (True):
        #for asteroid
        vpython.rate(200)
        Mtrue = 2*pi/period*(time) + radians(goodMA[-1])
        Etrue = solvekep(Mtrue)
        array1 = np.array([[cos(radians(goodLAN[-1])), -sin(radians(goodLAN[-1])), 0],
                          [sin(radians(goodLAN[-1])), cos(radians(goodLAN[-1])), 0],
                          [0, 0, 1]])
        array2 = np.array([[1, 0, 0],
                          [0, cos(radians(goodInc[-1])), -sin(radians(goodInc[-1]))],
                          [0, sin(radians(goodInc[-1])), cos(radians(goodInc[-1]))]])
        array3 = np.array([[cos(radians(goodPer[-1])), -sin(radians(goodPer[-1])), 0],
                          [sin(radians(goodPer[-1])), cos(radians(goodPer[-1])), 0],
                          [0, 0, 1]])
        array4 = np.array([[a*cos(Etrue)-a*e],
                          [a*math.sqrt(1-e**2)*sin(Etrue)],
                          [0]])
        multipliedArray = np.matmul(array3, array4)
        multArray2 = np.matmul(array2, multipliedArray)
        multArray3 = np.matmul(array1, multArray2)
        r1ecliptic.x = multArray3[0]
        r1ecliptic.y = multArray3[1]
        r1ecliptic.z = multArray3[2]
        asteroid.pos = r1ecliptic*150
        asteroid.trail.append(pos=asteroid.pos)
        
        ##for earth
        vpython.rate(200)
        EMtrue = 2*pi/Eperiod*(time) + radians(355.53)
        EEtrue = solvekep(EMtrue)
        Earray1 = np.array([[cos(radians(348.74)), -sin(radians(348.74)), 0],
                          [sin(radians(348.74)), cos(radians(348.74)), 0],
                          [0, 0, 1]])
        Earray2 = np.array([[1, 0, 0],
                          [0, cos(radians(23.44)), -sin(radians(23.44))],
                          [0, sin(radians(23.44)), cos(radians(23.44))]])
        Earray3 = np.array([[cos(radians(102.94719)), -sin(radians(102.94719)), 0],
                          [sin(radians(102.94719)), cos(radians(102.94719)), 0],
                          [0, 0, 1]])
        Earray4 = np.array([[1*cos(EEtrue)-1*0.01671022],
                          [1*math.sqrt(1-0.01671022**2)*sin(EEtrue)],
                          [0]])
        EmultipliedArray = np.matmul(Earray3, Earray4)
        EmultArray2 = np.matmul(Earray2, EmultipliedArray)
        EmultArray3 = np.matmul(Earray1, EmultArray2)
        Earthecliptic.x = EmultArray3[0]
        Earthecliptic.y = EmultArray3[1]
        Earthecliptic.z = EmultArray3[2]
        earth.pos = Earthecliptic*150
        earth.trail.append(pos=earth.pos)

        ##for mars
        vpython.rate(200)
        MMtrue = 2*pi/Mperiod*(time) + radians(355.45332)
        MEtrue = solvekep(MMtrue)
        Marray1 = np.array([[cos(radians(49.57854)), -sin(radians(49.57854)), 0],
                          [sin(radians(49.57854)), cos(radians(49.57854)), 0],
                          [0, 0, 1]])
        Marray2 = np.array([[1, 0, 0],
                          [0, cos(radians(1.85061)), -sin(radians(1.85061))],
                          [0, sin(radians(1.85061)), cos(radians(1.85061))]])
        Marray3 = np.array([[cos(radians(336.04084)), -sin(radians(336.04084)), 0],
                          [sin(radians(336.04084)), cos(radians(336.04084)), 0],
                          [0, 0, 1]])
        Marray4 = np.array([[1.52366231*cos(MEtrue)-1.52366231*0.09341233],
                          [1.52366231*math.sqrt(1-0.09341233**2)*sin(MEtrue)],
                          [0]])
        MmultipliedArray = np.matmul(Marray3, Marray4)
        MmultArray2 = np.matmul(Marray2, MmultipliedArray)
        MmultArray3 = np.matmul(Marray1, MmultArray2)
        Marsecliptic.x = MmultArray3[0]
        Marsecliptic.y = MmultArray3[1]
        Marsecliptic.z = MmultArray3[2]
        mars.pos = Marsecliptic*150
        mars.trail.append(pos=mars.pos)
        
        time = time + 1

        scene2 = canvas(title='for scifair poster',width=600, height=200,center=vector(5,0,0), background=vector(0, 1, 1))

#make 6 histograms for the 6 orbital elements
####for SEMI-MAJOR AXIS
#####find average
avgSumSemMaj = 0
for y in range(len(mcSemMaj)):
    avgSumSemMaj += float(mcSemMaj[y])
avgSemMaj = avgSumSemMaj / len(mcSemMaj)
#define gaussian distribution function
def gaussian(x, a, mean, std):
    return 1./(np.sqrt(np.pi * 2) * std)*np.exp(-(x-1.58)**2/(2*std**2))
omega1 = np.random.normal(avgSemMaj, stdSemMaj, 500)
omega2 = np.random.normal(avgSemMaj, stdSemMaj, 500)
fig, ax = plt.subplots(1,1)
#ax.set_facecolor("#ccc9deff")
###for the mean line
ax.axvline(1.58, color="black", linestyle="--",
            label="Mean = " + "%.2F" % 1.58)
###graph where JPL is at
ax.axvline(1.54, color="blue", linestyle=":",
            label="JPL : " + "%.2F" % 1.54)
###standard deviation
span = np.linspace(1.61, 1.55, 50)  ##### Just the range of mean +/- std
xrange = np.linspace(1.35,1.75,50) # Something like the full xspan of the plot
###plot the functional form of the gaussian
ax.plot(xrange, gaussian(xrange, None, avgSemMaj, stdSemMaj), color="blue")
###let's fill in the 1 sigma range with color
plt.fill_between(span, np.zeros(50), gaussian(span, None, avgSemMaj, stdSemMaj),
                 alpha=0.50, zorder=3, label = r"$\sigma$= %.2F" % stdSemMaj)
plt.title('Gaussian Distribution for Semi-major Axis (AU)')
plt.ylabel('Normalized Distribution')
plt.xlabel('Semi-major Axis (AU)')
plt.ylim(0, 14)
plt.xlim(1.35, 1.75)
plt.legend()
plt.show()

####for ECCENTRICITY
#####find average
##avgSumEcc = 0
##for y in range(len(mcEcc)):
##    avgSumEcc += float(mcEcc[y])
##avgEcc = avgSumEcc / len(mcEcc)
###define gaussian distribution function
##def gaussian(x, a, mean, std):
##    return 1./(np.sqrt(np.pi * 2) * std)*np.exp(-(x-mean)**2/(2*std**2))
##omega1 = np.random.normal(avgEcc, stdEcc, 500)
##omega2 = np.random.normal(avgEcc, stdEcc, 500)
##fig, ax = plt.subplots(1,1, facecolor="#ccc9deff")
##ax.set_facecolor("#ccc9deff")
#####for the mean line
##ax.axvline(np.mean(omega1), color="black", linestyle="--",
##            label="Mean = " + "%.2F" % np.mean(omega1))
#####graph where JPL is at
##ax.axvline(0.40594244, color="blue", linestyle=":",
##            label="JPL : " + "%.2F" % 0.40594)
#####standard deviation
##span = np.linspace(avgEcc - stdEcc, avgEcc + stdEcc, 50)  ##### Just the range of mean +/- std
##xrange = np.linspace(0.35,0.55,50) # Something like the full xspan of the plot
#####plot the functional form of the gaussian
##ax.plot(xrange, gaussian(xrange, None, avgEcc, stdEcc), color="blue")
#####let's fill in the 1 sigma range with color
##plt.fill_between(span, np.zeros(50), gaussian(span, None, avgEcc, stdEcc),
##                 alpha=0.50, color="#FFC0CB", zorder=3, label = r"$\sigma$= %.2F" % stdEcc)
##plt.title('Gaussian Distribution for Eccentricity')
##plt.ylabel('Normalized Distribution')
##plt.xlabel('Eccentricity')
##plt.ylim(0, 43)
##plt.xlim(0.375, 0.5)
##plt.legend()
##fig.savefig("output.pdf", facecolor=fig.get_facecolor(), transparent=True)
####plt.set_facecolor("#ccc9deff")
##plt.show()

####for INCLINATION
#####find average
##avgSumInc = 0
##for y in range(len(mcInc)):
##    avgSumInc += float(mcInc[y])
##avgInc = avgSumInc / len(mcInc)
###define gaussian distribution function
##def gaussian(x, a, mean, std):
##    return 1./(np.sqrt(np.pi * 2) * std)*np.exp(-(x-mean)**2/(2*std**2))
##omega1 = np.random.normal(avgInc, stdInc, 500)
##omega2 = np.random.normal(avgInc, stdInc, 500)
##fig, ax = plt.subplots(1,1)
#####for the mean line
##ax.axvline(np.mean(omega1), color="black", linestyle="--",
##            label="Mean = " + "%.1F" % np.mean(omega1))
#####graph where JPL is at
##ax.axvline(24.52816384749816, color="blue", linestyle=":",
##            label="JPL : " + "%.1F" % 24.52816384749816)
#####standard deviation
##span = np.linspace(avgInc - stdInc, avgInc + stdInc, 50)  ##### Just the range of mean +/- std
##xrange = np.linspace(24,26.5,50) # Something like the full xspan of the plot
#####plot the functional form of the gaussian
##ax.plot(xrange, gaussian(xrange, None, avgInc, stdInc), color="blue")
#####let's fill in the 1 sigma range with color
##plt.fill_between(span, np.zeros(50), gaussian(span, None, avgInc, stdInc),
##                 alpha=0.50, color="#FFC0CB", zorder=3, label = r"$\sigma$= %.1F" % stdInc)
##plt.title('Gaussian Distribution for Inclination (degrees)')
##plt.ylabel('Normalized Distribution')
##plt.xlabel('Inclination (degrees)')
##plt.ylim(0, 2)
##plt.xlim(24, 26.5)
##plt.legend()
##plt.show()

####for LONGITUDE OF ASCENDING NODE
#####find average
##avgSumLAN = 0
##for y in range(len(mcLAN)):
##    avgSumLAN += float(mcLAN[y])
##avgLAN = avgSumLAN / len(mcLAN)
###define gaussian distribution function
##def gaussian(x, a, mean, std):
##    return 1./(np.sqrt(np.pi * 2) * std)*np.exp(-(x-mean)**2/(2*std**2))
##omega1 = np.random.normal(avgLAN, stdLAN, 500)
##omega2 = np.random.normal(avgLAN, stdLAN, 500)
##fig, ax = plt.subplots(1,1)
#####for the mean line
##ax.axvline(np.mean(omega1), color="black", linestyle="--",
##            label="Mean = " + "%.1F" % np.mean(omega1))
#####graph where JPL is at
##ax.axvline(220.7465854289499, color="blue", linestyle=":",
##            label="JPL : " + "%.1F" % 220.7465854289499)
#####standard deviation
##span = np.linspace(avgLAN - stdLAN, avgLAN + stdLAN, 50)  ##### Just the range of mean +/- std
##xrange = np.linspace(219,221,50) # Something like the full xspan of the plot
#####plot the functional form of the gaussian
##ax.plot(xrange, gaussian(xrange, None, avgLAN, stdLAN), color="blue")
#####let's fill in the 1 sigma range with color
##plt.fill_between(span, np.zeros(50), gaussian(span, None, avgLAN, stdLAN),
##                 alpha=0.50, color="#FFC0CB", zorder=3, label = r"$\sigma$= %.1F" % stdLAN)
##plt.title('Gaussian Distribution for Longitude of Ascending Node (degrees)')
##plt.ylabel('Normalized Distribution')
##plt.xlabel('Longitude of Ascending Node (degrees)')
##plt.ylim(0, 2)
##plt.xlim(219.25, 221)
##plt.legend()
##plt.show()

####for ARGUMENT OF PERIHELION
#####find average
##avgSumPer = 0
##for y in range(len(mcPer)):
##    avgSumPer += float(mcPer[y])
##avgPer = avgSumPer / len(mcPer)
###define gaussian distribution function
##def gaussian(x, a, mean, std):
##    return 1./(np.sqrt(np.pi * 2) * std)*np.exp(-(x-mean)**2/(2*std**2))
##omega1 = np.random.normal(avgPer, stdPer, 500)
##omega2 = np.random.normal(avgPer, stdPer, 500)
##fig, ax = plt.subplots(1,1)
#####for the mean line
##ax.axvline(np.mean(omega1), color="black", linestyle="--",
##            label="Mean = " + "%.1F" % np.mean(omega1))
#####graph where JPL is at
##ax.axvline(321.7288591834879, color="blue", linestyle=":",
##            label="JPL : " + "%.1F" % 321.7288591834879)
#####standard deviation
##span = np.linspace(avgPer - stdPer, avgPer + stdPer, 50)  ##### Just the range of mean +/- std
##xrange = np.linspace(321,326,50) # Something like the full xspan of the plot
#####plot the functional form of the gaussian
##ax.plot(xrange, gaussian(xrange, None, avgPer, stdPer), color="blue")
#####let's fill in the 1 sigma range with color
##plt.fill_between(span, np.zeros(50), gaussian(span, None, avgPer, stdPer),
##                 alpha=0.50, color="#FFC0CB", zorder=3, label = r"$\sigma$= %.1F" % stdPer)
##plt.title('Gaussian Distribution for Argument of Perihelion (degrees)')
##plt.ylabel('Normalized Distribution')
##plt.xlabel('Argument of Perihelion (degrees)')
##plt.ylim(0, 0.75)
##plt.xlim(321, 326)
##plt.legend()
##plt.show()

####for MEAN ANOMALY
#####find average
##avgSumMA = 0
##for y in range(len(mcMA)):
##    avgSumMA += float(mcMA[y])
##avgMA = avgSumMA / len(mcMA)
###define gaussian distribution function
##def gaussian(x, a, mean, std):
##    return 1./(np.sqrt(np.pi * 2) * std)*np.exp(-(x-mean)**2/(2*std**2))
##omega1 = np.random.normal(avgMA, stdMA, 500)
##omega2 = np.random.normal(avgMA, stdMA, 500)
##fig, ax = plt.subplots(1,1)
#####for the mean line
##ax.axvline(np.mean(omega1), color="black", linestyle="--",
##            label="Mean = " + "%.0F" % np.mean(omega1))
#####graph where JPL is at
##ax.axvline(maJPLpre2, color="blue", linestyle=":",
##            label="JPL : " + "%.0F" % maJPLpre2)
#####standard deviation
##span = np.linspace(avgMA - stdMA, avgMA + stdMA, 50)  ##### Just the range of mean +/- std
##xrange = np.linspace(29,38,50) # Something like the full xspan of the plot
#####plot the functional form of the gaussian
##ax.plot(xrange, gaussian(xrange, None, avgMA, stdMA), color="blue")
#####let's fill in the 1 sigma range with color
##plt.fill_between(span, np.zeros(50), gaussian(span, None, avgMA, stdMA),
##                 alpha=0.50, color="#FFC0CB", zorder=3, label = r"$\sigma$= %.0F" % stdMA)
##plt.title('Gaussian Distribution for Mean Anomaly (degrees)')
##plt.ylabel('Normalized Distribution')
##plt.xlabel('Mean Anomaly (degrees)')
##plt.ylim(0, 0.375)
##plt.xlim(29, 38)
##plt.legend()
##plt.show()
###END
