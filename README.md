# asteroid-orbits
Determines the orbital characteristics of a given near-Earth asteroid. 

Project completed as part of the 2019 Summer Science Program in Astrophysics.

OrbitDetermination.py takes in the RA/DEC of an asteroid as well as the corresponding sun vectors from three distinct observations of the asteroid. Utilizing the Method of Gauss, the six classical orbital elements characterizing the asteroid's orbit are calculated. Monte Carlo simulations (1000 iterations) may be run on the data to generate the full width half maximum uncertainty of the six classical orbital elements.

Additionally, the asteroid's orbit in relation to the orbits of Earth and Mars is simulated using VPython, and Gaussian Distribution histograms for each of the six calculated classical orbital elements are included.

Tang.txt is a sample input file for OrbitDetermination.py. It includes the RA/DEC of asteroid 1998 OH as well as the corresponding sun vectors from three distinct observations of the asteroid in June and July of 2019. Measurements of 1998 OH's location on these nights were collected using the Etscorn Observatory in Socorro, New Mexico as part of the Summer Science Program.

1998 OH is a potentially hazardous asteroid, as its minimum orbit intersection distance with Earth is 0.03 AU and its absolute magnitude is 15.8. Since 1998 OH is a potential danger to Earth, its orbit must be closely monitored and updated.

Feel free to send any questions to me at sarahtang07@gmail.com!
