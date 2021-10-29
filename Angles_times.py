# -*- coding: utf-8 -*-
"""
Created on the lords day of thursday on the lords month of spooktober of the lords year of 2021

@author: no
"""

import numpy as np
# Position of the center of the antenna array 2.83544303797 & 3.64556962025
ant = [-2.84, 3.65]
# Position of the mirror sources as given by the miniproject description
sources = [[-0.78,6.14],[0.78,6.14],[-11.82,6.14],[-2.39,-5.71],[-0.37,8.78]]
# Initilazied arrays/matricies
relative = np.zeros([5,2])
angles = np.zeros([5,1])
lengths = np.zeros([5,1])
time = np.zeros([5,1])

# Calculate the relative x & y distance analogous to the adjecent and opposite sides in a right triangle
for i in range(len(sources)):
    relative[i] = np.subtract(sources[i],ant)

# Find the angle with arctan, and if the point is to the right of the antenna mirror the angle
# If the angle is negative, express it as a positive one instead
for k in range(len(sources)):
    angles[k] = np.arctan((relative[k,1])/(relative[k,0]))
    if relative[k,0] <= 0:
        angles[k] = np.pi+angles[k]
    if angles[k] < 0:
        angles[k] = np.pi*2 + angles[k]

# Calculates the straight path propgation time from the mirror sources to the antenna
for h in range(len(sources)):
    lengths[h] = np.sqrt((relative[h,0])**2+(relative[h,1])**2)
    time[h] = lengths[h]/299792458

# Output the angles in degrees
print(angles*(180/np.pi))
print(time)