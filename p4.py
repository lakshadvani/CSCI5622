from __future__ import division
import math
import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt



priors=[1/8,1/8,1/8,5/8]

g=[0.75,-0.6,1.4,-0.2]

priors2=[1/4,1/4,1/4,1/4]
sigma = 5
post=[0.0,0.0,0.0,0.0]
d=[[-1,0],[0,1],[1,0],[0,-1]]
for i in d: # left[dx,dy], up, right, down
    b = 1/(math.sqrt(2*math.pi*sigma**2))
    den = (2*sigma**2)
    R1=b*np.exp(-((g[0]-i[0])**2)/den)
    R2=b*np.exp(-((g[1]-i[1])**2)/den)
    B1=b*np.exp(-((g[2]-i[0])**2)/den)
    B2=b*np.exp(-((g[3]-i[1])**2)/den)
    post[d.index(i)]=priors[d.index(i)]*R1*R2*B1*B2
plt.clf()
v = plt.bar(['left','up','right','down'],post)
plt.xlabel('Direcition of Motion')
v[0].set_color('black')
v[1].set_color('black')
v[2].set_color('black')
v[3].set_color('black')


plt.ylabel('Normalized Probability')
plt.title('Scenario 1')
plt.show()
p = 0


post=post/sum(post)
print(post)
