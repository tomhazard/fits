# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

"plt.plot([1,2,3,4], [1,4,9,16], 'ro')"
"plt.axis([0, 6, 0, 20])"
"plt.show()"

startf=4.0e9
endf=4.1e9
f0=4.05e9
numpnts=1000
Qe=1.0e3
Qi=100.0e3
Qt=1/(1/Qe+1/Qi)
print(Qt)
freq=[startf+abs(endf-startf)*(i+1)/numpnts for i in range(1000)]
phase=[np.arctan(2*Qt*(i/f0-1)) for i in freq]
S21=[np.absolute(1-Qt/(2*Qe)*(1+np.e**(2*i*complex(0,1)))) for i in phase]
'when Qi = Qe, S21 is 0.5? And as the system is more overcoupled (Qe > Qi) then the dip gets deeper?'
'arctan goes from pi/2 to -pi/2'
plt.figure(1)
plt.subplot(121)
plt.plot(freq,phase)
plt.subplot(122)
plt.plot(freq,S21)
plt.show() 