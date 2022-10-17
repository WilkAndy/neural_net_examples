import sys
import numpy as np
import pandas as pd

ep = 0.01
noise = 0.2
xvals = np.arange(0, 1 + ep, ep)
smooth = 0.5 + 0.5 * np.sin(1.5 * np.pi * xvals)
noisy = smooth + noise * (np.random.rand(len(xvals)) - 0.5)

df = pd.DataFrame({'x': xvals, 'smooth': smooth, 'noisy': noisy})
df.to_csv('data.csv', index = False)

plotit = True
if plotit:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(xvals, smooth, 'k-', label = 'smooth')
    plt.scatter(xvals, noisy, c = 'b', label = 'noisy')
    plt.legend()
    plt.grid()
    plt.xlabel('input')
    plt.ylabel('output')
    plt.xlim([0, 1])
    plt.title('The two datasets')
    plt.show()
    plt.savefig('data.svg', bbox_inches = 'tight')

sys.exit(0)
