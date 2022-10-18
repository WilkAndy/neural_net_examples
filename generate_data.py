import sys
import numpy as np
import pandas as pd

ep = 0.01
noise = 0.2
x = np.arange(0, 1 + ep, ep)
smooth = 0.5 + 0.5 * np.sin(1.5 * np.pi * x)
oscillating = smooth + noise * np.sin(7.5 * np.pi * x)
noisy = smooth + noise * (np.random.rand(len(x)) - 0.5)

df = pd.DataFrame({'x': x, 'smooth': smooth, 'oscillating': oscillating, 'noisy': noisy})
df.to_csv('data.csv', index = False)

plotit = True
if plotit:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, smooth, 'k-', label = 'smooth')
    plt.plot(x, oscillating, c = 'lime', label = 'oscillating')
    plt.scatter(x, noisy, c = 'b', label = 'noisy')
    plt.legend()
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('output')
    plt.xlim([0, 1])
    plt.title('The datasets')
    plt.savefig('data.svg', bbox_inches = 'tight')
    plt.show()

sys.exit(0)
