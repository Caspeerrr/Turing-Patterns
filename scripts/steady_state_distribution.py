"""
Illustrates the distribution of the staedy state parameters.
Only considers the images in DIR. Specify a threshold to only
consider a subset of the the steady state parameters.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


DIR = r'C:\\Users\\caspe\Documents\\UvA Fluidlab\\Code\\images'
EXT = '.png.ppm'

sns.set_theme()

steady_state = []
for filename in os.listdir(DIR):
        if filename.endswith('.png'):

            s = float(filename.split('-mu')[0])
            steady_state.append(s)

steady_state = np.array(steady_state)

plt.hist(steady_state, density=False)
plt.text(0.015, 0.65, 'mean = ' + str(round(np.mean(steady_state), 8)), fontsize=11, transform=plt.gcf().transFigure)
plt.text(0.015, 0.6, 'min = ' + str(round(min(steady_state), 8)), fontsize=11, transform=plt.gcf().transFigure)
plt.text(0.015, 0.55, 'max = ' + str(round(max(steady_state), 8)), fontsize=11, transform=plt.gcf().transFigure)
plt.grid(True)
plt.yscale('log')
plt.subplots_adjust(left=0.3)
plt.title('Final concentration difference values')
plt.show()

while True:
    threshold = float(input('What is your threshold? '))
    subset = steady_state[steady_state < threshold]
    plt.hist(subset, density=False)

    plt.text(0.015, 0.7, str(len(steady_state) - len(subset)) + ' removed of ' + str(len(steady_state)), fontsize=11, transform=plt.gcf().transFigure)
    plt.text(0.015, 0.65, 'mean = ' + str(round(np.mean(subset), 8)), fontsize=11, transform=plt.gcf().transFigure)
    plt.text(0.015, 0.6, 'min = ' + str(round(min(subset), 8)), fontsize=11, transform=plt.gcf().transFigure)
    plt.text(0.015, 0.55, 'max = ' + str(round(max(subset), 8)), fontsize=11, transform=plt.gcf().transFigure)
    plt.yscale('log')
    plt.title('Final concentration difference values')
    plt.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.show()


