import matplotlib.pyplot as plt
import numpy as np

DIR = '\\\wsl$\\Ubuntu-20.04\\home\\caspeerrr\\basilisk\\src\\examples\\brusselator\\concentrations.txt'

with open(DIR) as f:
    lines = f.readlines()

    concentration = []
    for line in lines:
        concentration.append(float(line))

n = len(concentration)
avg = 10
avg_concentration = np.zeros(n-avg)

for i in range(avg, n):
    avg_concentration[i-avg] = sum(concentration[i-avg+1:i+1])/10

x = np.arange(10, n)

f = plt.figure()
f.suptitle("Change in concentration C1", fontsize=14)
f.add_subplot(1,2, 1)
plt.scatter(x, avg_concentration, s=5)
plt.ylabel('concentration')
plt.xlabel('time')
plt.text(2000, 14, concentration)
f.add_subplot(1,2, 2)
plt.scatter(x, avg_concentration, s=5)
plt.title('Log scale')
plt.yscale('log')
plt.xlabel('time')
plt.show(block=True)
