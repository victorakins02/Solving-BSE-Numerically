import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('ex-psi-initial-norm2.csv')
position1 = df1[df1.columns[0]]
probability_density1 = df1[df1.columns[1]]

df2 = pd.read_csv('cn.csv')
position2 = df2[df2.columns[0]]
probability_density2 = df2[df2.columns[1]]

relative_error = abs(probability_density2 - probability_density1) / abs(probability_density1)

plt.figure(figsize=(10, 6))
plt.plot(position1, probability_density1, label='Probability Density 1 (Analytical)', color='b')
plt.plot(position2, probability_density2, label='Probability Density 2 (Numerical)', linestyle='--', color='r')
plt.xlabel('Position', fontsize=16)
plt.ylabel('Probability Density', fontsize=16)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=14)
plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.show()

plt.figure(figsize=(10, 6))
plt.xlabel('Position', fontsize=16)
plt.ylabel('Relative Error', fontsize=16)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=14)
plt.plot(position1, relative_error, label='Relative Error', color='g')
plt.show()
