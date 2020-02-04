import matplotlib.pyplot as plt

x = [3, 4, 5, 6, 7, 8, 9, 10]
EER = [0.333, 0.3, 0.29, 0.29, 0.29, 0.298, 0.287, 0.284]
plt.plot(x, EER, color="b")
plt.plot(x, EER, "bo")
plt.grid()
plt.xlabel("Maksymalna liczba i-vectorow na osobe")
plt.ylabel("EER")
plt.show()
print