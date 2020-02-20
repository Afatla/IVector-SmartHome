import matplotlib.pyplot as plt

x = [3, 4, 5, 6, 7, 8, 9, 10]
EER_d = [0.328, 0.316, 0.297, 0.286, 0.293, 0.278, 0.287, 0.293]
EER_m = [0.292, 0.266, 0.284, 0.292, 0.295, 0.278, 0.278, 0.281]
EER_s = [0.318, 0.293, 0.316, 0.307, 0.309, 0.283, 0.290, 0.301]
EER_t = [0.333, 0.3, 0.29, 0.29, 0.29, 0.298, 0.287, 0.284]

plt.plot(x, EER_d, color="b")
#plt.plot(x, EER_d, "bo")
plt.plot(x, EER_m, color="r")
#plt.plot(x, EER_m, "ro")
plt.plot(x, EER_s, color="g")
#plt.plot(x, EER_s, "go")
plt.plot(x, EER_t, color="k")
#plt.plot(x, EER_t, "ko")
plt.grid()
plt.xlabel("Maksymalna liczba i-vectorow na osobe")
plt.ylabel("EER")
plt.legend(["Otworz drzwi", "Wlacz muzyke", "Zapal swiatlo", "Zwieksz temperature"])
plt.show()
print


