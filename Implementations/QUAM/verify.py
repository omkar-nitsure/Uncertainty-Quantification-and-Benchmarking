import numpy as np
import matplotlib.pyplot as plt

file = open("/ASR_Kmeans/QUAM/entropies_ID_C.txt", "r")

lines = file.readlines()

indexes = np.arange(0, 1000, 1)

# thresh = 2
# c=1

for i in range(len(lines)):
    lines[i] = np.abs(float(lines[i][:-1]))
    # if(lines[i] > thresh):
    #     c +=1


id_o = [
    39,
    220,
    292,
    330,
    347,
    390,
    449,
    484,
    519,
    599,
    611,
    702,
    735,
    741,
    765,
    768,
    775,
    800,
    806,
    841,
    845,
    994,
    998,
]

id_d = np.zeros(1000)

for i in range(len(id_o)):
    id_d[id_o[i]] = 1


p = np.arange(0, 100, 1)
acc = []

for i in range(len(p)):
    id = np.argsort(lines)[1000 - 10 * p[i] :]

    nc = 0
    for j in range(len(id)):
        if id_d[id[j]] == 1:
            nc += 1

    left = 1000 - 10 * p[i]

    acc.append(100 * (left - (23 - nc)) / left)

plt.plot(p, acc)
plt.xlabel("% of highly uncertain samples removed")
plt.ylabel("Accuracy in %")
plt.title("accuracy plot")
plt.savefig("/ASR_Kmeans/QUAM/plot.png")
