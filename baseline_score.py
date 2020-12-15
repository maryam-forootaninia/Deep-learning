import numpy as np
import matplotlib.pyplot as plt

# pull output data from .csv files
data = np.genfromtxt('Output Data/trainOutput.csv', delimiter=',')
eps2 = np.genfromtxt('Output Data/testOutput2.csv', delimiter=',')

# pick out training data to plot
eps = np.empty((15, len(data.T)))
for i in reversed(range(len(data))):
    if (i+1) % 11 == 0:
        eps[int((i+1)/11-1), :] = data[i, :]
x = range(1, len(eps)+1)

# plot model loss over time (epochs)
plt.figure()
plt.plot(x, eps[:, 3])
plt.plot(x, eps2[:, 0])
plt.xticks(range(1, len(eps)+1), range(1, len(eps)+1))
plt.xlabel('Epoch')
plt.ylabel('Loss [%]')
plt.title('Baseline Model Loss')
plt.legend(['Training Set Loss', 'Test Set Loss'])
plt.show()

# plot model accuracy over time (epochs)
plt.figure()
plt.plot(x, eps[:, 5])
plt.plot(x, eps2[:, 1])
plt.xticks(range(1, len(eps)+1), range(1, len(eps)+1))
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.title('Baseline Model Accuracy')
plt.legend(['Training Set Accuracy', 'Test Set Accuracy'])
plt.show()
