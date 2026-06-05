from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os

X_train = np.load('Data/X_test.npy')
X_test = np.load('Data/X_test.npy')
y_train = np.load('Data/y_test.npy')
y_test = np.load('Data/y_test.npy')
y_predicted = np.load('Data/y_predicted.npy')

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(y_predicted, axis= 1).tolist()

# Confusion matrix
cm = confusion_matrix(ytrue, yhat)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = os.listdir(os.path.join('MP_Data'))) 
cm_display.plot()
accuracy = accuracy_score(ytrue, yhat)
plt.title(f"Accuracy: {accuracy}")
plt.show()


print(f"Accuracy: {accuracy}")





