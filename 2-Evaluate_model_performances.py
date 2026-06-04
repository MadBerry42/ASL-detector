from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np

X_train = np.load('Data/X_test.npy')
X_test = np.load('Data/X_test.npy')
y_train = np.load('Data/y_test.npy')
y_test = np.load('Data/y_test.npy')
y_predicted = np.load('Data/y_predicted.npy')

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(y_predicted, axis= 1).tolist()

cm = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)

print(f"Confusion Matrix: {cm[0]}  {cm[1]} \n {cm[2]}  {cm[3]}")
print(f"Accuracy: {accuracy}")

