import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from data_utils import load_moons_dataset
from nearest_neighbor import KNearestNeighbor
# Generate data
X_train, X_test, y_train, y_test = load_moons_dataset()

# initialize model
k = 4
model = KNearestNeighbor(X_train, y_train, k)


# predict
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# additional metric
print(f"Accuracy score is: {accuracy:.3f}")
print("\nClassification Report:\n")
print(classification_rep)

# evaluate with confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title(f"KNN Confusion Matrix (k={k})")
plt.show()