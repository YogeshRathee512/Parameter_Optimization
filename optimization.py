import pandas as pd
from ucimlrepo import fetch_ucirepo
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVC
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

# Fetch the Iris dataset from UCI repository
iris_data = fetch_ucirepo(id=53)

# Split the features and targets
X = iris_data.data.features
y = iris_data.data.targets

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the number of splits and initialize StratifiedShuffleSplit
num_splits = 10
sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.3, random_state=42)

# Lists to store the sampled data and results
X_samples = []
y_samples = []
Accuracies = []
Nus = []
Kernels = []

# Split the data into train and test samples
for train_index, test_index in sss.split(X, y_encoded):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    X_samples.append((X_train, X_test))
    y_samples.append((y_train, y_test))

# Perform SVM classification on each sample
for i, ((X_train, X_test), (y_train, y_test)) in enumerate(zip(X_samples, y_samples)):
    best_accuracy = 0
    best_kernel = ""
    best_nu = 0

    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']

    # Randomly search for the best parameters
    for _ in range(100):
        nu = random.random()
        kernel = random.choice(kernel_list)
        
        model = NuSVC(kernel=kernel, nu=nu)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel
            best_nu = nu

    Accuracies.append(best_accuracy)
    Kernels.append(best_kernel)
    Nus.append(best_nu)

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    "Sample": [f"S{i+1}" for i in range(num_splits)],
    "Best Accuracy": Accuracies,
    "Best Nu": Nus,
    "Best Kernel": Kernels
})

print(results_df)

# Plotting accuracy vs. iterations
iterations = list(range(100, 1100, 100))
plt.plot(iterations, Accuracies, marker='o', linestyle='-')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iterations')
plt.grid(True)
plt.show()
