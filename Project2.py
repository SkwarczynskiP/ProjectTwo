from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Section 1: Load the Breast Cancer Wisconsin Dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

# Section 2: Convert the Data to a DataFrame
cancer_df = pd.DataFrame(data=x, columns=cancer.feature_names)
cancer_df['target'] = y

# Section 3: Do Some Exploratory Data Analysis
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)
print(cancer_df.head())  # Display the first few rows of the DataFrame
print(cancer_df.describe())  # Display the summary statistics of the DataFrame
print(cancer_df.isnull().sum())  # Display the number of missing values in the DataFrame
print()

# Section 4: Visualize the Distribution of the Target Variable
plt.title("Distribution of Malignant and Benign Tumors")
sns.countplot(x=cancer_df['target'], palette='coolwarm')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(cancer_df.corr(), annot=False, cmap='coolwarm')  # Create a correlation heatmap
plt.title("Correlation Heatmap of Breast Cancer Wisconsin Dataset")
plt.show()

# Section 5: Create a Standard Scaler to Standardize the Features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Section 6: Split the Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)
print("Train Features Shape: ", x_train.shape)
print("Test Features Shape: ", x_test.shape)
print()

# Section 7: Build and Evaluate the SVM Model
# Part 1: SVC with Linear Kernel
svc_linear = SVC(kernel='linear')
svc_linear.fit(x_train, y_train)
svc_linear_predict = svc_linear.predict(x_test)

accuracy_linear = accuracy_score(y_test, svc_linear_predict)
print("Linear Kernel Accuracy: ", accuracy_linear)

f1_linear = f1_score(y_test, svc_linear_predict)
print("Linear Kernel F1 Score: ", f1_linear)

print("Classification Report for Linear Kernel: ")
print(metrics.classification_report(y_test, svc_linear_predict))

print("Confusion Matrix for Linear Kernel: ")
print(confusion_matrix(y_test, svc_linear_predict))
print()

# Part 2: SVC with RBG Kernel
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(x_train, y_train)
svc_rbf_predict = svc_rbf.predict(x_test)

accuracy_rbf = accuracy_score(y_test, svc_rbf_predict)
print("RBF Kernel Accuracy: ", accuracy_rbf)

f1_rbf = f1_score(y_test, svc_rbf_predict)
print("RBF Kernel F1 Score: ", f1_rbf)

print("Classification Report for RBF Kernel: ")
print(metrics.classification_report(y_test, svc_rbf_predict))

print("Confusion Matrix for RBF Kernel: ")
print(confusion_matrix(y_test, svc_rbf_predict))
print()

# Part 3: SVC with Polynomial Kernel
svc_poly = SVC(kernel='poly')
svc_poly.fit(x_train, y_train)
svc_poly_predict = svc_poly.predict(x_test)

accuracy_poly = accuracy_score(y_test, svc_poly_predict)
print("Polynomial Kernel Accuracy: ", accuracy_poly)

f1_poly = f1_score(y_test, svc_poly_predict)
print("Polynomial Kernel F1 Score: ", f1_poly)

print("Classification Report for Polynomial Kernel: ")
print(metrics.classification_report(y_test, svc_poly_predict))

print("Confusion Matrix for Polynomial Kernel: ")
print(confusion_matrix(y_test, svc_poly_predict))

# Section 8: Create a Confusion Matrix Heatmap
# Part 1: Confusion Matrix Heatmap for Linear Kernel
confusion_matrix_linear = confusion_matrix(y_test, svc_linear_predict)
sns.heatmap(confusion_matrix_linear, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Linear Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Part 2: Confusion Matrix Heatmap for RBF Kernel
confusion_matrix_rbf = confusion_matrix(y_test, svc_rbf_predict)
sns.heatmap(confusion_matrix_rbf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for RBF Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Part 3: Confusion Matrix Heatmap for Polynomial Kernel
confusion_matrix_poly = confusion_matrix(y_test, svc_poly_predict)
sns.heatmap(confusion_matrix_poly, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Polynomial Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
