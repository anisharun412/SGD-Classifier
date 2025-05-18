# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data.
2. Split Dataset into Training and Testing Sets.
3. Train the Model Using Stochastic Gradient Descent (SGD).
4. Make Predictions and Evaluate Accuracy.
5. Generate Confusion Matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Arunsamy D
RegisterNumber: 212224240016
*/
```

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```
```python
data = pd.DataFrame(load_iris().data, columns = load_iris().feature_names)
data['Target']=load_iris().target
data.head()
```
```python
x = data.drop(['Target'], axis=1)
y = data[['Target']]

print(x)
print(y)
```
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
```python
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
model.fit(x_train, y_train)
```
```python
y_pred = model.predict(x_test)
y_pred
```
```python
accuracy = accuracy_score(y_test, y_pred)
accuracy
```
```python
cm = confusion_matrix(y_test, y_pred)
cm
```
## Output:

### Data Head
![image](https://github.com/user-attachments/assets/c488b800-5b30-4736-a6c9-3e82570642a8)

### Input & Output Features
![image](https://github.com/user-attachments/assets/1c28f187-06a7-4fc0-a10f-875c48a80bb4)

### Predicted Values
![image](https://github.com/user-attachments/assets/f3652b54-c2ad-444c-aad6-69cd143b90c5)

### Accuracy
![image](https://github.com/user-attachments/assets/8016ecbe-207e-4d9a-a47c-3b4b198728c2)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/f164bd53-15fe-464f-87c6-328d53755381)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
