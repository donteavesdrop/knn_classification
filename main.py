import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Чтение данных из файла с заменой запятых на точки
data = pd.read_csv('iris.txt', delimiter='\t', header=0, decimal=',')

# Разделение на признаки и метки классов
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Кодирование меток классов
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание KNN-классификатора
knn_classifier = KNeighborsClassifier()

# Обучение классификатора
knn_classifier.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = knn_classifier.predict(X_test)

# Оценка производительности
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Точность:", accuracy)
print("Точность (Precision):", precision)
print("Полнота (Recall):", recall)
print("F1-мера (F1 Score):", f1)

# Оценка числа опорных векторов
# В KNN-классификаторе нет опорных векторов, так как это метод "ленивого" обучения
print("Количество опорных векторов: Не применимо для KNN-классификатора")
