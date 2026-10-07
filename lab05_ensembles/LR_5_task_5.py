"""ЛР-5, завдання 2.5. Прогнозування інтенсивності дорожнього руху
регресором на основі гранично випадкових лісів (ExtraTreesRegressor).

Набір даних: traffic_data.txt (попередньо оброблений Dodgers Loop Sensor).
Формат рядка: день_тижня, час_доби, команда-суперник, матч(yes/no), к-ть авто.

Запуск:
    python LR_5_task_5.py
"""

import matplotlib                                # налаштування бекенда ДО pyplot
matplotlib.use("Agg")                            # малюємо у файли

import numpy as np                               # масиви
import matplotlib.pyplot as plt                  # графіки
from sklearn.metrics import mean_absolute_error  # середня абсолютна помилка
from sklearn import preprocessing                # LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor # гранично випадковий ліс (регресія)
from sklearn.model_selection import train_test_split   # поділ вибірки

# Завантаження даних із файлу traffic_data.txt
input_file = 'traffic_data.txt'                  # вхідний файл
data = []                                        # сюди складаємо рядки
with open(input_file, 'r') as f:                 # відкриваємо файл на читання
    for line in f.readlines():                   # перебираємо рядки
        items = line[:-1].split(',')             # відкидаємо '\n' і ділимо по комі
        data.append(items)                       # додаємо список полів

data = np.array(data)                            # перетворюємо на масив рядків
print('Розмір набору даних:', data.shape)        # контроль: (17568, 5)

# Перетворення рядкових даних на числові.
# ОКРЕМИЙ LabelEncoder для КОЖНОЇ рядкової ознаки; числові ознаки НЕ кодуємо.
# Кодувальники зберігаємо, бо вони знадобляться для невідомої точки даних.
label_encoder = []                               # список кодувальників
X_encoded = np.empty(data.shape)                 # масив під закодовані дані
for i, item in enumerate(data[0]):               # дивимось на перший рядок
    if item.isdigit():                           # якщо поле суто числове —
        X_encoded[:, i] = data[:, i]             # переносимо стовпець як є
    else:                                        # інакше — це категорія:
        label_encoder.append(preprocessing.LabelEncoder())        # новий кодувальник
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])  # кодуємо стовпець

X = X_encoded[:, :-1].astype(int)                # ознаки (4 стовпці)
y = X_encoded[:, -1].astype(int)                 # цільова змінна — кількість авто
print('Створено кодувальників:', len(label_encoder))   # має бути 4

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# Регресор на основі гранично випадкових лісів
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)        # створюємо регресор
regressor.fit(X_train, y_train)                  # навчаємо на тренувальних даних

# Обчислення характеристик ефективності регресора на тестових даних
y_pred = regressor.predict(X_test)               # прогноз для тестової вибірки
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# Тестування кодування на одиночному прикладі (невідома точка даних)
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']       # вхідні дані
test_datapoint_encoded = [-1] * len(test_datapoint)           # заготовка під коди
count = 0                                                      # лічильник кодувальників
for i, item in enumerate(test_datapoint):        # перебираємо поля точки
    if item.isdigit():                           # числове поле —
        test_datapoint_encoded[i] = int(test_datapoint[i])     # просто число
    else:                                        # категорійне поле —
        # УВАГА: у методичці викликається transform(скаляр); сучасний sklearn
        # вимагає 1-вимірний масив, тому передаємо список з одного елемента.
        test_datapoint_encoded[i] = int(
            label_encoder[count].transform([test_datapoint[i]])[0])
        count = count + 1                        # переходимо до наступного кодувальника

test_datapoint_encoded = np.array(test_datapoint_encoded)      # у вигляді масиву
print("Закодована точка:", test_datapoint_encoded)

# Прогнозування результату для тестової точки даних
# predict очікує двовимірний масив, тому загортаємо точку у [ ]
prediction = regressor.predict([test_datapoint_encoded])[0]
print("Predicted traffic:", int(prediction))

# Додатковий рисунок: прогноз проти фактичних значень на перших 200 точках тесту
plt.figure(figsize=(11, 5))
n_show = 200                                     # скільки точок показати
plt.plot(y_test[:n_show], label='Фактична к-ть авто', linewidth=1.2)
plt.plot(y_pred[:n_show], label='Прогноз ExtraTrees', linewidth=1.2)
plt.xlabel('Номер тестової точки')
plt.ylabel('Кількість транспортних засобів')
plt.title('Прогноз інтенсивності руху (перші 200 точок тестової вибірки)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig15_traffic_prediction.png', dpi=150, bbox_inches='tight')
print('[saved] fig15_traffic_prediction.png')
