"""ЛР-5, завдання 2.2. Обробка дисбалансу класів (data_imbalance.txt).

Запуск:
    python LR_5_task_2.py              # ОБИДВА прогони підряд (власне розширення)
    python LR_5_task_2.py balance      # лише з class_weight='balanced'
    python LR_5_task_2.py nobalance    # лише без балансування (власне розширення)

ВІДХИЛЕННЯ ВІД МЕТОДИЧКИ (описане у звіті, розділ «Зауваження до методички»):
методичка передбачає лише аргумент 'balance', а запуск без аргументів виконує
один прогін без балансування. Тут додано третє значення 'nobalance', а запуск
без аргументів виконує обидва прогони підряд — це дозволяє отримати всі чотири
рисунки завдання 2.2 та обидві пари звітів за один запуск. Будь-який інший
аргумент, як і вимагає методичка, викликає TypeError.
"""

import sys                                       # доступ до аргументів командного рядка

import matplotlib                                # налаштування бекенда ДО pyplot
matplotlib.use("Agg")                            # малюємо у файли

import numpy as np                               # масиви
import matplotlib.pyplot as plt                  # графіки
from sklearn.ensemble import ExtraTreesClassifier            # гранично випадковий ліс
from sklearn.model_selection import train_test_split         # поділ вибірки
from sklearn.metrics import classification_report            # звіт якості

from utilities import visualize_classifier       # власна функція візуалізації


# Завантаження вхідних даних
input_file = 'data_imbalance.txt'                # файл з двома НЕзбалансованими класами
data = np.loadtxt(input_file, delimiter=',')     # читаємо числа через кому
X, y = data[:, :-1], data[:, -1]                 # ознаки та мітки

# Поділ вхідних даних на два класи на підставі міток
class_0 = np.array(X[y == 0])                    # точки класу 0 (їх мало)
class_1 = np.array(X[y == 1])                    # точки класу 1 (їх багато)
print('Кількість точок класу 0:', len(class_0))  # демонструємо дисбаланс
print('Кількість точок класу 1:', len(class_1))
print('Співвідношення класів 1:0 =', round(len(class_1) / len(class_0), 2))

# Візуалізація вхідних даних
plt.figure()
# УВАГА: у методичці для маркера 'x' задано facecolors+edgecolors, що дає
# matplotlib UserWarning (незаповнений маркер не має заливки). Використовуємо color.
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, color='black',
            linewidth=1, marker='x')                         # клас 0 — хрестики
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='o')     # клас 1 — кола
plt.title('Input data')
plt.savefig('fig8_imbalance_input.png', dpi=150, bbox_inches='tight')
print('[saved] fig8_imbalance_input.png')

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

class_names = ['Class-0', 'Class-1']             # підписи класів у звіті


def run(balance: bool, tag: str, n_train: int, n_test: int):
    """Один прогін класифікатора: balance=True вмикає class_weight='balanced'."""
    # Класифікатор на основі гранично випадкових лісів
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if balance:
        # class_weight='balanced' робить ваги обернено пропорційними
        # до кількості точок у класі — рідкісний клас «важить» більше
        params['class_weight'] = 'balanced'

    classifier = ExtraTreesClassifier(**params)  # створюємо класифікатор
    classifier.fit(X_train, y_train)             # навчаємо на тренувальних даних

    # Візуалізація меж рішень на тренувальному наборі
    visualize_classifier(classifier, X_train, y_train, 'Training dataset',
                         save_path=f'fig{n_train}_{tag}_train.png')

    # Прогноз та візуалізація для тестового набору
    y_test_pred = classifier.predict(X_test)     # прогноз на тесті
    visualize_classifier(classifier, X_test, y_test, 'Test dataset',
                         save_path=f'fig{n_test}_{tag}_test.png')

    # Обчислення показників ефективності класифікатора.
    # Методичка (с.10) вимагає ДВА звіти: на навчальному і на тестовому наборах.
    print("\n" + "#" * 40)
    print(f"\nclass_weight = {'balanced' if balance else 'None'}\n")
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train),
                                target_names=class_names))
    print("#" * 40 + "\n")

    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")


# Розбір прапорця так, як це описано в методичці
if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        run(balance=True, tag='imbalance_balanced', n_train=11, n_test=12)   # лише збалансований
    elif sys.argv[1] == 'nobalance':
        run(balance=False, tag='imbalance_nobalance', n_train=9, n_test=10)  # лише незбалансований
    else:
        raise TypeError("Invalid input argument; should be 'balance'")
else:
    # За замовчуванням робимо ОБИДВА прогони, щоб порівняти їх у звіті
    run(balance=False, tag='imbalance_nobalance', n_train=9, n_test=10)
    run(balance=True, tag='imbalance_balanced', n_train=11, n_test=12)
