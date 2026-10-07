"""ЛР-5, завдання 2.1. Класифікатори на основі випадкових (RandomForest)
та гранично випадкових (ExtraTrees) лісів.

Базується на заготовці викладача random_forests.py.
Запуск:
    python LR_5_task_1.py --classifier-type rf
    python LR_5_task_1.py --classifier-type erf
"""

import argparse                                  # розбір аргументів командного рядка

import matplotlib                                # налаштування бекенда ДО pyplot
matplotlib.use("Agg")                            # малюємо у файли, вікна не відкриваємо

import numpy as np                               # масиви та числові операції
import matplotlib.pyplot as plt                  # побудова графіків
from sklearn.metrics import classification_report                 # звіт якості
from sklearn.model_selection import train_test_split              # поділ вибірки
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # ансамблі

from utilities import visualize_classifier       # власна функція візуалізації


# Парсер аргументів
def build_arg_parser():
    """Створює парсер, який приймає тип класифікатора: 'rf' або 'erf'."""
    parser = argparse.ArgumentParser(
        description='Classify data using Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest='classifier_type',
                        required=True, choices=['rf', 'erf'],
                        help="Type of classifier to use; 'rf' or 'erf'")
    return parser


if __name__ == '__main__':
    # Вилучення вхідних аргументів
    args = build_arg_parser().parse_args()       # читаємо аргументи
    classifier_type = args.classifier_type       # 'rf' або 'erf'

    # Завантаження вхідних даних
    input_file = 'data_random_forests.txt'       # файл з даними (3 класи)
    data = np.loadtxt(input_file, delimiter=',') # читаємо числа через кому
    X, y = data[:, :-1], data[:, -1]             # перші 2 стовпці — ознаки, останній — мітка

    # Розбиття вхідних даних на три класи
    class_0 = np.array(X[y == 0])                # точки класу 0
    class_1 = np.array(X[y == 1])                # точки класу 1
    class_2 = np.array(X[y == 2])                # точки класу 2

    # Візуалізація вхідних даних (квадрати / кола / трикутники)
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='s')   # клас 0 — квадрат
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')   # клас 1 — коло
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='^')   # клас 2 — трикутник
    plt.title('Input data')
    plt.savefig('fig1_input_data.png', dpi=150, bbox_inches='tight')
    print('[saved] fig1_input_data.png')

    # Розбивка даних на навчальний та тестовий набори
    # УВАГА: у заготовці викладача було train_test_split.train_test_split(...)
    # (у методичці — cross_validation.train_test_split). Це помилка:
    # модуль sklearn.cross_validation вилучено, функція імпортується напряму.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

    # Класифікатор на основі ансамблевого навчання
    # n_estimators — кількість дерев, max_depth — максимальна глибина дерева,
    # random_state — зерно генератора випадкових чисел (відтворюваність).
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)   # випадковий ліс
    else:
        classifier = ExtraTreesClassifier(**params)     # гранично випадковий ліс

    classifier.fit(X_train, y_train)                    # навчання на тренувальних даних

    # Номери рисунків для звіту: свої для rf і свої для erf
    n_train, n_test, n_pts = {'rf': (2, 3, 4), 'erf': (5, 6, 7)}[classifier_type]

    # Візуалізація меж рішень на тренувальному наборі
    visualize_classifier(classifier, X_train, y_train, 'Training dataset',
                         save_path=f'fig{n_train}_{classifier_type}_train.png')

    # Прогноз і візуалізація на тестовому наборі
    y_test_pred = classifier.predict(X_test)            # прогноз для тесту
    visualize_classifier(classifier, X_test, y_test, 'Test dataset',
                         save_path=f'fig{n_test}_{classifier_type}_test.png')

    # Перевірка роботи класифікатора
    class_names = ['Class-0', 'Class-1', 'Class-2']     # назви класів для звіту
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train),
                                target_names=class_names))
    print("#" * 40 + "\n")

    print("#" * 40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#" * 40 + "\n")

    # Обчислення параметрів довірливості (мір достовірності прогнозів)
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

    print("\nConfidence measure:")
    for datapoint in test_datapoints:                   # перебираємо тестові точки
        # predict_proba повертає ймовірності належності кожному з трьох класів
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Class-' + str(np.argmax(probabilities))  # клас з макс. ймовірністю
        print('\nDatapoint:', datapoint)
        print('Probabilities:', np.round(probabilities, 4))
        print('Predicted class:', predicted_class)

    # Візуалізація тестових точок на тлі меж класифікатора
    visualize_classifier(classifier, test_datapoints, [0] * len(test_datapoints),
                         'Test datapoints',
                         save_path=f'fig{n_pts}_{classifier_type}_datapoints.png')
