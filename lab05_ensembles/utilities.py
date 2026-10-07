"""Допоміжні функції для ЛР-5 (ансамблеві методи).

У методичці файл utilities.py лише імпортується (`from utilities import
visualize_classifier`), але його текст не наведений, тому функція написана
самостійно за описом з книги П. Джоши.
"""

import matplotlib                      # базовий пакет побудови графіків
matplotlib.use("Agg")                  # неінтерактивний бекенд: малюємо у файл, а не у вікно

import numpy as np                     # робота з масивами та сітками
import matplotlib.pyplot as plt        # інтерфейс побудови графіків


def visualize_classifier(classifier, X, y, title='', save_path=None):
    """Візуалізує межі прийняття рішень класифікатора на двовимірних даних.

    classifier -- навчений класифікатор sklearn з методом predict;
    X          -- матриця ознак розміру (n, 2);
    y          -- вектор міток класів;
    title      -- заголовок рисунка;
    save_path  -- шлях до PNG-файлу; якщо заданий, рисунок зберігається.

    Повертає об'єкт Figure, щоб його можна було додатково налаштувати.
    """
    X = np.asarray(X)                                   # гарантуємо тип ndarray
    y = np.asarray(y)                                   # мітки теж як ndarray

    # Межі сітки: беремо мінімум/максимум кожної ознаки з запасом в 1.0
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Крок сітки: чим менший, тим гладкіша межа, але довше обчислення
    mesh_step_size = 0.01

    # Будуємо прямокутну сітку точок, що покриває всю область даних
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size))

    # Класифікуємо КОЖНУ точку сітки -> отримуємо «карту» областей рішень
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)               # повертаємо форму сітки

    fig = plt.figure()                                  # новий рисунок
    plt.title(title)                                    # заголовок

    # Заливка областей рішень відтінками сірого (pcolormesh = «кольорова сітка»)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')

    # Поверх заливки наносимо самі точки даних, колір = справжній клас
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',
                linewidth=1, cmap=plt.cm.Paired)

    # Обмежуємо область відображення точно межами сітки
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Цілочисельні поділки на осях (дані в усіх завданнях лежать у межах 0..10)
    plt.xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    plt.yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    if save_path:                                       # якщо вказано шлях —
        plt.savefig(save_path, dpi=150, bbox_inches='tight')   # зберігаємо PNG
        print(f"[saved] {save_path}")                   # повідомляємо про запис

    return fig                                          # повертаємо рисунок
