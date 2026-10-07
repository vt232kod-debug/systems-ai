# -*- coding: utf-8 -*-
"""
Лабораторна робота №8 (№10 за методичкою)
Тема: «Ресурси Keras. TensorFlow. Навчання лінійної регресії»

Файл 2 з 2: СУЧАСНА реалізація того самого алгоритму на TensorFlow 2.x.
Студент: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.

Той самий генератор даних і ті самі гіперпараметри, що й у файлі 1, але:
  * немає tf.placeholder — дані передаються як звичайні аргументи функції;
  * немає tf.Session і feed_dict — обчислення виконуються одразу (eager);
  * градієнти беремо через tf.GradientTape замість автоматичної побудови
    статичного графа;
  * оптимізатор — keras.optimizers.SGD замість tf.train.GradientDescentOptimizer.

Додатково наведено ще коротшу реалізацію тієї самої моделі через
keras.Sequential з одним шаром Dense(1) — щоб показати на практиці ресурси
Keras, описані в теоретичній частині методички.
"""

import os                                   # доступ до змінних оточення
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # прибираємо службові C++ повідомлення TF

import time                                 # вимірювання часу навчання
import matplotlib                           # бібліотека побудови графіків
matplotlib.use("Agg")                       # неінтерактивний бекенд: малюємо у файл
import matplotlib.pyplot as plt             # основний інтерфейс побудови рисунків

import numpy as np                          # масиви та генерація даних
import tensorflow as tf                     # TensorFlow 2.x, «рідний» API
from tensorflow import keras                # високорівневий ресурс Keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # тека, де лежить цей скрипт
print('TensorFlow:', tf.__version__, '| Keras:', keras.__version__,
      '| бекенд Keras:', keras.backend.backend())      # підтверджуємо зв'язку Keras<->TF

# --------------------------------------------------------------------------
# Крок 1. Генерація вхідних даних — ІДЕНТИЧНО до файлу 1.
# --------------------------------------------------------------------------
RANDOM_SEED = 3                             # номер варіанта (3) як зерно генератора
np.random.seed(RANDOM_SEED)                 # фіксуємо генератор numpy
tf.random.set_seed(RANDOM_SEED)             # фіксуємо генератор TensorFlow

n_samples, batch_size, num_steps = 1000, 100, 20000   # ті самі гіперпараметри
X_data = np.random.uniform(1, 10, (n_samples, 1))     # 1000 точок x на [1; 10]
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))  # y = 2x + 1 + шум N(0; 2)

X_tf = tf.constant(X_data, dtype=tf.float32)          # дані одразу як тензори TF
y_tf = tf.constant(y_data, dtype=tf.float32)          # (у TF1 вони підставлялись через feed_dict)

A_ols = np.hstack([X_data, np.ones_like(X_data)])     # матриця плану [x, 1]
ols_k, ols_b = np.linalg.lstsq(A_ols, y_data, rcond=None)[0].ravel()  # аналітичний МНК-еталон

LEARNING_RATE = 1e-5                        # той самий темп навчання, що й у файлі 1
display_step = 100                          # крок друку діагностики

# ==========================================================================
# ВАРІАНТ А. Низькорівневий TF2: tf.Variable + tf.GradientTape + SGD.
# ==========================================================================
# Крок 3 методички: змінні k (нормальний розподіл) і b (нулі).
k = tf.Variable(tf.random.normal((1, 1)), name='slope')   # нахил
b = tf.Variable(tf.zeros((1,)), name='bias')              # зсув
k_init_tf2 = float(k.numpy().ravel()[0])                  # стартове k: TF2 дає ІНШЕ число, ніж TF1
print('Стартове значення k у TF2 (tf.random.set_seed): %.4f' % k_init_tf2)
opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE)   # стохастичний градієнтний спуск


@tf.function                                 # компіляція у граф: прискорює цикл навчання
def train_step(X_batch, y_batch):
    """Один крок спуску: прогноз -> втрата -> градієнти -> оновлення ваг."""
    with tf.GradientTape() as tape:                       # стрічка запису операцій
        y_pred = tf.matmul(X_batch, k) + b                # крок 4: модель y = X*k + b
        loss = tf.reduce_sum((y_batch - y_pred) ** 2)     # крок 4: СУМА квадратів відхилень
    grads = tape.gradient(loss, [k, b])                   # автоматичне диференціювання
    opt.apply_gradients(zip(grads, [k, b]))               # крок 5: оновлення k та b
    return loss                                           # повертаємо втрату для діагностики


hist_epoch, hist_loss, hist_k, hist_b = [], [], [], []    # історія для рисунків
t_start = time.perf_counter()                             # початок відліку часу
for i in range(num_steps):                                            # крок 6: цикл навчання
    indices = np.random.choice(n_samples, batch_size)                 # індекси міні-батча
    X_batch = tf.gather(X_tf, indices)                                # міні-батч входів
    y_batch = tf.gather(y_tf, indices)                                # міні-батч цілей
    loss_val = train_step(X_batch, y_batch)                           # один крок навчання
    k_s, b_s = float(k.numpy().ravel()[0]), float(b.numpy().ravel()[0])  # поточні параметри
    hist_epoch.append(i + 1)                                          # номер епохи
    hist_loss.append(float(loss_val))                                 # втрата
    hist_k.append(k_s)                                                # поточне k
    hist_b.append(b_s)                                                # поточне b
    if i < 5 or (i + 1) % display_step == 0:               # друк кожні 100 кроків, як у методичці
        print('Епоха %d: %.8f, k=%.4f, b=%.4f' % (i + 1, float(loss_val), k_s, b_s))
elapsed_tape = time.perf_counter() - t_start              # тривалість навчання, с
k_tape, b_tape = hist_k[-1], hist_b[-1]                   # підсумкові параметри

print('-' * 64)
print('TF2 (GradientTape): k = %.4f, b = %.4f, час навчання = %.2f с'
      % (k_tape, b_tape, elapsed_tape))

# ==========================================================================
# ВАРІАНТ Б. Високорівневий Keras: Sequential з одним шаром Dense(1).
# ==========================================================================
# Dense(1) реалізує рівно те саме перетворення output = dot(input, kernel) + bias,
# тобто kernel тут — це k, а bias — це b. Різниця лише в тому, що стандартна
# втрата 'mse' — це СЕРЕДНЄ, а не сума по батчу, тому градієнт у batch_size разів
# менший; щоб динаміка навчання збіглася з варіантом А, темп навчання збільшуємо
# рівно в batch_size разів.
keras.utils.set_random_seed(RANDOM_SEED)                              # відтворюваність
model = keras.Sequential([                                            # послідовна модель
    keras.layers.Input(shape=(1,)),                                   # вхід: один признак x
    keras.layers.Dense(1,                                             # один нейрон: y = kernel*x + bias
                       # УВАГА: рядковий псевдонім 'random_normal' у Keras — це
                       # RandomNormal(mean=0.0, stddev=0.05), тобто k стартувало б
                       # практично з нуля. Щоб старт збігався з варіантом А
                       # (tf.random.normal -> стандартний нормальний розподіл),
                       # задаємо ініціалізатор явно зі stddev = 1.0.
                       kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
                       bias_initializer='zeros')                      # b = 0
], name='linear_regression')
model.compile(optimizer=keras.optimizers.SGD(learning_rate=LEARNING_RATE * batch_size),
              loss='mse')                                             # налаштування навчання
model.summary()                                                       # зведення про модель

epochs_keras = num_steps // (n_samples // batch_size)                 # 20000 оновлень = 2000 епох
t_start = time.perf_counter()                                         # початок відліку часу
history = model.fit(X_data, y_data, batch_size=batch_size,            # навчання
                    epochs=epochs_keras, shuffle=True, verbose=0)
elapsed_keras = time.perf_counter() - t_start                         # тривалість навчання, с
w, bias = model.get_weights()                                         # ваги шару Dense
k_keras, b_keras = float(w.ravel()[0]), float(bias.ravel()[0])        # k та b моделі Keras

print('-' * 64)
print('Keras Sequential(Dense(1)): k = %.4f, b = %.4f, час навчання = %.2f с (%d епох)'
      % (k_keras, b_keras, elapsed_keras, epochs_keras))
print('Остання mse: %.4f  (mse * batch_size = %.1f -> та сама шкала, що й loss у TF1)'
      % (history.history['loss'][-1], history.history['loss'][-1] * batch_size))

# --------------------------------------------------------------------------
# Зведена таблиця результатів.
# --------------------------------------------------------------------------
tf1 = None                                                             # результати першого скрипта
tf1_path = os.path.join(BASE_DIR, 'result_tf1.npz')
if os.path.exists(tf1_path):                                           # якщо файл 1 уже відпрацював
    tf1 = np.load(tf1_path)

print('=' * 64)
print('%-28s %9s %9s %10s' % ('Реалізація', 'k', 'b', 'час, с'))
if tf1:
    print('%-28s %9.4f %9.4f %10.2f' % ('TF1-compat (Session)', tf1['k'], tf1['b'], tf1['time']))
print('%-28s %9.4f %9.4f %10.2f' % ('TF2 (GradientTape)', k_tape, b_tape, elapsed_tape))
print('%-28s %9.4f %9.4f %10.2f' % ('Keras Sequential(Dense)', k_keras, b_keras, elapsed_keras))
print('%-28s %9.4f %9.4f %10s' % ('Аналітичний МНК', ols_k, ols_b, '-'))
print('%-28s %9.4f %9.4f %10s' % ('Істинні параметри', 2.0, 1.0, '-'))
print('=' * 64)

# --------------------------------------------------------------------------
# Скільки живе слід різної початкової ініціалізації k?
# --------------------------------------------------------------------------
# TF1 і TF2 мають РІЗНІ генератори випадкових чисел (tf.set_random_seed проти
# tf.random.set_seed), тому стартові k не збігаються. Дані й послідовність
# міні-батчів однакові, тож уся різниця траєкторій — це слід старту. Нижче
# показано, що в k він згасає швидко («швидка» мода), а в b — на порядок
# повільніше («повільна» мода з власним числом 34.3).
if tf1:
    print('Слід різної початкової ініціалізації k (TF1 проти TF2):')
    print('%-8s %18s %18s' % ('крок', 'різниця |dk|', 'різниця |db|'))
    for st in (1, 60, 100, 1000, 4000, 10000, 20000):
        dk = abs(float(tf1['k_hist'][st - 1]) - hist_k[st - 1])        # розбіжність у нахилі
        db = abs(float(tf1['b_hist'][st - 1]) - hist_b[st - 1])        # розбіжність у зсуві
        print('%-8d %18.6f %18.6f' % (st, dk, db))

    # Контроль: чи може ця різниця бути похибкою накопичення float32?
    # Повторюємо ТОЙ САМИЙ спуск у numpy двічі — міняємо лише тип (float32/float64)
    # при однаковому старті, а потім лише старт при однаковому типі.
    def _sgd(dtype, k0):
        """Той самий SGD, але чистим numpy — щоб керувати типом і стартом."""
        np.random.seed(RANDOM_SEED)                                    # той самий потік випадковості
        Xn = np.random.uniform(1, 10, (n_samples, 1)).astype(dtype)    # ті самі дані
        yn = (2 * Xn + 1 + np.random.normal(0, 2, (n_samples, 1)).astype(dtype)).astype(dtype)
        kk, bb = dtype(k0), dtype(0.0)                                 # старт
        for _ in range(num_steps):
            jj = np.random.choice(n_samples, batch_size)               # той самий потік індексів
            xb, yb = Xn[jj, 0], yn[jj, 0]
            rr = yb - (kk * xb + bb)                                   # нев'язка
            kk = dtype(kk + dtype(LEARNING_RATE) * dtype(2) * rr.dot(xb))
            bb = dtype(bb + dtype(LEARNING_RATE) * dtype(2) * rr.sum())
        return float(kk), float(bb)

    k_init_tf1 = -1.8647                                               # стартове k у TF1 (див. файл 1)
    b32 = _sgd(np.float32, k_init_tf1)[1]                              # float32, старт TF1
    b64 = _sgd(np.float64, k_init_tf1)[1]                              # float64, той самий старт
    b64_tf2 = _sgd(np.float64, k_init_tf2)[1]                          # float64, старт TF2
    print('Контроль причини розбіжності b у 4-му знаку:')
    print('  лише тип обчислень (float32 проти float64, однаковий старт): |db| = %.2e' % abs(b32 - b64))
    print('  лише початкове k (%.4f проти %.4f, однаковий тип):           |db| = %.2e'
          % (k_init_tf1, k_init_tf2, abs(b64 - b64_tf2)))
    print('  відтворені значення: b = %.6f (старт TF1) проти b = %.6f (старт TF2)' % (b64, b64_tf2))
    print('=' * 64)

# --------------------------------------------------------------------------
# Рисунок 4. Хмара точок і підігнані прямі (всі три реалізації).
# --------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_data, y_data, s=8, alpha=0.35, color='#4C72B0', label='дані (1000 точок)')
x_line = np.linspace(X_data.min(), X_data.max(), 100)
plt.plot(x_line, 2 * x_line + 1, 'k--', lw=2, label='істинна: y = 2x + 1')
plt.plot(x_line, k_tape * x_line + b_tape, color='#C44E52', lw=2,
         label='TF2 GradientTape: y = %.3fx + %.3f' % (k_tape, b_tape))
plt.plot(x_line, k_keras * x_line + b_keras, color='#DD8452', lw=2, ls='-.',
         label='Keras Dense: y = %.3fx + %.3f' % (k_keras, b_keras))
plt.plot(x_line, ols_k * x_line + ols_b, color='#55A868', lw=1.5, ls=':',
         label='МНК: y = %.3fx + %.3f' % (ols_k, ols_b))
plt.xlabel('x'); plt.ylabel('y')
plt.title('Завдання 2 (TF2): лінійна регресія, зерно = 3 (номер у списку групи)')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig4_tf2_scatter_fit.png'), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# Рисунок 5. Спадання функції втрат (TF2 GradientTape та Keras).
# --------------------------------------------------------------------------
loss_arr = np.array(hist_loss)                                      # історія втрат TF2
win = 200                                                           # вікно ковзного середнього
loss_smooth = np.convolve(loss_arr, np.ones(win) / win, mode='valid')
keras_loss = np.array(history.history['loss']) * batch_size         # mse -> сума, щоб шкали збіглись
keras_steps = np.arange(1, epochs_keras + 1) * (n_samples // batch_size)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(hist_epoch[:300], loss_arr[:300], color='#C44E52', lw=1.2)
ax[0].axhline(batch_size * 4, color='gray', ls='--', lw=1.5)
ax[0].set_yscale('log'); ax[0].set_xlabel('крок'); ax[0].set_ylabel('loss (лог. шкала)')
ax[0].set_title('GradientTape, перші 300 кроків'); ax[0].grid(alpha=0.3, which='both')

ax[1].plot(hist_epoch, loss_arr, color='#C44E52', lw=0.4, alpha=0.2, label='GradientTape (міні-батч)')
ax[1].plot(hist_epoch[win - 1:], loss_smooth, color='#8B1A1A', lw=1.8, label='GradientTape, ковзне серед.')
ax[1].plot(keras_steps, keras_loss, color='#DD8452', lw=1.2, label='Keras Dense (mse x batch)')
ax[1].axhline(batch_size * 4, color='gray', ls='--', lw=1.5, label='межа batch*sigma^2 = 400')
ax[1].set_yscale('log'); ax[1].set_xlabel('крок'); ax[1].set_ylabel('loss (лог. шкала)')
ax[1].set_title('усі 20000 кроків'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')

fig.suptitle('Завдання 2 (TF2): спадання функції втрат')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, 'fig5_tf2_loss.png'), dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# Рисунок 6. Збіжність k та b (TF2).
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, sl, ttl in ((ax[0], slice(0, 300), 'перші 300 кроків'),
                   (ax[1], slice(None), 'усі 20000 кроків')):
    a.plot(hist_epoch[sl], hist_k[sl], color='#4C72B0', lw=1.3, label='k')
    a.plot(hist_epoch[sl], hist_b[sl], color='#55A868', lw=1.3, label='b')
    a.axhline(2.0, color='#4C72B0', ls='--', lw=1, label='істинне k = 2')
    a.axhline(1.0, color='#55A868', ls='--', lw=1, label='істинне b = 1')
    a.axhline(ols_k, color='#1F3B63', ls=':', lw=1.4, label='МНК k = %.4f' % ols_k)
    a.axhline(ols_b, color='#1E5631', ls=':', lw=1.4, label='МНК b = %.4f' % ols_b)
    a.set_xlabel('крок'); a.set_ylabel('значення параметра'); a.set_title(ttl); a.grid(alpha=0.3)
ax[1].legend(fontsize=8, loc='center right')
fig.suptitle('Завдання 2 (TF2 GradientTape): збіжність k та b')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, 'fig6_tf2_kb.png'), dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# Рисунок 7. Пряме порівняння TF1-compat та TF2.
# --------------------------------------------------------------------------
if tf1:
    tf1_loss = np.array(tf1['loss'])                                # історія втрат TF1
    tf1_smooth = np.convolve(tf1_loss, np.ones(win) / win, mode='valid')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(tf1['epoch'][win - 1:], tf1_smooth, color='#4C72B0', lw=1.6, label='TF1-compat')
    ax[0].plot(hist_epoch[win - 1:], loss_smooth, color='#C44E52', lw=1.6, label='TF2 GradientTape')
    ax[0].axhline(batch_size * 4, color='gray', ls='--', lw=1.5, label='межа = 400')
    ax[0].set_yscale('log'); ax[0].set_xlabel('крок'); ax[0].set_ylabel('loss (ковзне середнє)')
    ax[0].set_title('функція втрат'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')

    ax[1].plot(tf1['epoch'], tf1['k_hist'], color='#4C72B0', lw=1.3, label='TF1-compat')
    ax[1].plot(hist_epoch, hist_k, color='#C44E52', lw=1.3, label='TF2 GradientTape')
    ax[1].axhline(ols_k, color='k', ls=':', lw=1.4, label='МНК k = %.4f' % ols_k)
    ax[1].set_ylim(1.8, 2.3); ax[1].set_xlabel('крок'); ax[1].set_ylabel('k')
    ax[1].set_title('коефіцієнт k'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ax[2].plot(tf1['epoch'], tf1['b_hist'], color='#4C72B0', lw=1.3, label='TF1-compat')
    ax[2].plot(hist_epoch, hist_b, color='#C44E52', lw=1.3, label='TF2 GradientTape')
    ax[2].axhline(ols_b, color='k', ls=':', lw=1.4, label='МНК b = %.4f' % ols_b)
    ax[2].set_xlabel('крок'); ax[2].set_ylabel('b')
    ax[2].set_title('коефіцієнт b'); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle('Порівняння реалізацій: TensorFlow 1.x (compat) та TensorFlow 2.x')
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'fig7_compare.png'), dpi=150)
    plt.close(fig)
    print('Рисунки збережено: fig4..fig7')
else:
    print('УВАГА: result_tf1.npz не знайдено — спочатку запустіть LR_8_task_1_tf1_compat.py')
