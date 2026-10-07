# -*- coding: utf-8 -*-
"""
Лабораторна робота №8 (№10 за методичкою)
Тема: «Ресурси Keras. TensorFlow. Навчання лінійної регресії»
Завдання: засобами TensorFlow реалізувати наведений у методичці код лінійної
регресії та дослідити структуру розрахункового алгоритму.

Файл 1 з 2: ДОСЛІВНЕ відтворення коду методички (TensorFlow 1.x API).
Студент: Камінський Олексій Дмитрович, група ВТ-23-2, варіант 3.

На машині встановлено TensorFlow 2.22, у якому немає ані tf.placeholder,
ані tf.Session, ані tf.train.GradientDescentOptimizer. Щоб відтворити саме
той граф обчислень, який описаний у методичці (статичний граф + сесія +
feed_dict), використовуємо шар сумісності tensorflow.compat.v1 і вимикаємо
поведінку другої версії.
"""

import os                                   # доступ до змінних оточення
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # прибираємо службові C++ повідомлення TF

import time                                 # вимірювання часу навчання
import matplotlib                           # бібліотека побудови графіків
matplotlib.use("Agg")                       # неінтерактивний бекенд: малюємо у файл, без вікон
import matplotlib.pyplot as plt             # основний інтерфейс побудови рисунків

import numpy as np                          # робота з масивами та генерація даних
import tensorflow.compat.v1 as tf           # шар сумісності з TensorFlow 1.x
tf.disable_v2_behavior()                    # вимикаємо eager-режим TF2 -> працює статичний граф

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # тека, де лежить цей скрипт

# --------------------------------------------------------------------------
# Крок 1. Генерація вхідних даних.
# --------------------------------------------------------------------------
# У методичці: 1000 точок, y = 2x + 1 + eps, eps ~ N(0; 2).
# Увага: у ТЕКСТІ методички сказано «рівномірно на інтервалі [0; 1]», але в
# КОДІ методички стоїть np.random.uniform(1, 10). Беремо код (інтервал [1; 10]),
# бо саме він дає наведений у методичці зразок виводу.
RANDOM_SEED = 3                             # номер варіанта (3) як зерно генератора -> відтворюваність
np.random.seed(RANDOM_SEED)                 # фіксуємо генератор numpy
tf.set_random_seed(RANDOM_SEED)             # фіксуємо генератор TensorFlow (ініціалізація k)

n_samples, batch_size, num_steps = 1000, 100, 20000   # обсяг вибірки, розмір міні-батча, кількість кроків
X_data = np.random.uniform(1, 10, (n_samples, 1))     # 1000 точок x, рівномірно на [1; 10]
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))  # «правильна відповідь» з шумом N(0; 2)

# Еталон для перевірки: аналітичний розв'язок МНК (нормальні рівняння).
A_ols = np.hstack([X_data, np.ones_like(X_data)])     # матриця плану [x, 1]
ols_k, ols_b = np.linalg.lstsq(A_ols, y_data, rcond=None)[0].ravel()  # точний розв'язок МНК

# --------------------------------------------------------------------------
# Крок 2. Заглушки (placeholder) для вхідних даних.
# --------------------------------------------------------------------------
# Розмірність (розмір міні-батча x 1) — так само, як у методичці.
X = tf.placeholder(tf.float32, shape=(batch_size, 1))  # заглушка для входів x
y = tf.placeholder(tf.float32, shape=(batch_size, 1))  # заглушка для цільових значень y

# --------------------------------------------------------------------------
# Крок 3. Змінні моделі k та b у власному просторі імен.
# --------------------------------------------------------------------------
with tf.variable_scope('linear-regression'):                       # простір імен змінних
    k = tf.Variable(tf.random_normal((1, 1)), name='slope')        # нахил: стандартний нормальний розподіл
    b = tf.Variable(tf.zeros((1,)), name='bias')                   # зсув: ініціалізація нулем

# --------------------------------------------------------------------------
# Крок 4. Модель та функція втрат.
# --------------------------------------------------------------------------
y_pred = tf.matmul(X, k) + b                      # прогноз нейрона: y = X*k + b
loss = tf.reduce_sum((y - y_pred) ** 2)           # СУМА квадратів відхилень саме засобами TF (не numpy)

# --------------------------------------------------------------------------
# Крок 5. Оптимізатор — стохастичний градієнтний спуск.
# --------------------------------------------------------------------------
# У методичці GradientDescentOptimizer() викликано БЕЗ аргументів, але
# learning_rate є обов'язковим параметром. Оскільки loss — це СУМА по 100
# елементах батча, крок має бути малим, інакше спуск розбігається
# (межа стійкості для цих даних ~2.6e-4).
# LEARNING_RATE = 1e-5 — робочий темп (3.8 % від межі стійкості): швидкий і стійкий.
# LR_REFERENCE   = 1.5e-6 — калібрований темп, при якому НАШ прогін відтворює
# зразок виводу методички на 100-й епосі (див. контрольний прогін нижче).
LEARNING_RATE = 1e-5                                                      # темп навчання
LR_REFERENCE = 1.5e-6                                                     # темп для звірки зі зразком методички
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)  # операція оновлення k та b

# --------------------------------------------------------------------------
# Крок 6. Цикл навчання з міні-батчами та feed_dict.
# --------------------------------------------------------------------------
display_step = 100                                 # друкуємо діагностику кожні 100 епох (як у методичці)
# Історію пишемо НА КОЖНОМУ кроці, а не кожні 100: основне спадання втрат
# відбувається за перші ~60 ітерацій, і з кроком 100 воно просто не потрапило б
# на рисунок. На друк це не впливає — друкуємо рівно кожні 100 епох, як у методичці
# (додатково, понад методичку, друкуємо перші 5 кроків, щоб побачити старт спуску).
hist_epoch, hist_loss, hist_k, hist_b = [], [], [], []   # історія для рисунків

with tf.Session() as sess:                         # відкриваємо сесію — середовище виконання графа
    # У методичці написано tf.initialize_global_variables() — такої функції не існує.
    # Правильна назва — tf.global_variables_initializer().
    sess.run(tf.global_variables_initializer())    # ініціалізуємо всі змінні графа
    k_init = float(k.eval()[0][0])                 # стартове (випадкове) значення k — знадобиться далі
    t_start = time.perf_counter()                  # початок відліку часу навчання
    for i in range(num_steps):                                      # головний цикл навчання
        indices = np.random.choice(n_samples, batch_size)           # випадкові індекси міні-батча
        X_batch, y_batch = X_data[indices], y_data[indices]         # сам міні-батч
        _, loss_val, k_val, b_val = sess.run(                       # один крок спуску + зчитування змінних
            [optimizer, loss, k, b],                                # що саме обчислити
            feed_dict={X: X_batch, y: y_batch})                     # підстановка даних у заглушки
        if i < 5:                                                   # ДОДАТКОВО до методички: перші кроки
            print('Крок %d: %.8f, k=%.4f, b=%.4f'
                  % (i + 1, loss_val, float(np.ravel(k_val)[0]), float(np.ravel(b_val)[0])))
        # k має форму (1,1), b — форму (1,), тому розгортаємо їх у скаляри: у методичці
        # '%.4f' % k_val застосовано напряму до масиву, що в сучасному numpy (>=1.25)
        # викликає TypeError.
        k_s, b_s = float(np.ravel(k_val)[0]), float(np.ravel(b_val)[0])
        hist_epoch.append(i + 1)                                    # номер епохи
        hist_loss.append(float(loss_val))                           # втрата на цьому міні-батчі
        hist_k.append(k_s)                                          # поточне k
        hist_b.append(b_s)                                          # поточне b
        if (i + 1) % display_step == 0:                             # кожні 100 епох — друк, як у методичці
            print('Епоха %d: %.8f, k=%.4f, b=%.4f' % (i + 1, loss_val, k_s, b_s))
    elapsed = time.perf_counter() - t_start        # тривалість навчання, с
    k_final, b_final = float(k.eval()[0][0]), float(b.eval()[0])    # підсумкові параметри моделі

print('-' * 64)
print('TF1-compat: k = %.4f, b = %.4f, час навчання = %.2f с' % (k_final, b_final, elapsed))
print('Аналітичний МНК (еталон): k = %.4f, b = %.4f' % (ols_k, ols_b))
print('Істинні параметри генератора: k = 2.0000, b = 1.0000')
print('Фінальна втрата на міні-батчі: %.4f (теоретична межа ~ batch*sigma^2 = %d)'
      % (hist_loss[-1], batch_size * 4))
print('Стартове значення k (випадкова ініціалізація): %.4f' % k_init)

# --------------------------------------------------------------------------
# КОНТРОЛЬНИЙ ПРОГІН: звірка зі зразком виводу методички.
# --------------------------------------------------------------------------
# Методичка наводить зразок виводу («Епоха 100: 5962.15625000, k=0.8925, b=0.0988»),
# але НЕ наводить learning_rate. Наш робочий прогін з LEARNING_RATE=1e-5 дає на
# 100-й епосі loss ~500 — у 12 разів менше, бо наш крок більший. Щоб показати, що
# розбіжність спричинена САМЕ темпом навчання, повторюємо той самий граф, ті самі
# дані й ту саму послідовність міні-батчів із меншим кроком LR_REFERENCE.
METH_SAMPLE = {                                   # зразок виводу з методички
    100: (5962.15625000, 0.8925, 0.0988),
    200: (5312.11621094, 0.9862, 0.1927),
    300: (3904.57006836, 1.0761, 0.2825),
    19900: (429.79974365, 2.0267, 0.9006),
    20000: (378.41503906, 2.0179, 0.8902),
}

def rerun_with_lr(lr, steps):
    """Повторює ТОЙ САМИЙ прогін (ті самі дані, те саме стартове k, та сама
    послідовність міні-батчів), змінюючи лише темп навчання. Повертає історії."""
    np.random.seed(RANDOM_SEED)                   # повертаємо генератор numpy у вихідний стан
    np.random.uniform(1, 10, (n_samples, 1))      # «прокручуємо» ті самі два виклики, що й на кроці 1,
    np.random.normal(0, 2, (n_samples, 1))        # щоб потік індексів міні-батчів збігся з основним прогоном
    g = tf.Graph()                                # окремий граф, щоб не чіпати основний
    with g.as_default():
        Xr = tf.placeholder(tf.float32, shape=(batch_size, 1))              # ті самі заглушки
        yr = tf.placeholder(tf.float32, shape=(batch_size, 1))
        kr = tf.Variable(np.array([[k_init]], dtype=np.float32), name='slope')  # те саме стартове k
        br = tf.Variable(tf.zeros((1,)), name='bias')                       # і те саме b = 0
        loss_r = tf.reduce_sum((yr - (tf.matmul(Xr, kr) + br)) ** 2)        # та сама функція втрат
        opt_r = tf.train.GradientDescentOptimizer(lr).minimize(loss_r)      # відрізняється лише темп
        l_h, k_h, b_h = [], [], []                                          # історії
        with tf.Session(graph=g) as s_r:
            s_r.run(tf.global_variables_initializer())
            for _i in range(steps):
                idx = np.random.choice(n_samples, batch_size)               # той самий потік індексів
                _, lv, kv, bv = s_r.run([opt_r, loss_r, kr, br],            # той самий порядок вибірки
                                        feed_dict={Xr: X_data[idx], yr: y_data[idx]})
                l_h.append(float(lv))
                k_h.append(float(np.ravel(kv)[0]))
                b_h.append(float(np.ravel(bv)[0]))
    return l_h, k_h, b_h


# Крок А. Скануємо темп навчання по 100 кроків і шукаємо той, що дає втрату
# методички на 100-й епосі (5962.16).
print('=' * 78)
print('Підбір темпу навчання під зразок методички (ціль: epoch 100 -> loss %.2f, k %.4f):'
      % (METH_SAMPLE[100][0], METH_SAMPLE[100][1]))
for lr_try in (5e-7, 1e-6, 1.5e-6, 2e-6, 5e-6, 1e-5):
    l_h, k_h, b_h = rerun_with_lr(lr_try, 100)                    # 100 кроків — цього досить для звірки
    print('  lr = %7.1e -> епоха 100: loss = %10.2f, k = %7.4f, b = %6.4f'
          % (lr_try, l_h[-1], k_h[-1], b_h[-1]))

# Крок Б. Повний контрольний прогін на відібраному темпі LR_REFERENCE.
ref_loss, ref_k, ref_b = rerun_with_lr(LR_REFERENCE, num_steps)

print('-' * 78)
print('Звірка зі зразком виводу методички (learning_rate у методичці не вказано):')
print('%-7s | %-28s | %-28s | %s' % ('епоха', 'зразок методички',
                                     'наш прогін, lr=%.1e' % LR_REFERENCE,
                                     'наш прогін, lr=%.1e' % LEARNING_RATE))
for ep in sorted(METH_SAMPLE):
    ml, mk, mb = METH_SAMPLE[ep]
    rl, rk, rb = ref_loss[ep - 1], ref_k[ep - 1], ref_b[ep - 1]
    wl, wk, wb = hist_loss[ep - 1], hist_k[ep - 1], hist_b[ep - 1]
    print('%-7d | %10.2f k=%.4f b=%.4f | %10.2f k=%.4f b=%.4f | %10.2f k=%.4f b=%.4f'
          % (ep, ml, mk, mb, rl, rk, rb, wl, wk, wb))
print('=' * 78)

# Чому стовпчик b у зразку методички не може належати тому самому прогону:
# для x ~ U[1; 10] градієнт по k приблизно в 6.6 раза більший за градієнт по b,
# отже за 100 кроків k має зростати приблизно в 6.6 раза швидше за b. У зразку ж
# методички прирости рівні (k: +0.0937, b: +0.0939 за епохи 100 -> 200).
_k_m, _b_m = METH_SAMPLE[100][1], METH_SAMPLE[100][2]                   # стан із зразка методички
_num = (2 - _k_m) * float((X_data ** 2).mean()) + (1 - _b_m) * float(X_data.mean())
_den = (2 - _k_m) * float(X_data.mean()) + (1 - _b_m)
print('Теоретичне співвідношення градієнтів dL/dk : dL/db у точці зразка = %.2f' % (_num / _den))
print('Фактичне співвідношення приростів у зразку методички (епохи 100->200) = %.2f'
      % ((METH_SAMPLE[200][1] - _k_m) / (METH_SAMPLE[200][2] - _b_m)))

# --------------------------------------------------------------------------
# КОНТРОЛЬНА ПЕРЕВІРКА: інтервал [0; 1] з ТЕКСТУ методички проти [1; 10] з КОДУ.
# --------------------------------------------------------------------------
def _step_eigenvalues(x_col):
    """Власні числа матриці одного кроку спуску (на одиницю learning_rate)."""
    H = 2 * batch_size * np.array([[float((x_col ** 2).mean()), float(x_col.mean())],
                                   [float(x_col.mean()), 1.0]])
    return np.linalg.eigvalsh(H)[::-1]

np.random.seed(RANDOM_SEED)                                   # той самий seed, інший інтервал
X0 = np.random.uniform(0, 1, (n_samples, 1))                  # «1000 точок на [0; 1]» з тексту методички
y0 = 2 * X0 + 1 + np.random.normal(0, 2, (n_samples, 1))      # ті самі y = 2x + 1 + шум
ev1, ev0 = _step_eigenvalues(X_data), _step_eigenvalues(X0)   # власні числа для обох інтервалів
ols0 = np.linalg.lstsq(np.hstack([X0, np.ones_like(X0)]), y0, rcond=None)[0].ravel()

print('-' * 78)
print('Інтервал [1; 10] (код методички): Var(x)=%.4f, E[x^2]=%.4f, власні числа %.2f і %.2f, межа %.2e'
      % (X_data.var(), (X_data ** 2).mean(), ev1[0], ev1[1], 2 / ev1[0]))
print('Інтервал [0; 1]  (текст методички): Var(x)=%.4f, E[x^2]=%.4f, власні числа %.2f і %.2f, межа %.2e'
      % (X0.var(), (X0 ** 2).mean(), ev0[0], ev0[1], 2 / ev0[0]))
print('Відношення: дисперсія у %.1f раза, E[x^2] у %.1f раза, межа стійкості у %.1f раза'
      % (X_data.var() / X0.var(), (X_data ** 2).mean() / (X0 ** 2).mean(), (2 / ev0[0]) / (2 / ev1[0])))
print('МНК на [0; 1]: k = %.4f, b = %.4f' % (ols0[0], ols0[1]))

# Головний аргумент на користь інтервалу [1; 10]: при значеннях k і b зі зразка
# методички на інтервалі [0; 1] втрата фізично не може дорівнювати 5962.
def _loss_at(x_col, y_col, k_v, b_v):
    """Очікувана втрата на міні-батчі зі 100 точок у стані (k, b)."""
    return float(np.mean((y_col - (k_v * x_col + b_v)) ** 2)) * batch_size

print('У стані зі зразка методички (k=%.4f, b=%.4f) очікувана втрата: на [1; 10] = %.0f, на [0; 1] = %.0f'
      % (METH_SAMPLE[100][1], METH_SAMPLE[100][2],
         _loss_at(X_data, y_data, *METH_SAMPLE[100][1:]),
         _loss_at(X0, y0, *METH_SAMPLE[100][1:])))
print('Методичка наводить %.2f -> узгоджується лише з інтервалом [1; 10] з її ж коду.'
      % METH_SAMPLE[100][0])

# Контроль твердження «на [0; 1] спуск не встигає збігтися»: перевіряємо прямо.
k0_, b0_ = k_init, 0.0                                        # той самий старт, що й у основному прогоні
np.random.seed(RANDOM_SEED)
np.random.uniform(0, 1, (n_samples, 1)); np.random.normal(0, 2, (n_samples, 1))   # той самий потік індексів
for lr_try in (LEARNING_RATE, 1e-3):
    kk, bb = k0_, b0_
    for _ in range(num_steps):                                # той самий SGD, але на даних з [0; 1]
        idx = np.random.choice(n_samples, batch_size)
        xb, yb = X0[idx, 0], y0[idx, 0]
        r = yb - (kk * xb + bb)
        kk += lr_try * 2 * float(np.sum(xb * r))
        bb += lr_try * 2 * float(np.sum(r))
    print('На [0; 1] за 20000 кроків з lr=%.0e: k = %.4f, b = %.4f (МНК: %.4f, %.4f)'
          % (lr_try, kk, bb, ols0[0], ols0[1]))

# --------------------------------------------------------------------------
# Рисунок 1. Хмара точок і підігнана пряма.
# --------------------------------------------------------------------------
plt.figure(figsize=(8, 5))                                          # нове полотно
plt.scatter(X_data, y_data, s=8, alpha=0.35, color='#4C72B0', label='дані (1000 точок)')
x_line = np.linspace(X_data.min(), X_data.max(), 100)               # сітка по осі x для прямих
plt.plot(x_line, 2 * x_line + 1, 'k--', lw=2, label='істинна: y = 2x + 1')
plt.plot(x_line, k_final * x_line + b_final, color='#C44E52', lw=2,
         label='TF1-compat: y = %.3fx + %.3f' % (k_final, b_final))
plt.xlabel('x'); plt.ylabel('y')                                     # підписи осей
plt.title('Завдання 1 (TF1-compat): лінійна регресія, зерно = 3 (номер у списку групи)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig1_tf1_scatter_fit.png'), dpi=150)
plt.close()

# --------------------------------------------------------------------------
# Рисунок 2. Крива спадання функції втрат.
# --------------------------------------------------------------------------
loss_arr = np.array(hist_loss)                                      # історія втрат як масив
win = 200                                                           # вікно ковзного середнього
loss_smooth = np.convolve(loss_arr, np.ones(win) / win, mode='valid')  # згладжена крива
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))                     # два підграфіки

ax[0].plot(hist_epoch[:300], loss_arr[:300], color='#C44E52', lw=1.2)   # старт навчання
ax[0].axhline(batch_size * 4, color='gray', ls='--', lw=1.5)
ax[0].set_yscale('log'); ax[0].set_xlabel('епоха'); ax[0].set_ylabel('loss (лог. шкала)')
ax[0].set_title('перші 300 епох: основне спадання'); ax[0].grid(alpha=0.3, which='both')

ax[1].plot(hist_epoch, loss_arr, color='#C44E52', lw=0.4, alpha=0.25, label='loss на міні-батчі')
ax[1].plot(hist_epoch[win - 1:], loss_smooth, color='#8B1A1A', lw=1.8,
           label='ковзне середнє (200)')
ax[1].axhline(batch_size * 4, color='gray', ls='--', lw=1.5,
              label='межа batch*sigma^2 = %d' % (batch_size * 4))
ax[1].set_yscale('log'); ax[1].set_xlabel('епоха'); ax[1].set_ylabel('loss (лог. шкала)')
ax[1].set_title('усі 20000 епох'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')

fig.suptitle('Завдання 1 (TF1-compat): спадання функції втрат')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, 'fig2_tf1_loss.png'), dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------
# Рисунок 3. Збіжність коефіцієнтів k та b.
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))                     # два підграфіки
for a, sl, ttl in ((ax[0], slice(0, 300), 'перші 300 епох'),
                   (ax[1], slice(None), 'усі 20000 епох')):
    a.plot(hist_epoch[sl], hist_k[sl], color='#4C72B0', lw=1.3, label='k')
    a.plot(hist_epoch[sl], hist_b[sl], color='#55A868', lw=1.3, label='b')
    a.axhline(2.0, color='#4C72B0', ls='--', lw=1, label='істинне k = 2')
    a.axhline(1.0, color='#55A868', ls='--', lw=1, label='істинне b = 1')
    a.axhline(ols_k, color='#1F3B63', ls=':', lw=1.4, label='МНК k = %.4f' % ols_k)
    a.axhline(ols_b, color='#1E5631', ls=':', lw=1.4, label='МНК b = %.4f' % ols_b)
    a.set_xlabel('епоха'); a.set_ylabel('значення параметра')
    a.set_title(ttl); a.grid(alpha=0.3)
ax[1].legend(fontsize=8, loc='center right')
fig.suptitle('Завдання 1 (TF1-compat): збіжність k та b')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, 'fig3_tf1_kb.png'), dpi=150)
plt.close(fig)

# Зберігаємо результати у стислий .npz, щоб другий скрипт міг побудувати
# порівняльний рисунок (fig7) без повторного навчання.
np.savez_compressed(os.path.join(BASE_DIR, 'result_tf1.npz'),
                    k=k_final, b=b_final, time=elapsed, lr=LEARNING_RATE,
                    ols_k=ols_k, ols_b=ols_b,
                    epoch=np.array(hist_epoch), loss=np.array(hist_loss, dtype=np.float32),
                    k_hist=np.array(hist_k, dtype=np.float32),
                    b_hist=np.array(hist_b, dtype=np.float32))

# --------------------------------------------------------------------------
# Рисунок 8. Вплив темпу навчання: наш робочий прогін проти зразка методички.
# --------------------------------------------------------------------------
meth_ep = sorted(e for e in METH_SAMPLE if e <= 500)                # епохи зразка на старті навчання
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))                     # два підграфіки

ax[0].plot(hist_epoch[:500], hist_loss[:500], color='#C44E52', lw=1.4,
           label='наш прогін, lr = %.0e' % LEARNING_RATE)
ax[0].plot(hist_epoch[:500], ref_loss[:500], color='#4C72B0', lw=1.4,
           label='контрольний прогін, lr = %.1e' % LR_REFERENCE)
ax[0].scatter(meth_ep, [METH_SAMPLE[e][0] for e in meth_ep], s=70, marker='X',
              color='k', zorder=5, label='зразок виводу методички')
ax[0].axhline(batch_size * 4, color='gray', ls='--', lw=1.3, label='межа batch*sigma^2 = 400')
ax[0].set_yscale('log'); ax[0].set_xlabel('епоха'); ax[0].set_ylabel('loss (лог. шкала)')
ax[0].set_title('функція втрат, перші 500 епох')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')

ax[1].plot(hist_epoch[:500], hist_k[:500], color='#C44E52', lw=1.4,
           label='наш прогін, lr = %.0e' % LEARNING_RATE)
ax[1].plot(hist_epoch[:500], ref_k[:500], color='#4C72B0', lw=1.4,
           label='контрольний прогін, lr = %.1e' % LR_REFERENCE)
ax[1].scatter(meth_ep, [METH_SAMPLE[e][1] for e in meth_ep], s=70, marker='X',
              color='k', zorder=5, label='зразок виводу методички')
ax[1].axhline(ols_k, color='#1F3B63', ls=':', lw=1.4, label='МНК k = %.4f' % ols_k)
ax[1].set_xlabel('епоха'); ax[1].set_ylabel('коефіцієнт k')
ax[1].set_title('коефіцієнт k, перші 500 епох')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

fig.suptitle('Звірка зі зразком виводу методички: роль темпу навчання')
fig.tight_layout()
fig.savefig(os.path.join(BASE_DIR, 'fig8_lr_vs_methodic.png'), dpi=150)
plt.close(fig)

print('Рисунки збережено: fig1_tf1_scatter_fit.png, fig2_tf1_loss.png, fig3_tf1_kb.png, '
      'fig8_lr_vs_methodic.png')
