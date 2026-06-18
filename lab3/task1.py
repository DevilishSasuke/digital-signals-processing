import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firls, freqz, lfilter


def my_mnk_filter(N, freqs_norm, gains, num_points=4096):
    """
    реализация МНК для КИХ-фильтра с линейной ФЧХ
    freqs_norm - нормированные частоты [0, 1]
    gains - усиления
    num_points - число точек частотной сетки
    """
    # число коэффициентов и центральный индекс
    N_coefs = N + 1
    M = N // 2  # центр симметрии

    # равномерная сетка частот от 0 до 0.5 (в нормированных единицах [0, 1])
    # 0.5 соответствует частоте Котельникова, выше информации нет
    freq_grid = np.linspace(0, 0.5, num_points)

    # желаемая АЧХ на сетке, интерполируем для каждой точки сетки
    # freqs_norm идёт от 0 до 1, поэтому делим на 2 для совмещения с freq_grid
    D = np.interp(freq_grid, freqs_norm / 2, gains)

    # матрица косинусов A размером N_coefs * (M+1)
    # для КИХ с линейной ФЧХ - косинус, синус взаимно уничтожается
    # из-за симметрии коэффициентов h[k] = h[N-k]
    m = np.arange(M + 1)
    A = np.cos(2 * np.pi * np.outer(freq_grid, m))

    # МНК - находит такое a, чтобы минимизировать сумму ошибок по всем уравнениям
    a = np.linalg.lstsq(A, D, rcond=None)[0]

    # превращаем a в симметричный массив коэффициентов
    # центральный коэффициент — a[0]
    # делим на 2, т.к. пара симм. коэфов вносит двойной вклад в АЧХ
    h = np.zeros(N_coefs)
    h[M] = a[0]
    for m_idx in range(1, M + 1):
        h[M - m_idx] = a[m_idx] / 2  # левая половина
        h[M + m_idx] = a[m_idx] / 2  # правая половина

    return h

Fs = 8000 # частота дискретизации входного сигнала, ГЦ
F_kot = Fs / 2  # частота котельникова, содержащая полезную информацию, выше - колебания сливаются с другими частицами
N = 256  # порядок фильтра, количество прошлых отсчетов (число коэффициентов = N+1)

freqs = np.array([
    0, 49, # до диапазона 1
    50,  150,  # диапазон 1, 50 - 150 Гц
    151, 349, # между диапазоном 1 и диапазоном 2
    350,  750,  # диапазон 2
    751, 899, # между диапазоном 2 и диапазоном 3
    900, 1500, # диапазон 3
    1501, F_kot # после диапазона 3
]) / F_kot  # нормализуем все частоты к диапазону [0, 1]

# множители усиления для каждого диапазона
gains = np.array([
    0, 0, 
    2.0, 2.0, 
    0, 0, 
    1.0, 1.0, 
    0, 0, 
    0.5, 0.5, 
    0, 0
])

# ручная реализация МНК
h_manual = my_mnk_filter(N, freqs, gains)

# встроенная функция firls для сравнения
h_builtin = firls(N + 1, freqs * F_kot, gains, fs=Fs)

# АЧХ обоих фильтров
w_m, H_m = freqz(h_manual,  worN=8192)
w_b, H_b = freqz(h_builtin, worN=8192)

# нормируем частоты в Гц
freq_hz_m = w_m * F_kot / np.pi
freq_hz_b = w_b * F_kot / np.pi

amplitude_m = np.abs(H_m)
amplitude_b = np.abs(H_b)

# создаем временную шкалу для тестового сигнала, 800 отсчетов - 0.1 секунды сигнала
t = np.linspace(0, 0.1, int(Fs * 0.1), endpoint=False)

# сам сигнал
signal = (
    np.sin(2 * np.pi * 100  * t) +  # диап 1
    np.sin(2 * np.pi * 500  * t) +  # диап 2
    np.sin(2 * np.pi * 1200 * t) +  # диап 3
    np.sin(2 * np.pi * 2000 * t)    # вне диап
)

# применяем фильтр к тестовому сигналу
# свертка сигнала с коэффициентами h
filtered_signal = lfilter(h_manual, 1.0, signal)

# ----------------------------------------------------------------------------------------------
# визуализация
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# график 1: АЧХ собственного фильтра vs желаемая АЧХ
ax1 = axes[0]
ax1.plot(freq_hz_m, amplitude_m, 'b-', linewidth=2, label=f'Ручной фильтр')
ax1.step(freqs * F_kot, gains, 'r--', linewidth=1.5,
         label='Желаемая АЧХ', where='mid')
ax1.set_xlim(0, 2000)
ax1.set_ylim(-0.1, 2.5)
ax1.set_xlabel('Частота (Гц)')
ax1.set_ylabel('Усиление')
ax1.set_title('АЧХ ручного КИХ-фильтра')
ax1.legend()
ax1.grid(True, alpha=0.3)
for f in [50, 150, 350, 750, 900, 1500]:
    ax1.axvline(x=f, color='gray', linestyle=':', alpha=0.5)

# график 2: сравнение своего фильтра со встроенным
ax2 = axes[1]
ax2.plot(freq_hz_m,   amplitude_m,   'b-',  linewidth=2,
         label=f'Свой фильтр')
ax2.plot(freq_hz_b, amplitude_b, 'g--', linewidth=2,
         label=f'Встроенная функция')
ax2.set_xlim(0, 2000)
ax2.set_ylim(-0.1, 2.5)
ax2.set_xlabel('Частота (Гц)')
ax2.set_ylabel('Усиление')
ax2.set_title('Сравнение своего и встроенного фильтров')
ax2.legend()
ax2.grid(True, alpha=0.3)

# график 3: тестовый сигнал до и после фильтрации
ax3 = axes[2]
ax3.plot(t[:200], signal[:200],          'b-',  linewidth=1,
         label='Исходный сигнал', alpha=0.7)
ax3.plot(t[:200], filtered_signal[:200], 'r-',  linewidth=2,
         label='Отфильтрованный сигнал')
ax3.set_xlabel('Время (сек)')
ax3.set_ylabel('Амплитуда')
ax3.set_title('Тестовый сигнал до и после фильтрации')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'result1_{N}.png', dpi=150)

# результаты фильтрации
print("Результаты фильтрации")
print("-" * 50)
print(f"Частота дискретизации: {Fs} Гц")
print(f"Частота Найквиста: {F_kot} Гц")
print(f"Порядок фильтра: {N}")
print(f"Число коэффициентов: {N+1}")
print()

test_freqs = [100, 500, 1200, 2000]
expected   = [2.0, 1.0,  0.5,   0.0]
print("Тестовые усиления на частотах - Ручной фильтр")
for f, exp in zip(test_freqs, expected):
    idx = np.argmin(np.abs(freq_hz_m - f))
    print(f"\t{f:5d}Гц: ожидалось {exp:.1f}, получено {amplitude_m[idx]:.3f}")
 
print()
print("Тестовые усиления на частотах - Встроенная firls")
for f, exp in zip(test_freqs, expected):
    idx = np.argmin(np.abs(freq_hz_b - f))
    print(f"\t{f:5d}Гц: ожидалось {exp:.1f}, получено {amplitude_b[idx]:.3f}")

print()
print(f"Графики сохранены в result1_{N}.png")