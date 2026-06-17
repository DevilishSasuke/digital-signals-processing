import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2, freqz, lfilter

Fs = 8000 # частота дискретизации входного сигнала, ГЦ
F_kot = Fs / 2  # частота котельникова, содержащая полезную информацию, выше - колебания сливаются с другими частицами
N = 128  # порядок фильтра, количество прошлых отсчетов (число коэффициентов = N+1)

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

# firwin - МНК, находит такие h[n], что АЧХ фильтра максимально близка к желаемой
# получаем импульсную характеристику
h = firwin2(N + 1, freqs, gains, window='hamming')

# freqz вычисляет комплексную частотную характеристику фильтра
# на основе коэффициентов, преобразование фурье (из временной области в частотную)
# N - количество точек
w, H = freqz(h, worN=8192)
# w - нормированные от 0 до pi числа
# H - массив комплексных чисел с усилением и сдвигом

# переводим нормированные частоты обратно в Гц
freq_hz = w * F_kot / np.pi

# амплитудная характеристика, модуль комплексного H(ω), усиление фильтра
amplitude = np.abs(H)

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
# свертка сигнала с коэфициентами h
filtered_signal = lfilter(h, 1.0, signal)

# встроенная функция для сравнения
h_builtin = firwin2(N // 2 + 1, freqs, gains, window='hamming')
w_b, H_b = freqz(h_builtin, worN=8192)
freq_hz_b = w_b * F_kot / np.pi
amplitude_b = np.abs(H_b)

# ----------------------------------------------------------------------------------------------
# визуализация
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# график 1: АЧХ собственного фильтра vs желаемая АЧХ
ax1 = axes[0]
ax1.plot(freq_hz, amplitude, 'b-', linewidth=2, label=f'Свой фильтр')
ax1.step(freqs * F_kot, gains, 'r--', linewidth=1.5,
         label='Желаемая АЧХ', where='mid')
ax1.set_xlim(0, 2000)
ax1.set_ylim(-0.1, 2.5)
ax1.set_xlabel('Частота (Гц)')
ax1.set_ylabel('Усиление')
ax1.set_title('АЧХ своего КИХ-фильтра')
ax1.legend()
ax1.grid(True, alpha=0.3)
for f in [50, 150, 350, 750, 900, 1500]:
    ax1.axvline(x=f, color='gray', linestyle=':', alpha=0.5)

# график 2: сравнение своего фильтра со встроенным
ax2 = axes[1]
ax2.plot(freq_hz,   amplitude,   'b-',  linewidth=2,
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
print("Тестовые усиления на частотах")
for f, exp in zip(test_freqs, expected):
    idx = np.argmin(np.abs(freq_hz - f))
    real_gain = amplitude[idx]
    print(f"\t{f:5d}Гц: ожидалось {exp:.1f}, получено {real_gain:.3f}")

print()
print(f"Графики сохранены в result1_{N}.png")