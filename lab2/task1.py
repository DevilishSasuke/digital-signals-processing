import numpy as np
import matplotlib.pyplot as plt

# параметры исходного сигнала
f0 = 2.0  # частота синуса (Гц)
w0 = 2 * np.pi * f0  # циклическая частота
duration = 1.0  # длительность наблюдения (сек)

# создаем временную шкалу для непрерывного сигнала
t_cont = np.linspace(0, duration, 1000)
x_cont = np.sin(w0 * t_cont)

# дискретизация (отсчеты сигнала)
# по теореме Котельникова Fs > 2*f0
Fs = 10.0  # количество отсчетов в секунду
Ts = 1.0 / Fs  # шаг дискретизации (шаг по времени между отсчетами)

# моменты времени для дискретных точек
t_discr = np.arange(0, duration, Ts)
# получаем значения синуса в этих моментах
x_discr = np.sin(w0 * t_discr)

# восстановление сигнала
# создаем результирующий массив
x_rec = np.zeros_like(t_cont)

# проходим циклом по каждому отсчету сигнала
for i in range(len(t_discr)):
    # по каждому отсчету получаем функцию sinc
    # t_cont - t_discr[i] - расстояние по времени от текущего отсчёта до каждой из 1000 точек
    # / Ts - нормализация значения для принятия значения 1 в исходной точке
    # np.sinc - получаем веса влияния на каждую из точек сигнала
    # * x_discr[i] - масштабируем волну на высоту отсчёта
    sinc_component = x_discr[i] * np.sinc((t_cont - t_discr[i]) / Ts)
    x_rec += sinc_component

# визуализация
plt.figure(figsize=(10, 8))

# аналоговый сигнал
plt.subplot(3, 1, 1)
plt.plot(t_cont, x_cont, 'b-', linewidth=2)
plt.title('1. исходный непрерывный сигнал')
plt.grid(True)

# дискретные отсчеты
plt.subplot(3, 1, 2)
plt.stem(t_discr, x_discr, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title(f'2. дискретные отсчеты {Fs} Гц')
plt.grid(True)

# восстановленный сигнал
plt.subplot(3, 1, 3)
plt.plot(t_cont, x_rec, 'g-', linewidth=3, label='восстановленный')
plt.plot(t_cont, x_cont, 'b--', linewidth=1.5, label='исходный')
plt.title('3. восстановленный сигнал')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig('result1.png', dpi=300)
print("график сохранен в файл result1.png")