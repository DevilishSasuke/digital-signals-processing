import numpy as np
import matplotlib.pyplot as plt

T = 0.1  # шаг дискретизации в секундах (то, как часто происходит опрос)
steps = 200  # количество выполняемых шагов
B = 0.5  # половина расстояния от колеса до колеса у робота

# дисперсии различных процессов робота
var_q = 0.01  # движение робота
var_r_pos = 0.5  # датчик координат x,y 
var_r_angle = 0.1  # датчик угла поворота r

# матрицы ковариации    
Q = np.diag([var_q, var_q, var_q*0.3]) # ковариация движения по x, y, r
R = np.diag([var_r_pos, var_r_pos, var_r_angle]) # ковариация датчиков коорд и угла

s1 = 50 # конец первого отрезка
s2 = 70 # конец второго отрезка
# задаём траекторию роботу
s_L_arr = np.zeros(steps)
s_R_arr = np.zeros(steps)
# сначала идёт езда по прямой - у колёс одинаковая скорость
s_L_arr[:s1] = 1.0
s_R_arr[:s1] = 1.0
# затем резкий поворот направо - левое колесо едет, правое почти стоит
s_L_arr[s1:s2] = 1.0
s_R_arr[s1:s2] = 0.2
# вырисовываем некотору дугу в движении - правое едет немного быстрее
# создавая плавный поворот в движении
s_L_arr[s2:] = 1.0
s_R_arr[s2:] = 1.2

# хранение данных
x_real = np.zeros((3, steps)) # истинная траектория
x_noise = np.zeros((3, steps)) # измерения с шумом
x_ekf = np.zeros((3, steps))  # оценка по расширенному фильтру калмана

# начальная позиция
x_real[:, 0] = [0, 0, 0]
x_noise[:, 0] = x_real[:, 0] + np.random.multivariate_normal([0, 0, 0], R)
x_ekf[:, 0] = [0, 0, 0]
P = np.eye(3) * 0.05 # матрица ковариации в положении

# получаем истинную траекторию
for k in range(steps - 1):
    # текущие скорости
    s_t = (s_R_arr[k] + s_L_arr[k]) / 2.0 # движение робота впредёт (передвижение)
    s_r = (s_R_arr[k] - s_L_arr[k]) / (2.0 * B) # поворот робота (вращение)
    
    # берём текущие истинные состояния
    x_curr = x_real[0, k]
    y_curr = x_real[1, k]
    r_curr = x_real[2, k]
    
    # следущее истинное положение робота
    # в сответствие с уравнениям дискретного движения
    x_next = x_curr + T * s_t * np.cos(r_curr) - 0.5 * T**2 * s_t * s_r * np.sin(r_curr)
    y_next = y_curr + T * s_t * np.sin(r_curr) + 0.5 * T**2 * s_t * s_r * np.cos(r_curr)
    r_next = r_curr + T * s_r
    
    # записываем истинные значения следующего шага
    x_real[:, k+1] = [x_next, y_next, r_next]
    
    # иммитируем шум датчиков
    noise = np.random.multivariate_normal([0, 0, 0], R)
    x_noise[:, k+1] = x_real[:, k+1] + noise

# расширенный фильтр Калмана
H = np.eye(3) # матрица наблюдения
I = np.eye(3) # единичная матрица

for k in range(steps - 1):
    # текущие скорости
    s_t = (s_R_arr[k] + s_L_arr[k]) / 2.0 # передвижение
    s_r = (s_R_arr[k] - s_L_arr[k]) / (2.0 * B) # вращение
    
    # значения с предыдущего шага
    x_prev_ekf = x_ekf[0, k]
    y_prev_ekf = x_ekf[1, k]
    r_prev_ekf = x_ekf[2, k]
    
    # предсказание по уравнению дискретного дважинеия
    x_pred = x_prev_ekf + T * s_t * np.cos(r_prev_ekf) - 0.5 * T**2 * s_t * s_r * np.sin(r_prev_ekf)
    y_pred = y_prev_ekf + T * s_t * np.sin(r_prev_ekf) + 0.5 * T**2 * s_t * s_r * np.cos(r_prev_ekf)
    r_pred = r_prev_ekf + T * s_r
    
    X_pred = np.array([x_pred, y_pred, r_pred])
    
    # линеаризация модели движения
    F13 = -T * s_t * np.sin(r_prev_ekf) - 0.5 * T**2 * s_t * s_r * np.cos(r_prev_ekf) # производные по r 
    F23 = T * s_t * np.cos(r_prev_ekf) - 0.5 * T**2 * s_t * s_r * np.sin(r_prev_ekf)
    
    F = np.array([
        [1, 0, F13],
        [0, 1, F23],
        [0, 0, 1]
    ])
    
    # предсказание ковариации
    P_pred = F @ P @ F.T + Q
    
    # шаг коррекции
    Z_k = x_noise[:, k+1]
    
    # получаем коэффициент Калмана
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # корректируем предсказание
    X_est_new = X_pred + K @ (Z_k - H @ X_pred) # разница между ожиданием и значением
    x_ekf[:, k+1] = X_est_new
    
    # обновляем матрицу ковариации
    P = (I - K @ H) @ P_pred

# отрисовываем пути
plt.figure(figsize=(10, 6))
plt.plot(x_real[0, :], x_real[1, :], label='Истинная траектория', color='green', linewidth=2)
plt.scatter(x_noise[0, :], x_noise[1, :], label='Измерения с шумом', color='red', s=10, alpha=0.5)
plt.plot(x_ekf[0, :], x_ekf[1, :], label='Расш. фильтр Калмана', color='blue', linewidth=2, linestyle='--')

plt.title('Траектории робота')
plt.xlabel('x coord')
plt.ylabel('y coord')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('ekf_result.png', dpi=300, bbox_inches='tight')
print("file ekf_result saved")