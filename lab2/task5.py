import numpy as np
from scipy.io import wavfile

# считываем файл, получаем частоту дискретизации sr и массив амплитуд data
sr, data = wavfile.read("test5.wav")

# запоминаем длину массива
original_length = len(data)

# нормализуем значения в числа с плавающей запятой в диапазоне [-1:1]
max_int16 = 2**15
data_float = data.astype(np.float64) / max_int16

# прямое преобразование Фурье
# используем rfft для получения спектра положительных частот
spectrum = np.fft.rfft(data_float)

# разделяем спектра на 4 части
# определяем размер одной четверти спектра
q = len(spectrum) // 4

# искаженный спектр имеет порядок [C, B, D, A]
# получаем соответствующие куски
part_C = spectrum[0 : q]          # C
part_B = spectrum[q : 2*q]        # B
part_D = spectrum[2*q : 3*q]      # D
part_A = spectrum[3*q :]          # A

# восстановление исходного порядка [A, B, C, D]
restored_spectrum = np.concatenate((part_A, part_B, part_C, part_D))

# обратное преобразование Фурье
# irfft переводит частоты обратно в звуковые амплитуды
# задаём ту же длину, что и изначальная длина
restored_data_float = np.fft.irfft(restored_spectrum, n=original_length)

# сохраняем результат
# возвращаем амплитуды обратно в формат 16-битных целых чисел
restored_data_int = np.int16(restored_data_float * (max_int16-1))

# сохраняем очищенный звук в новый файл
wavfile.write("test5_restored.wav", sr, restored_data_int)

print("восстановление:")
print(f"частота дискретизации: {sr} Гц")
print(f"длина аудио: {original_length / sr:.2f} секунд")
print("результат восстановления в файле test5_restored.wav")