import numpy as np  # Biblioteca  para cálculos numéricos
import matplotlib.pyplot as plt  # Biblioteca para visualización de gráficos
import librosa  # Biblioteca para análisis de audio
import librosa.display  # Módulo display de biblioteca librosa para visualización de audio
import soundfile as sf  # Biblioteca soundfile para leer y escribir archivos de audio

# Paso 1: Cargar el archivo de audio original
audio_original = "cancion.wav"

# Cargar el archivo de audio y obtener la tasa de muestreo
audio, tasa_muestreo = librosa.load(audio_original, sr=None)

# Paso 2: Calcular la transformada de Fourier de la señal de audio
transformada_fourier = np.fft.fft(audio)

# Paso 3: Calcular el espectro de amplitud
espectro_amplitud = np.abs(transformada_fourier)

# Paso 4: Establecer un umbral para eliminar componentes de baja amplitud
umbral = np.percentile(espectro_amplitud, 96.5)  # Mantener el 1% de los componentes más altos

# Paso 5: Filtrar los componentes de baja amplitud
transformada_filtrada = transformada_fourier * (espectro_amplitud > umbral)

# Paso 6: Reconstruir la señal de audio filtrada
audio_reconstruido = np.fft.ifft(transformada_filtrada).real

# Paso 7: Guardar el archivo de audio reconstruido en formato WAV
sf.write("reconstruido.mp3", audio_reconstruido, tasa_muestreo, format='mp3')

# Crear una nueva figura para unir las gráficas
plt.figure(figsize=(12, 8))

# Graficar la forma de onda original
plt.subplot(3, 1, 1)
librosa.display.waveshow(audio, sr=tasa_muestreo, color='b')
plt.title('Forma de onda original')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

# Graficar la forma de onda reconstruida
plt.subplot(3, 1, 2)
librosa.display.waveshow(audio_reconstruido, sr=tasa_muestreo, color='r')
plt.title('Forma de onda reconstruida (compresión de audio)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

# Graficar ambas formas de onda superpuestas
plt.subplot(3, 1, 3)
plt.plot(audio, color='b', label='Original')
plt.plot(audio_reconstruido, color='r', label='Reconstruida')
plt.title('Superposición de formas de onda')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()

plt.tight_layout()  # Ajusta automáticamente el espaciado entre los subplots
plt.show()  # Muestra la figura