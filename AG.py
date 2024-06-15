import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tkinter as tk
from tkinter import ttk
import cv2
import os

# Definir la ecuación como una variable global
ecuacion = "x * math.cos(x)"

# Clase AlgoritmoGenetico
class AlgoritmoGenetico:
    def __init__(self, ecuacion):
        self.ecuacion = ecuacion

    # Función objetivo
    def aptitud(self, x):
        return eval(self.ecuacion)

    # Inicialización de la población
    def inicializar_poblacion(self, tamano_poblacion, num_bits):
        return [''.join(random.choice('01') for _ in range(num_bits)) for _ in range(tamano_poblacion)]

    # Decodificación de un individuo
    def convertir_binario(self, individuo, num_bits, A, B):
        valor_maximo = 2**num_bits - 1
        entero = int(individuo, 2)
        return A + (entero / valor_maximo) * (B - A)

    # Evaluación de la población
    def evaluar_poblacion(self, poblacion, num_bits, A, B):
        return [self.aptitud(self.convertir_binario(ind, num_bits, A, B)) for ind in poblacion]

    # Selección de parejas (Estrategia A1)
    def seleccionar_parejas(self, poblacion, n):
        parejas = []
        for i in range(len(poblacion)):
            m = random.randint(0, n)  # Genera un número aleatorio entre 0 y n
            companeros = random.sample([j for j in range(len(poblacion)) if j != i], m)
            # Genera m números aleatorios que hacen referencia a individuos distintos de sí mismo
            parejas.extend([(i, companero) for companero in companeros])
        return parejas

    # Cruza (Estrategia C1)
    def cruzar(self, pareja, poblacion, num_bits):
        p1, p2 = pareja
        punto = random.randint(1, num_bits - 1)
        hijo1 = poblacion[p1][:punto] + poblacion[p2][punto:]
        hijo2 = poblacion[p2][:punto] + poblacion[p1][punto:]
        return hijo1, hijo2

    # Mutación (Estrategia M2)
    def mutar(self, individuo, num_bits, prob_mutacion_individuo, prob_mutacion_gen):
        if random.random() < prob_mutacion_individuo:
            individuo = list(individuo)
            for i in range(num_bits):
                if random.random() < prob_mutacion_gen:
                    j = random.randint(0, num_bits - 1)
                    # Intercambio de posición de bits
                    individuo[i], individuo[j] = individuo[j], individuo[i]
            individuo = ''.join(individuo)
        return individuo

    # Poda (Estrategia P2)
    def podar_poblacion(self, poblacion, aptitudes, tamano_poblacion):
        mejor_individuo = poblacion[np.argmax(aptitudes)]
        indices = list(range(len(poblacion)))
        indices.remove(np.argmax(aptitudes))
        mantener = random.sample(indices, tamano_poblacion - 1)
        nueva_poblacion = [poblacion[i] for i in mantener]
        nueva_poblacion.append(mejor_individuo)
        return nueva_poblacion

    # Algoritmo Genético para maximizar la aptitud
    def maximizar_aptitud(self, A, B, delta_x, generaciones, tamano_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen):
        num_bits = math.ceil(math.log2((B - A) / delta_x + 1))

        poblacion = self.inicializar_poblacion(tamano_poblacion, num_bits)

        mejores_aptitudes = []
        peores_aptitudes = []
        aptitudes_promedio = []

        if not os.path.exists('gen_images'):
            os.makedirs('gen_images')

        for generacion in range(generaciones):
            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)

            mejor_aptitud = max(aptitudes)
            peor_aptitud = min(aptitudes)
            aptitud_promedio = sum(aptitudes) / len(aptitudes)

            mejores_aptitudes.append(mejor_aptitud)
            peores_aptitudes.append(peor_aptitud)
            aptitudes_promedio.append(aptitud_promedio)

            mejor_indice = np.argmax(aptitudes)
            peor_indice = np.argmin(aptitudes)
            mejor_individuo = poblacion[mejor_indice]
            peor_individuo = poblacion[peor_indice]
            mejor_valor = self.convertir_binario(mejor_individuo, num_bits, A, B)
            peor_valor = self.convertir_binario(peor_individuo, num_bits, A, B)

            ActualizadorTabla.actualizar_tabla(generacion, mejor_individuo, mejor_indice, mejor_valor, mejor_aptitud)

            n = random.randint(1, len(poblacion) - 1)
            prob_cruce = random.uniform(0.5, 1.0)

            parejas = self.seleccionar_parejas(poblacion, n)
            descendientes = []
            for pareja in parejas:
                if random.random() < prob_cruce:
                    desc1, desc2 = self.cruzar(pareja, poblacion, num_bits)
                    descendientes.append(self.mutar(desc1, num_bits, prob_mutacion_individuo, prob_mutacion_gen))
                    descendientes.append(self.mutar(desc2, num_bits, prob_mutacion_individuo, prob_mutacion_gen))

            poblacion.extend(descendientes)

            if len(poblacion) > max_poblacion:
                raise ValueError("El tamaño de la población no puede exceder el tamaño máximo especificado.")

            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)
            poblacion = self.podar_poblacion(poblacion, aptitudes, min(tamano_poblacion, max_poblacion))

            GraficoEvolucion.generar_grafico(self.aptitud, poblacion, mejor_valor, peor_valor, generacion, 'Maximización', A, B, self.ecuacion)

        VideoCreador.crear_video('gen_images', 'evolucion_maximizacion.mp4')
        GraficoEvolucion.generar_grafico_fitness(mejores_aptitudes, peores_aptitudes, aptitudes_promedio)

    # Algoritmo Genético para minimizar la aptitud
    def minimizar_aptitud(self, A, B, delta_x, generaciones, tamano_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen):
        num_bits = math.ceil(math.log2((B - A) / delta_x + 1))

        poblacion = self.inicializar_poblacion(tamano_poblacion, num_bits)

        mejores_aptitudes = []
        peores_aptitudes = []
        aptitudes_promedio = []

        if not os.path.exists('gen_images'):
            os.makedirs('gen_images')

        for generacion in range(generaciones):
            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)

            mejor_aptitud = min(aptitudes)
            peor_aptitud = max(aptitudes)
            aptitud_promedio = sum(aptitudes) / len(aptitudes)

            mejores_aptitudes.append(mejor_aptitud)
            peores_aptitudes.append(peor_aptitud)
            aptitudes_promedio.append(aptitud_promedio)

            mejor_indice = np.argmin(aptitudes)
            peor_indice = np.argmax(aptitudes)
            mejor_individuo = poblacion[mejor_indice]
            peor_individuo = poblacion[peor_indice]
            mejor_valor = self.convertir_binario(mejor_individuo, num_bits, A, B)
            peor_valor = self.convertir_binario(peor_individuo, num_bits, A, B)

            ActualizadorTabla.actualizar_tabla(generacion, mejor_individuo, mejor_indice, mejor_valor, mejor_aptitud)

            n = random.randint(1, len(poblacion) - 1)
            prob_cruce = random.uniform(0.5, 1.0)

            parejas = self.seleccionar_parejas(poblacion, n)
            descendientes = []
            for pareja in parejas:
                if random.random() < prob_cruce:
                    desc1, desc2 = self.cruzar(pareja, poblacion, num_bits)
                    descendientes.append(self.mutar(desc1, num_bits, prob_mutacion_individuo, prob_mutacion_gen))
                    descendientes.append(self.mutar(desc2, num_bits, prob_mutacion_individuo, prob_mutacion_gen))

            poblacion.extend(descendientes)

            if len(poblacion) > max_poblacion:
                raise ValueError("El tamaño de la población no puede exceder el tamaño máximo especificado.")

            aptitudes = self.evaluar_poblacion(poblacion, num_bits, A, B)
            poblacion = self.podar_poblacion(poblacion, aptitudes, min(tamano_poblacion, max_poblacion))

            GraficoEvolucion.generar_grafico(self.aptitud, poblacion, mejor_valor, peor_valor, generacion, 'Minimización', A, B, self.ecuacion)

        VideoCreador.crear_video('gen_images', 'evolucion_minimizacion.mp4')
        GraficoEvolucion.generar_grafico_fitness(mejores_aptitudes, peores_aptitudes, aptitudes_promedio)

# Clase para actualizar la tabla
class ActualizadorTabla:
    @staticmethod
    def actualizar_tabla(generacion, individuo, indice, valor, aptitud):
        tabla.insert("", "end", values=(generacion, individuo, indice, valor, aptitud))

# Clase para generar gráficos de evolución
class GraficoEvolucion:
    @staticmethod
    def generar_grafico(funcion_aptitud, poblacion, mejor_valor, peor_valor, generacion, tipo, A, B, ecuacion, color_funcion = 'orange'):
        plt.figure(figsize=(10, 6))
        x = np.linspace(A, B, 1000)
        y = [funcion_aptitud(val) for val in x]
        plt.plot(x, y, label=f'Función: {ecuacion}', color = color_funcion)

        individuos_x = [AlgoritmoGenetico.convertir_binario(None, ind, int(math.log2((B - A) / 0.01 + 1)), A, B) for ind in poblacion]
        individuos_y = [funcion_aptitud(val) for val in individuos_x]
        plt.scatter(individuos_x, individuos_y, color='blue', label='Individuos')

        plt.scatter(mejor_valor, funcion_aptitud(mejor_valor), color='green', label='Mejor individuo', zorder=5)
        plt.scatter(peor_valor, funcion_aptitud(peor_valor), color='red', label='Peor individuo', zorder=5)

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Función y individuos - Generación {generacion} ({tipo})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'gen_images/gen_{generacion}.png')
        plt.close()

    @staticmethod
    def generar_grafico_fitness(mejores_aptitudes, peores_aptitudes, aptitudes_promedio):
        plt.figure(figsize=(10, 6))
        generaciones = list(range(len(mejores_aptitudes)))
        plt.plot(generaciones, mejores_aptitudes, label='Mejor Fitness', color='green')
        plt.plot(generaciones, peores_aptitudes, label='Peor Fitness', color='red')
        plt.plot(generaciones, aptitudes_promedio, label='Fitness Promedio', color='blue')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Evolución del Fitness a lo largo de las Generaciones')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('fitness_evolucion.png')
        plt.show()

# Clase para crear videos
class VideoCreador:
    @staticmethod
    def crear_video(carpeta_imagenes, video_salida):
        imagenes = [img for img in os.listdir(carpeta_imagenes) if img.endswith(".png")]
        imagenes.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        frame = cv2.imread(os.path.join(carpeta_imagenes, imagenes[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_salida, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

        for imagen in imagenes:
            video.write(cv2.imread(os.path.join(carpeta_imagenes, imagen)))

        cv2.destroyAllWindows()
        video.release()

# Interfaz gráfica
def iniciar_gui():
    global tabla

    def validar_entrada_valor(valor):
        try:
            return float(valor)
        except ValueError:
            return None

    def validar_entrada_entero(valor):
        try:
            entero = int(valor)
            return entero if entero > 0 else None
        except ValueError:
            return None

    def mostrar_advertencia(mensaje):
        advertencia = tk.Toplevel()
        advertencia.title("Advertencia")
        tk.Label(advertencia, text=mensaje, padx=20, pady=20).pack()
        tk.Button(advertencia, text="OK", command=advertencia.destroy, padx=10, pady=5).pack()

    def iniciar_algoritmo():
        errores = []
        valor_inicial = validar_entrada_valor(entrada_valor_inicial.get())
        valor_final = validar_entrada_valor(entrada_valor_final.get())
        delta_x = validar_entrada_valor(entrada_delta_x.get())
        generaciones = validar_entrada_entero(entrada_generaciones.get())
        tamano_poblacion = validar_entrada_entero(entrada_tamano_poblacion.get())
        max_poblacion = validar_entrada_entero(entrada_max_poblacion.get())
        prob_mutacion_individuo = validar_entrada_valor(entrada_prob_mutacion_individuo.get())
        prob_mutacion_gen = validar_entrada_valor(entrada_prob_mutacion_gen.get())

        if valor_inicial is None:
            errores.append("Valor inicial no válido. Por favor, ingrese un número.")
        if valor_final is None:
            errores.append("Valor final no válido. Por favor, ingrese un número.")
        if valor_inicial is not None and valor_final is not None and valor_inicial >= valor_final:
            errores.append("El valor inicial debe ser menor que el valor final.")
        if delta_x is None:
            errores.append("Valor de delta x no válido. Por favor, ingrese un número.")
        elif not (0 < delta_x <= 1):
            errores.append("Valor de delta x debe estar entre 0 y 1.")
        if generaciones is None:
            errores.append("Número de generaciones no válido. Por favor, ingrese un número entero positivo.")
        if tamano_poblacion is None:
            errores.append("Tamaño de la población no válido. Por favor, ingrese un número entero positivo.")
        if max_poblacion is None:
            errores.append("Tamaño máximo de la población no válido. Por favor, ingrese un número entero positivo.")
        if tamano_poblacion is not None and max_poblacion is not None and tamano_poblacion >= max_poblacion:
            errores.append("El tamaño de la población debe ser menor que el tamaño máximo de la población.")
        if prob_mutacion_individuo is None:
            errores.append("Probabilidad de mutación del individuo no válida. Por favor, ingrese un número.")
        elif not (0 <= prob_mutacion_individuo <= 1):
            errores.append("Probabilidad de mutación del individuo debe estar entre 0 y 1.")
        if prob_mutacion_gen is None:
            errores.append("Probabilidad de mutación del gen no válida. Por favor, ingrese un número.")
        elif not (0 <= prob_mutacion_gen <= 1):
            errores.append("Probabilidad de mutación del gen debe estar entre 0 y 1.")

        if errores:
            mostrar_advertencia("\n".join(errores))
            return

        try:
            A = valor_inicial
            B = valor_final
            delta_x = float(entrada_delta_x.get())
            generaciones = int(entrada_generaciones.get())
            maximizar = entrada_maximizar.get().lower() == 's'
            tamano_poblacion = int(entrada_tamano_poblacion.get())
            max_poblacion = int(entrada_max_poblacion.get())
            prob_mutacion_individuo = float(entrada_prob_mutacion_individuo.get())
            prob_mutacion_gen = float(entrada_prob_mutacion_gen.get())

            tabla.delete(*tabla.get_children())

            ag = AlgoritmoGenetico(ecuacion)
            if maximizar:
                ag.maximizar_aptitud(A, B, delta_x, generaciones, tamano_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen)
            else:
                ag.minimizar_aptitud(A, B, delta_x, generaciones, tamano_poblacion, max_poblacion, prob_mutacion_individuo, prob_mutacion_gen)

        except ValueError as e:
            mostrar_advertencia(f"Error: {str(e)}")

    root = tk.Tk()
    root.title("Algoritmo Genético")

    tk.Label(root, text="Valor Inicial:").grid(row=0, column=0)
    entrada_valor_inicial = tk.Entry(root)
    entrada_valor_inicial.grid(row=0, column=1)

    tk.Label(root, text="Valor Final:").grid(row=1, column=0)
    entrada_valor_final = tk.Entry(root)
    entrada_valor_final.grid(row=1, column=1)

    tk.Label(root, text="Valor de delta x:").grid(row=2, column=0)
    entrada_delta_x = tk.Entry(root)
    entrada_delta_x.grid(row=2, column=1)

    tk.Label(root, text="Maximizar? (s/n):").grid(row=3, column=0)
    entrada_maximizar = tk.Entry(root)
    entrada_maximizar.grid(row=3, column=1)

    tk.Label(root, text="Número de generaciones:").grid(row=4, column=0)
    entrada_generaciones = tk.Entry(root)
    entrada_generaciones.grid(row=4, column=1)

    tk.Label(root, text="Probabilidad de mutación del individuo:").grid(row=5, column=0)
    entrada_prob_mutacion_individuo = tk.Entry(root)
    entrada_prob_mutacion_individuo.grid(row=5, column=1)

    tk.Label(root, text="Probabilidad de mutación del gen:").grid(row=6, column=0)
    entrada_prob_mutacion_gen = tk.Entry(root)
    entrada_prob_mutacion_gen.grid(row=6, column=1)

    tk.Label(root, text="Tamaño de la población:").grid(row=7, column=0)
    entrada_tamano_poblacion = tk.Entry(root)
    entrada_tamano_poblacion.grid(row=7, column=1)

    tk.Label(root, text="Tamaño máximo de la población:").grid(row=8, column=0)
    entrada_max_poblacion = tk.Entry(root)
    entrada_max_poblacion.grid(row=8, column=1)

    tk.Button(root, text="Ejecutar Algoritmo", command=iniciar_algoritmo).grid(row=9, column=0, columnspan=2)

    columnas = ("Generación", "Individuo", "Índice", "Valor", "Aptitud")
    tabla = ttk.Treeview(root, columns=columnas, show="headings")
    for col in columnas:
        tabla.heading(col, text=col)
    tabla.grid(row=10, column=0, columnspan=2)

    root.mainloop()

# Ejecutar la interfaz gráfica
if __name__ == "__main__":
    iniciar_gui()
