import numpy as np
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR
from collections import defaultdict
from tqdm import tqdm
from jugador import Jugador
import random
import csv

class AmbienteDiezMil:
    
    def __init__(self):
        """Definir las variables de instancia de un ambiente.
        ¿Qué es propio de un ambiente de 10.000?
        """
        self.estado = EstadoDiezMil()
        self.dados_restantes = [1, 2, 3, 4, 5, 6]
        self.turno = 1
        self.tope_turnos = 1000
        self.flag = False

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio.
        """
        self.flag = False
        self.dados_restantes = [1, 2, 3, 4, 5, 6]
        self.estado.cant_dados_restantes = len(self.dados_restantes)
        self.estado.puntaje_turno = 0
        self.estado.puntaje_total = 0
        self.turno = 1

    def step(self, accion):
        """Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el turno. 
        """
        puntaje = 0
        
        if accion == JUGADA_TIRAR:
            dados: list[int] = [random.randint(1, 6) for _ in range(len(self.dados_restantes))]
            puntaje, self.dados_restantes = puntaje_y_no_usados(dados)

            if puntaje == 0:  # No suma nada. Pierde el turno.
                self.dados_restantes = [1, 2, 3, 4, 5, 6]
                self.estado.fin_turno()
                self.turno += 1
            else:
                self.estado.actualizar_estado(puntaje, len(self.dados_restantes))
        
        if accion == JUGADA_PLANTARSE: 
            self.dados_restantes = [1, 2, 3, 4, 5, 6]
            self.estado.fin_turno()
            self.turno += 1

        if self.estado.puntaje_total >= 10000 or self.turno >= self.tope_turnos:
            self.flag = True

        return (puntaje, self.flag)        

class EstadoDiezMil:
    def __init__(self):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.
        """
        self.puntaje_turno = 0
        self.cant_dados_restantes = 6
        self.puntaje_total = 0

        self.puntaje_turno_discretizado = 0
        self.puntaje_total_discretizado = 0

        # Definimos rangos para puntaje_turno y puntaje_total
        self.bins_puntaje_turno = [0, 150, 300, 450, 650, 1000, 10000] 
        self.bins_puntaje_total = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    def actualizar_estado(self, puntaje, cant_dados_restantes, *args, **kwargs) -> None:
        """Modifica las variables internas del estado luego de una tirada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """
        self.puntaje_turno += puntaje
        self.cant_dados_restantes = cant_dados_restantes
        self.puntaje_total += puntaje
    
    def asignar_a_bin(self, puntaje, bins):
        """
        Asigna un puntaje a un bin basado en los límites proporcionados.

        Args:
            puntaje (int): Puntaje a asignar.
            bins (list[int]): Límites de los bins.

        Returns:
            int: Índice del bin correspondiente.
        """
        return np.digitize(puntaje, bins) - 1  # Restar 1 para usar índices basados en 0
    
    def obtener_estado_discretizado(self):
        # Asignar los valores a los bins correspondientes
        self.puntaje_turno_discretizado = self.bins_puntaje_turno[self.asignar_a_bin(self.puntaje_turno, self.bins_puntaje_turno)]
        self.puntaje_total_discretizado = self.bins_puntaje_total[self.asignar_a_bin(self.puntaje_total, self.bins_puntaje_total)]

        return (self.puntaje_turno_discretizado, self.cant_dados_restantes, self.puntaje_total_discretizado)

    def fin_turno(self):
        """Modifica el estado al terminar el turno.
        """
        self.puntaje_turno = 0
        self.cant_dados_restantes = 6

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """
        return (f"Estado actual -> Puntaje Turno: {self.puntaje_turno}, Dados Restantes: {self.cant_dados_restantes}, Puntaje Total: {self.puntaje_total}")  

class AgenteQLearning:
    def __init__(
        self,
        ambiente: AmbienteDiezMil,
        alpha: float,
        gamma: float,
        epsilon: float,
        *args,
        **kwargs
    ):
        """Definir las variables internas de un Agente que implementa el algoritmo de Q-Learning.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
            alpha (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad de explorar.
        """
        self.ambiente = ambiente
        self.estado = self.ambiente.estado
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon    
        
        self.Q = {} # Q[(puntaje_turno, dados_restantes, puntaje_total)][accion]

        # Inicializamos en 0 todas las combinaciones de estados discretizados y acciones
        for bin_turnos in self.estado.bins_puntaje_turno:
            for bin_total in self.estado.bins_puntaje_total:
                for k in range(7):  # Considera 0 a 6 dados restantes (0 a 6 valores posibles)
                    estado = (bin_turnos, k, bin_total)
                    self.Q[estado] = {accion: 0 for accion in JUGADAS_STR.keys()}

        self.politica_optima = {}

    def elegir_accion(self, estado):
        """Selecciona una acción de acuerdo a una política ε-greedy.
        """    
        # Obtenemos las acciones y sus valores Q
        acciones = list(self.Q[estado].keys())
        valores = list(self.Q[estado].values())

        if max(valores) == min(valores): 
            # Si los valores en Q de ambas acciones en ese estado son iguales, entonces elige una acción de forma aleatoria
            return random.choice(acciones) 

        else:
            if random.random() > self.epsilon:
                return acciones[np.argmax(valores)]
            else:
                return acciones[np.argmin(valores)]


    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        for i in tqdm(range(episodios)):
            while self.ambiente.turno < self.ambiente.tope_turnos and not self.ambiente.flag:
                estado_anterior = self.estado.obtener_estado_discretizado()
                accion = self.elegir_accion(estado_anterior)
                recompensa, _ = self.ambiente.step(accion)
                estado_actual = self.estado.obtener_estado_discretizado()
                #valor_anterior = self.Q[estado_anterior][accion]
                self.Q[estado_anterior][accion] += self.alpha * (recompensa + self.gamma * np.max(list(self.Q[estado_actual].values())) - self.Q[estado_anterior][accion])
                
                #print(f"Estado: {estado_anterior}, Acción: {accion}, Valor Q actualizado de {valor_anterior} a {self.Q[estado_anterior][accion]}")
            
            self.ambiente.reset()

        self.politica_optima = {estado: max(self.Q[estado], key=self.Q[estado].get) for estado in self.Q}

    def guardar_politica(self, filename: str):
        """Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["estado", "accion_optima"])
            # Guardamos la acción óptima para cada estado
            for estado, accion_optima in self.politica_optima.items():
                writer.writerow([estado, accion_optima])
        

class JugadorEntrenado(Jugador):
    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica)

        # Definimos rangos para puntaje_turno y puntaje_total
        self.bins_puntaje_turno = [0, 150, 300, 450, 650, 1000, 10000] 
        self.bins_puntaje_total = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        
    def _leer_politica(self, filename: str, SEP: str=','):
        """Carga una politica entrenada con un agente de RL, que está guardada
        en el archivo filename en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo que contiene a una política almacenada. 
        """
        politica = {}
    
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter=SEP)
            next(reader)  # Omitimos el header
            
            for row in reader:
                estado = eval(row[0])  # Convertimos el estado de nuevo a una tupla
                accion_optima = JUGADA_TIRAR if row[1] == JUGADA_TIRAR else JUGADA_PLANTARSE
                politica[estado] = accion_optima
                
        return politica
    
    def asignar_a_bin(self, puntaje, bins):
        """
        Asigna un puntaje a un bin basado en los límites proporcionados.

        Args:
            puntaje (int): Puntaje a asignar.
            bins (list[int]): Límites de los bins.

        Returns:
            int: Índice del bin correspondiente.
        """
        return np.digitize(puntaje, bins) - 1  # Restar 1 para usar índices basados en 0
    
    def jugar(
        self,
        puntaje_total:int,
        puntaje_turno:int,
        dados:list[int],
    ) -> tuple[int,list[int]]:
        """Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        """
        puntaje, no_usados = puntaje_y_no_usados(dados)

        # Convertir el estado actual a su representación discretizada
        puntaje_turno_discretizado = self.bins_puntaje_turno[self.asignar_a_bin(puntaje_turno, self.bins_puntaje_turno)]
        puntaje_total_discretizado = self.bins_puntaje_total[self.asignar_a_bin(puntaje_total, self.bins_puntaje_total)]
        
        estado_discretizado = (puntaje_turno_discretizado, len(no_usados), puntaje_total_discretizado)
        
        estado = (puntaje_turno, len(no_usados), puntaje_total)
        jugada = self.politica[estado_discretizado]
       
        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)