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
        # self.puntaje_total = 0
        # self.puntaje_turno = 0 
        self.estado = EstadoDiezMil()
        self.dados_restantes = [1, 2, 3, 4, 5, 6]
        self.turno = 1
        self.tope_turnos = 1000
        self.flag = False

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio.
        """
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
                self.estado.fin_turno()
                self.turno += 1
            else:
                self.estado.actualizar_estado(puntaje, len(self.dados_restantes))
        
        if accion == JUGADA_PLANTARSE: 
            self.dados_restantes = [1,2,3,4,5,6]
            self.estado.fin_turno()
            self.turno += 1

        if self.estado.puntaje_total >= 10000 or self.turno >= self.tope_turnos:
            self.reset()
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

    def actualizar_estado(self, puntaje, cant_dados_restantes, *args, **kwargs) -> None:
        """Modifica las variables internas del estado luego de una tirada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """
        self.puntaje_turno += puntaje
        self.cant_dados_restantes = cant_dados_restantes
        self.puntaje_total += puntaje

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
        return(f"Estado actual -> Puntaje Turno: {self.puntaje_turno}, Dados Restantes: {self.cant_dados_restantes}, Puntaje Total: {self.puntaje_total}")  

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
        self.Q = defaultdict(lambda: {JUGADA_PLANTARSE: 0.0, JUGADA_TIRAR: 0.0}) # Q[(puntaje_turno, dados_restantes, puntaje_total)][accion]
        self.politica_optima = {}

    def elegir_accion(self):
        """Selecciona una acción de acuerdo a una política ε-greedy.
        """    
        # Obtenemos las acciones y sus valores Q
        acciones = list(self.Q[self.estado].keys())
        valores = list(self.Q[self.estado].values())
        
        if max(valores) == min(valores): 
            # Si los valores en Q de ambas acciones en ese estado son iguales, entonces elige una acción de forma aleatoria
            return random.choice(acciones) 

        else:
            if random.random() > self.epsilon:
                return acciones[max(valores)]
            else:
                return acciones[min(valores)]


    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        for episodio in tqdm(range(episodios)):
            while episodio < episodios and not self.ambiente.flag:
                #estado_anterior = self.estado
                estado_anterior = (self.estado.puntaje_turno, self.estado.cant_dados_restantes, self.estado.puntaje_total)
                accion = self.elegir_accion()
                recompensa, _ = self.ambiente.step(accion)
                estado_actual = (self.estado.puntaje_turno, self.estado.cant_dados_restantes, self.estado.puntaje_total)
                self.Q[estado_anterior][accion] += self.alpha * (recompensa + self.gamma * np.max(list(self.Q[estado_actual].values())) - self.Q[estado_anterior][accion])
                print(f"Estado: {estado_anterior}, Acción: {accion}, Valor Q actualizado de {self.Q[estado_anterior][accion]} a {self.Q[estado_anterior][accion]}")
            
            self.ambiente.reset()

        self.politica_optima = {estado: np.argmax(self.Q[estado]) for estado in self.Q}

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
                accion_optima = JUGADA_TIRAR if row[1] == "JUGADA_TIRAR" else JUGADA_PLANTARSE
                politica[estado] = accion_optima
                
        return politica
    
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
        
        estado = (puntaje_turno, len(no_usados), puntaje_total)
        jugada = self.politica[estado]
       
        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada==JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)