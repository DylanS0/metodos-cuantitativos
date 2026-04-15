"""
Simulacion de Logistica con SimPy - Programacion Orientada a Objetos
Materia: Metodos Cuantitativos
Descripcion: Simula el recorrido de camiones que realizan entregas a clientes,
             calcula distancias, tiempos y genera conclusiones usando IA (Gemini).
"""

import sys
import io

# Forzar UTF-8 en la consola de Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import simpy
import math
import os
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def calcular_distancia(p1: tuple, p2: tuple) -> float:
    """Calcula la distancia euclidiana entre dos puntos (x, y)."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Clases del dominio
# ─────────────────────────────────────────────────────────────────────────────

class Cliente:
    """Representa un cliente con ubicación y demanda de productos."""

    def __init__(self, nombre: str, x: float, y: float, demanda: int):
        self.nombre = nombre
        self.ubicacion = (x, y)
        self.demanda = demanda

    def __repr__(self):
        return f"Cliente('{self.nombre}', ubicacion={self.ubicacion}, demanda={self.demanda})"


class Deposito:
    """Punto de origen y retorno de todos los camiones."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.ubicacion = (x, y)
        self.nombre = "Depósito"

    def __repr__(self):
        return f"Deposito(ubicacion={self.ubicacion})"


class Camion:
    """
    Camión que realiza entregas siguiendo una ruta definida.

    Atributos
    ---------
    nombre        : identificador del camión
    capacidad     : unidades máximas que puede transportar
    velocidad     : unidades de distancia por unidad de tiempo
    tiempo_entrega: tiempo fijo de descarga por cliente
    """

    VELOCIDAD_DEFAULT = 1.0        # km / min  (ajustar según escala)
    TIEMPO_ENTREGA_DEFAULT = 5.0   # minutos por parada

    def __init__(
        self,
        nombre: str,
        capacidad: int,
        deposito: Deposito,
        ruta: list,
        velocidad: float = VELOCIDAD_DEFAULT,
        tiempo_entrega: float = TIEMPO_ENTREGA_DEFAULT,
    ):
        self.nombre = nombre
        self.capacidad = capacidad
        self.deposito = deposito
        self.ruta = ruta            # lista de objetos Cliente
        self.velocidad = velocidad
        self.tiempo_entrega = tiempo_entrega

        # Estado en tiempo de ejecución
        self.carga_actual = sum(c.demanda for c in ruta)
        self.posicion_actual = deposito.ubicacion
        self.distancia_total = 0.0
        self.registro: list[dict] = []

    # ------------------------------------------------------------------
    # Proceso SimPy
    # ------------------------------------------------------------------

    def ejecutar(self, env: simpy.Environment):
        """Proceso principal del camión dentro del entorno SimPy."""
        print(f"\n{'='*60}")
        print(f"  {self.nombre} | Capacidad: {self.capacidad} u. | Carga: {self.carga_actual} u.")
        print(f"  Ruta: {' -> '.join(c.nombre for c in self.ruta)} -> {self.deposito.nombre}")
        print(f"{'='*60}\n")

        # Verificar si la carga supera la capacidad
        if self.carga_actual > self.capacidad:
            print(f"[ADVERTENCIA] La carga ({self.carga_actual} u.) supera la capacidad "
                  f"({self.capacidad} u.). Se transportará lo que sea posible.\n")

        # ── Visitar cada cliente ──────────────────────────────────────
        for cliente in self.ruta:
            destino = cliente.ubicacion

            # Tiempo de viaje
            distancia = calcular_distancia(self.posicion_actual, destino)
            tiempo_viaje = distancia / self.velocidad
            self.distancia_total += distancia

            yield env.timeout(tiempo_viaje)

            # Llegada
            print(f"[T={env.now:7.2f} min] {self.nombre} llegó a {cliente.nombre} "
                  f"(dist. tramo: {distancia:.2f} km)")

            # Entrega
            entregado = min(cliente.demanda, self.carga_actual)
            self.carga_actual -= entregado
            yield env.timeout(self.tiempo_entrega)

            evento = {
                "tiempo": round(env.now, 2),
                "camion": self.nombre,
                "cliente": cliente.nombre,
                "ubicacion": cliente.ubicacion,
                "demanda": cliente.demanda,
                "entregado": entregado,
                "carga_restante": self.carga_actual,
            }
            self.registro.append(evento)

            print(f"[T={env.now:7.2f} min]   ↳ Entregado: {entregado} u. | "
                  f"Carga restante: {self.carga_actual} u.")

            self.posicion_actual = destino

        # ── Regresar al depósito ────────────────────────────────────
        distancia_regreso = calcular_distancia(self.posicion_actual, self.deposito.ubicacion)
        tiempo_regreso = distancia_regreso / self.velocidad
        self.distancia_total += distancia_regreso

        yield env.timeout(tiempo_regreso)

        print(f"\n[T={env.now:7.2f} min] {self.nombre} regresó al {self.deposito.nombre}. "
              f"Distancia regreso: {distancia_regreso:.2f} km")
        print(f"{'─'*60}")
        print(f"  Distancia total recorrida: {self.distancia_total:.2f} km")
        print(f"  Tiempo total de operación: {env.now:.2f} min")
        print(f"{'─'*60}\n")

    def __repr__(self):
        return f"Camion('{self.nombre}', capacidad={self.capacidad})"


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal de simulación
# ─────────────────────────────────────────────────────────────────────────────

class Simulacion:
    """
    Orquesta el entorno SimPy, lanza los procesos de los camiones
    y recopila los resultados finales.
    """

    def __init__(self, deposito: Deposito, camiones: list[Camion]):
        self.deposito = deposito
        self.camiones = camiones
        self.env = simpy.Environment()
        self.tiempo_total = 0.0

    def iniciar(self):
        """Registra todos los procesos de camiones y corre la simulación."""
        for camion in self.camiones:
            self.env.process(camion.ejecutar(self.env))
        self.env.run()
        self.tiempo_total = self.env.now

    def resumen(self) -> dict:
        """Genera y muestra el resumen global de la simulación."""
        distancia_global = sum(c.distancia_total for c in self.camiones)
        todos_registros = []
        for c in self.camiones:
            todos_registros.extend(c.registro)

        print("\n" + "=" * 60)
        print("  RESUMEN FINAL DE LA SIMULACIÓN")
        print("=" * 60)
        print(f"  Distancia total (todos los camiones): {distancia_global:.2f} km")
        print(f"  Tiempo total de simulación          : {self.tiempo_total:.2f} min")
        print(f"  Entregas realizadas                 : {len(todos_registros)}")
        print("\n  Registro de entregas:")
        print(f"  {'Camión':<12} {'Cliente':<15} {'Demanda':>8} {'Entregado':>10} {'Tiempo':>8}")
        print(f"  {'─'*55}")
        for r in todos_registros:
            print(f"  {r['camion']:<12} {r['cliente']:<15} {r['demanda']:>8} "
                  f"{r['entregado']:>10} {r['tiempo']:>7.1f} min")
        print("=" * 60 + "\n")

        return {
            "distancia_total_km": round(distancia_global, 2),
            "tiempo_total_min": round(self.tiempo_total, 2),
            "total_entregas": len(todos_registros),
            "registros": todos_registros,
            "camiones": [
                {
                    "nombre": c.nombre,
                    "capacidad": c.capacidad,
                    "distancia_km": round(c.distancia_total, 2),
                    "clientes_visitados": len(c.registro),
                }
                for c in self.camiones
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Integración con Gemini AI
# ─────────────────────────────────────────────────────────────────────────────

class AnalistaIA:
    """Envia los resultados de la simulacion a Gemini y retorna conclusiones."""

    MODELO = "gemini-2.0-flash"

    def __init__(self):
        if not GENAI_AVAILABLE:
            raise EnvironmentError("Libreria google-genai no disponible. Instala: pip install google-genai")
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Variable de entorno GEMINI_API_KEY no encontrada. "
                "Crea un archivo .env con GEMINI_API_KEY=<tu_clave>."
            )
        self.cliente = genai.Client(api_key=api_key)

    def analizar(self, resumen: dict) -> str:
        """Genera una conclusion y recomendacion basada en el resumen."""
        prompt = self._construir_prompt(resumen)
        print(">>> Consultando a Gemini AI para analisis...\n")
        respuesta = self.cliente.models.generate_content(
            model=self.MODELO,
            contents=prompt,
        )
        return respuesta.text

    @staticmethod
    def _construir_prompt(resumen: dict) -> str:
        lineas_registros = "\n".join(
            f"  - Camión {r['camion']}: entregó {r['entregado']} u. al cliente "
            f"{r['cliente']} (demanda {r['demanda']} u.) a los {r['tiempo']} min"
            for r in resumen["registros"]
        )
        lineas_camiones = "\n".join(
            f"  - {c['nombre']}: {c['distancia_km']} km, {c['clientes_visitados']} clientes"
            for c in resumen["camiones"]
        )

        return f"""
Eres un experto en logística y optimización de rutas de distribución.
Analiza los siguientes resultados de una simulación de entregas con camiones:

RESUMEN GLOBAL:
- Distancia total recorrida: {resumen['distancia_total_km']} km
- Tiempo total de simulación: {resumen['tiempo_total_min']} minutos
- Total de entregas realizadas: {resumen['total_entregas']}

DESEMPEÑO POR CAMIÓN:
{lineas_camiones}

DETALLE DE ENTREGAS:
{lineas_registros}

Con base en estos datos, por favor proporciona:
1. Una CONCLUSIÓN objetiva sobre la eficiencia de la operación logística.
2. Al menos TRES RECOMENDACIONES concretas para mejorar la ruta, el tiempo o la distribución de carga.
3. Identifica si hubo algún problema de capacidad o ineficiencia notable.

Responde en español, de forma clara, profesional y estructurada.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Definir depósito ───────────────────────────────────────────
    deposito = Deposito(x=0, y=0)

    # ── 2. Definir clientes ───────────────────────────────────────────
    clientes = [
        Cliente("Cliente A", x=2,  y=4,  demanda=30),
        Cliente("Cliente B", x=5,  y=1,  demanda=50),
        Cliente("Cliente C", x=8,  y=7,  demanda=40),
        Cliente("Cliente D", x=3,  y=9,  demanda=20),
        Cliente("Cliente E", x=10, y=3,  demanda=35),
    ]

    # ── 3. Definir camiones y sus rutas ───────────────────────────────
    #  Camión 1 atiende los primeros 3 clientes
    #  Camión 2 atiende los últimos 2 clientes
    camion1 = Camion(
        nombre="Camión 1",
        capacidad=130,
        deposito=deposito,
        ruta=clientes[:3],         # A, B, C
        velocidad=1.0,             # km/min
        tiempo_entrega=5.0,        # min por cliente
    )

    camion2 = Camion(
        nombre="Camión 2",
        capacidad=80,
        deposito=deposito,
        ruta=clientes[3:],         # D, E
        velocidad=1.2,
        tiempo_entrega=4.0,
    )

    # ── 4. Crear y ejecutar simulación ────────────────────────────────
    print("\n" + "=" * 60)
    print("   SIMULACIÓN DE LOGÍSTICA CON SimPy")
    print("=" * 60)
    sim = Simulacion(deposito=deposito, camiones=[camion1, camion2])
    sim.iniciar()
    resumen = sim.resumen()

    # ── 5. Análisis con IA ────────────────────────────────────────────
    try:
        analista = AnalistaIA()
        analisis = analista.analizar(resumen)
        print("=" * 60)
        print("  ANÁLISIS DE IA (Gemini)")
        print("=" * 60)
        print(analisis)
        print("=" * 60 + "\n")
    except EnvironmentError as exc:
        print(f"\n[INFO] Análisis de IA omitido: {exc}\n")
    except Exception as exc:
        print(f"\n[ERROR] No se pudo obtener el análisis de IA: {exc}\n")


if __name__ == "__main__":
    main()
