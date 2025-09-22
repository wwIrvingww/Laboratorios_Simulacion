"""
Estructura:
1. Calcular distancias
2. Definición de variables de decisión: recordar son variables binarias, (sí o no)
   Además, acá es por pares por ejemplo Xij, esto representa si viaja de i a j.
3. Función Objetivo: Es la parte más importante ya que acá es donde minimiza la distancia total
   si Xij es 1, entonces significa que sí la vamos a añadir al recorrido.
4. Reestriciones: 
    a) Cada ciudad debe tener una salida. No se puede salir dos veces de la misma ciudad, ni quedarse atrapado.
    b) Cada ciudad debe tener SOLO una entrada (no se puede visitar dos veces)
    c) No se puede viajar de una ciudad a sí misma
5. Evitar subciclos: Obligatoriamente tiene que ser una ruta que pase por todas las ciudades.
6. Resolver

"""