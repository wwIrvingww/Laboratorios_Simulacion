#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navegador de Figuras Interactivo.

Permite navegar entre multiples figuras usando flechas del teclado.
Una sola ventana por escenario.
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent


class FigureNavigator:
    """Maneja navegacion entre multiples figuras en UNA sola ventana."""

    def __init__(self, axes_list, title_prefix=""):
        """
        Inicializa el navegador con una sola figura que contiene multiples axes.

        Parametros:
            axes_list (list): Lista de objetos matplotlib axes
            title_prefix (str): Prefijo para el titulo de la ventana
        """
        self.axes_list = axes_list
        self.current_index = 0
        self.title_prefix = title_prefix
        self.total_figures = len(axes_list)

        if self.total_figures == 0:
            raise ValueError("Lista de figuras vacia")

        # Obtener la figura de los axes
        self.fig = axes_list[0].figure if axes_list else None

        if self.fig is None:
            raise ValueError("No se pudo obtener la figura de los axes")

        # Conectar evento de teclado solo una vez a la figura
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event: KeyEvent):
        """Maneja eventos de teclado."""
        if event.key == 'right' or event.key == 'up':
            self.next_figure()
        elif event.key == 'left' or event.key == 'down':
            self.previous_figure()
        elif event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)

    def next_figure(self):
        """Muestra la siguiente figura."""
        self.current_index = (self.current_index + 1) % self.total_figures
        self.show_current()

    def previous_figure(self):
        """Muestra la figura anterior."""
        self.current_index = (self.current_index - 1) % self.total_figures
        self.show_current()

    def show_current(self):
        """Muestra solo el subplot actual."""
        # Ocultar todos los axes y sus colorbars
        for i, ax in enumerate(self.axes_list):
            ax.set_visible(False)
            # Ocultar colorbar si existe
            if hasattr(ax, 'colorbar') and ax.colorbar:
                ax.colorbar.ax.set_visible(False)

        # Mostrar solo el actual
        current_ax = self.axes_list[self.current_index]
        current_ax.set_visible(True)

        # Mostrar colorbar si existe
        if hasattr(current_ax, 'colorbar') and current_ax.colorbar:
            current_ax.colorbar.ax.set_visible(True)

        # Actualizar titulo
        self.fig.suptitle(
            f"{self.title_prefix}\n[{self.current_index + 1}/{self.total_figures}] - "
            f"Usa flechas < > o arriba/abajo para navegar, ESC para salir",
            fontsize=12,
            fontweight='bold'
        )

        self.fig.canvas.draw()

    def display(self):
        """Muestra la figura con navegacion."""
        self.show_current()
        plt.show()


def create_figure_with_subplots(plot_functions, titles, figsize=(14, 8)):
    """
    Crea una sola figura con multiples subplots para navegacion.

    Parametros:
        plot_functions (list): Lista de funciones que crean plots
        titles (list): Titulos para cada subplot
        figsize (tuple): Tamaño de la figura

    Retorna:
        tuple: (figure, axes_list, navigator)
    """
    if not plot_functions:
        raise ValueError("Lista de funciones vacia")

    # Crear figura con un subplot inicial
    fig = plt.figure(figsize=figsize)
    axes_list = []

    # Crear todos los subplots
    for i, plot_func in enumerate(plot_functions):
        ax = fig.add_subplot(1, 1, 1)
        plot_func(ax)
        ax.set_title(titles[i] if i < len(titles) else f"Grafica {i+1}")
        axes_list.append(ax)

    # Crear navegador
    navigator = FigureNavigator(axes_list, title_prefix="Escenario")

    return fig, axes_list, navigator
