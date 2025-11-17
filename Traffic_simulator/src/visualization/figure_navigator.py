#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent


class FigureNavigator:
    def __init__(self, axes_list, title_prefix=""):
        self.axes_list = axes_list
        self.current_index = 0
        self.title_prefix = title_prefix
        self.total_figures = len(axes_list)

        if self.total_figures == 0:
            raise ValueError("Lista de figuras vacia")

        self.fig = axes_list[0].figure if axes_list else None

        if self.fig is None:
            raise ValueError("No se pudo obtener la figura de los axes")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event: KeyEvent):
        if event.key == 'right' or event.key == 'up':
            self.next_figure()
        elif event.key == 'left' or event.key == 'down':
            self.previous_figure()
        elif event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)

    def next_figure(self):
        self.current_index = (self.current_index + 1) % self.total_figures
        self.show_current()

    def previous_figure(self):
        self.current_index = (self.current_index - 1) % self.total_figures
        self.show_current()

    def show_current(self):
        for i, ax in enumerate(self.axes_list):
            ax.set_visible(False)
            if hasattr(ax, 'colorbar') and ax.colorbar:
                ax.colorbar.ax.set_visible(False)

        current_ax = self.axes_list[self.current_index]
        current_ax.set_visible(True)

        if hasattr(current_ax, 'colorbar') and current_ax.colorbar:
            current_ax.colorbar.ax.set_visible(True)

        self.fig.suptitle(
            f"{self.title_prefix}\n[{self.current_index + 1}/{self.total_figures}] - "
            f"Usa flechas < > o arriba/abajo para navegar, ESC para salir",
            fontsize=12,
            fontweight='bold'
        )

        self.fig.canvas.draw()

    def display(self):
        self.show_current()
        plt.show()


class DashboardNavigator:
    def __init__(self, axes_list, titles=None, title_prefix="", figsize=(16, 12)):
        self.axes_list = axes_list
        self.title_prefix = title_prefix
        self.total_figures = len(axes_list)
        self.titles = titles if titles else [f"Grafica {i+1}" for i in range(self.total_figures)]

        if self.total_figures == 0:
            raise ValueError("Lista de graficas vacia")

        self.fig = axes_list[0].figure if axes_list else None

        if self.fig is None:
            raise ValueError("No se pudo obtener la figura de los axes")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event: KeyEvent):
        if event.key == 'escape' or event.key == 'q':
            plt.close(self.fig)

    def display(self):
        self.fig.suptitle(
            f"{self.title_prefix}\nDashboard - Todas las métricas\nPresiona ESC para salir",
            fontsize=14,
            fontweight='bold'
        )
        self.fig.canvas.draw()
        plt.show()


def create_figure_with_subplots(plot_functions, titles, figsize=(14, 8)):
    if not plot_functions:
        raise ValueError("Lista de funciones vacia")

    fig = plt.figure(figsize=figsize)
    axes_list = []

    for i, plot_func in enumerate(plot_functions):
        ax = fig.add_subplot(1, 1, 1)
        plot_func(ax)
        ax.set_title(titles[i] if i < len(titles) else f"Grafica {i+1}")
        axes_list.append(ax)

    navigator = FigureNavigator(axes_list, title_prefix="Escenario")

    return fig, axes_list, navigator
