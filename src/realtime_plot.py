# realtime_plot.py
import matplotlib.pyplot as plt

class RealtimePlot:
    def __init__(self, max_gens=50):
        self.best_fits = []
        self.avg_fits = []
        self.max_gens = max_gens

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line_best, = self.ax.plot([], [], label='Melhor', color='blue')
        self.line_avg, = self.ax.plot([], [], label='Média', color='orange')
        self.ax.set_xlim(0, max_gens)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel("Geração")
        self.ax.set_ylabel("Pontuação")
        self.ax.set_title("Evolução do Algoritmo Genético")
        self.ax.legend()
        self.fig.canvas.draw()

    def update(self, best, avg):
        self.best_fits.append(best)
        self.avg_fits.append(avg)

        self.line_best.set_data(range(len(self.best_fits)), self.best_fits)
        self.line_avg.set_data(range(len(self.avg_fits)), self.avg_fits)

        # Ajusta eixo Y se necessário
        current_max = max(self.best_fits + self.avg_fits)
        if current_max > self.ax.get_ylim()[1]:
            self.ax.set_ylim(0, current_max + 10)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize(self):
        plt.ioff()
        plt.show()
