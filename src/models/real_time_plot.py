import pygame
import numpy as np

class RealTimePlot:
    """
    Classe para desenhar um gráfico da evolução do fitness em tempo real
    """
    def __init__(self, screen_width=640, screen_height=400, title="Evolução do fitness"):
        """
        Inicializa a janela do pygame
        """
        pygame.init()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.title = title
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(self.title)

        self.font = pygame.font.SysFont(None, 20)
        self.data = []
        self.colors = {
            "background": (255, 255, 255), # Fundo branco
            "line": (0, 100, 255), # Linha azul
            "axis": (0, 0, 0), # Eixos pretos
            "text": (50, 50, 50) # Texto cinza escuro
        }
        self.padding = 40
        self.draw()
    
    def update_data(self, new_point):
        """
        Adiciona um novo ponto de dados (fitness) ao gráfico e redesenha.
        """
        self.data.append(new_point)
        self.draw()
    
    def draw(self):
        """Desenha o estado atual do gráfico na tela."""
        self.screen.fill(self.colors["background"])

        # --- Desenha elementos estáticos (sempre visíveis) ---
        pygame.draw.line(self.screen, self.colors["axis"], (self.padding, self.padding), (self.padding, self.screen_height - self.padding), 1)
        pygame.draw.line(self.screen, self.colors["axis"], (self.padding, self.screen_height - self.padding), (self.screen_width - self.padding, self.screen_height - self.padding), 1)
        
        title_surface = self.font.render('Fitness por Geração', True, self.colors["text"])
        self.screen.blit(title_surface, (self.screen_width / 2 - title_surface.get_width() / 2, 5))
        
        xlabel = self.font.render('Geração', True, self.colors["text"])
        self.screen.blit(xlabel, (self.screen_width / 2 - xlabel.get_width() / 2, self.screen_height - self.padding + 10))

        # --- Desenha elementos dinâmicos (apenas se houver dados) ---
        if self.data:
            max_fitness = max(self.data)
            # Define o mínimo do eixo Y como 0, a menos que o fitness seja negativo
            min_fitness = min(self.data) if min(self.data) < 0 else 0 
            
            range_fitness = max_fitness - min_fitness
            if range_fitness == 0:
                range_fitness = 1 # Evita divisão por zero

            # Desenha a linha de dados
            points = []
            for i, value in enumerate(self.data):
                x = self.padding + i * (self.screen_width - 2 * self.padding) / (len(self.data) - 1 if len(self.data) > 1 else 1)
                y = (self.screen_height - self.padding) - ((value - min_fitness) / range_fitness) * (self.screen_height - 2 * self.padding)
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors["line"], False, points, 2)

            # Desenha os rótulos dos dados
            max_label = self.font.render(f'{max_fitness:.2f}', True, self.colors["text"])
            self.screen.blit(max_label, (self.padding - max_label.get_width() - 5, self.padding))
            
            min_label = self.font.render(f'{min_fitness:.2f}', True, self.colors["text"])
            self.screen.blit(min_label, (self.padding - min_label.get_width() - 5, self.screen_height - self.padding - min_label.get_height()))

            last_fitness_label = self.font.render(f'Atual: {self.data[-1]:.2f}', True, self.colors["line"])
            self.screen.blit(last_fitness_label, (self.screen_width - last_fitness_label.get_width() - 15, 15))

        # Atualiza a tela inteira para exibir as mudanças
        pygame.display.flip()
        
    def handle_events(self):
        """
        Lida com eventos do Pygame, como fechar a janela. Retorna False para sair.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        """
        Fecha o Pygame.
        """
        pygame.quit()