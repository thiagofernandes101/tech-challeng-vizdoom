# NeuralNetwork.py - Implementação com LSTM
from typing import Tuple
import numpy as np

class NeuralNetwork:
    """
    Uma Rede Neural Recorrente com uma camada LSTM e uma camada de saída densa.
    Implementada do zero usando apenas numpy.
    """
    DEFAULT_INPUT_SIZE: int = 8
    DEFAULT_LSTM_HIDDEN_SIZE: int = 128

    def __init__(self, input_size, lstm_hidden_size, output_size):
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size
        
        # --- Pesos e Biases para a Camada LSTM ---
        # A entrada para as portas LSTM é a concatenação do input atual (x) e do hidden state anterior (h)
        combined_size = input_size + lstm_hidden_size
        
        # Portão de Esquecimento (Forget Gate)
        self.Wf = np.zeros((combined_size, lstm_hidden_size))
        self.bf = np.zeros((1, lstm_hidden_size))
        
        # Portão de Entrada (Input Gate)
        self.Wi = np.zeros((combined_size, lstm_hidden_size))
        self.bi = np.zeros((1, lstm_hidden_size))
        
        # Portão de Candidatos (Candidate Gate)
        self.Wc = np.zeros((combined_size, lstm_hidden_size))
        self.bc = np.zeros((1, lstm_hidden_size))
        
        # Portão de Saída (Output Gate)
        self.Wo = np.zeros((combined_size, lstm_hidden_size))
        self.bo = np.zeros((1, lstm_hidden_size))
        
        # --- Pesos e Biases para a Camada de Saída Densa ---
        self.Wy = np.zeros((lstm_hidden_size, output_size))
        self.by = np.zeros((1, output_size))
        
        # Calcula o tamanho total do genoma
        self.total_weights = (
            4 * (self.Wf.size + self.bf.size) + 
            self.Wy.size + self.by.size
        )

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x):
        return np.tanh(x)

    def set_weights_from_flat(self, flat_weights):
        if flat_weights.size != self.total_weights:
            raise ValueError(f"Tamanho do genoma incorreto. Esperado: {self.total_weights}, Recebido: {flat_weights.size}")

        pointer = 0
        
        # Desempacota pesos para a camada LSTM
        for w_matrix in [self.Wf, self.Wi, self.Wc, self.Wo]:
            w_size = w_matrix.size
            w_shape = w_matrix.shape
            w_matrix[:] = flat_weights[pointer : pointer + w_size].reshape(w_shape)
            pointer += w_size
            
        for b_vector in [self.bf, self.bi, self.bc, self.bo]:
            b_size = b_vector.size
            b_shape = b_vector.shape
            b_vector[:] = flat_weights[pointer : pointer + b_size].reshape(b_shape)
            pointer += b_size
            
        # Desempacota pesos para a camada de saída
        w_size = self.Wy.size
        w_shape = self.Wy.shape
        self.Wy[:] = flat_weights[pointer : pointer + w_size].reshape(w_shape)
        pointer += w_size
        
        b_size = self.by.size
        b_shape = self.by.shape
        self.by[:] = flat_weights[pointer : pointer + b_size].reshape(b_shape)

    def forward(self, x, hidden):
        """
        Executa uma passagem para a frente para um único passo de tempo.
        x: vetor de entrada para o passo atual.
        hidden: tupla (h_prev, c_prev) contendo o estado oculto e o estado de célula anteriores.
        """
        h_prev, c_prev = hidden
        
        # Concatena a entrada atual com o estado oculto anterior
        combined = np.concatenate((x.reshape(1, -1), h_prev), axis=1)
        
        # Cálculos das portas da LSTM
        ft = self._sigmoid(combined @ self.Wf + self.bf) # Portão de Esquecimento
        it = self._sigmoid(combined @ self.Wi + self.bi) # Portão de Entrada
        c_tilde = self._tanh(combined @ self.Wc + self.bc)  # Valores Candidatos
        
        # Atualiza o estado da célula (a memória de longo prazo)
        c_next = ft * c_prev + it * c_tilde
        
        # Calcula o portão de saída e o novo estado oculto (a memória de trabalho)
        ot = self._sigmoid(combined @ self.Wo + self.bo) # Portão de Saída
        h_next = ot * self._tanh(c_next)
        
        # Calcula a saída final (pontuações de ação)
        output_scores = h_next @ self.Wy + self.by
        
        return output_scores.flatten(), (h_next, c_next)

    def init_hidden(self):
        """Retorna um estado oculto inicializado com zeros."""
        h0 = np.zeros((1, self.lstm_hidden_size))
        c0 = np.zeros((1, self.lstm_hidden_size))
        return (h0, c0)
    
    @classmethod
    def from_action_space_size(cls, output_size: int) -> 'NeuralNetwork':
        """
        Cria uma instância da rede neural configurada com base no tamanho
        do espaço de ações, usando tamanhos de entrada e LSTM padrão.

        Args:
            output_size (int): O número de ações possíveis, que define o tamanho da camada de saída.

        Returns:
            NeuralNetwork: Uma nova instância da rede neural configurada.
        """
        # Acessa as configurações padrão definidas como atributos de classe
        input_size: int = cls.DEFAULT_INPUT_SIZE
        lstm_hidden_size: int = cls.DEFAULT_LSTM_HIDDEN_SIZE
        
        # Instancia a própria classe (cls) com as configurações
        return cls(input_size, lstm_hidden_size, output_size)

    def get_config(self) -> Tuple[int, int, int]:
        """
        Retorna a tupla de configuração (input_size, lstm_hidden_size, output_size)
        desta instância da rede neural.
        """
        return (self.input_size, self.lstm_hidden_size, self.output_size)