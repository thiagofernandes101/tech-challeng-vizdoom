import numpy as np

# Adicione esta classe perto do topo do seu arquivo, após as importações.
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializa os pesos com valores aleatórios pequenos
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        """Propagação direta (processo de 'pensar')."""
        x = np.array(x).reshape(1, -1) # Garante que a entrada seja uma matriz de linha
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.tanh(z1) # Função de ativação não-linear (tangente hiperbólica)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2[0] # Retorna o vetor de saída

    def get_weights_flat(self):
        """Pega todos os pesos e vieses e os achata em um único vetor (o 'genoma')."""
        return np.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()])

    def set_weights_from_flat(self, flat_weights):
        """Recebe um vetor de pesos achatado ('genoma') e o usa para configurar a rede."""
        w1_end = self.W1.size
        b1_end = w1_end + self.b1.size
        w2_end = b1_end + self.W2.size
        
        self.W1 = flat_weights[0:w1_end].reshape(self.W1.shape)
        self.b1 = flat_weights[w1_end:b1_end].reshape(self.b1.shape)
        self.W2 = flat_weights[b1_end:w2_end].reshape(self.W2.shape)
        self.b2 = flat_weights[w2_end:].reshape(self.b2.shape)

    @property
    def total_weights(self):
        """Calcula o número total de pesos/vieses, que será o nosso GENOME_LENGTH."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size