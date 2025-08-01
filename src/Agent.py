import numpy as np
import tensorflow as tf
from collections import deque
import random

class Agent:
    """
    The Agent that learns to play. It contains the neural network (model)
    and the experience replay memory (replay buffer).
    """
    def __init__(self, state_size: int, action_size: int):
        """
        Initializes the Agent.

        Args:
            state_size (int): The dimensionality of the state vector.
            action_size (int): The number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Memory for Experience Replay
        self.memory = deque(maxlen=2000)
        
        # Learning parameters
        self.gamma = 0.95      # Discount factor
        self.epsilon = 1.0     # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Builds the LSTM neural network."""
        # The input needs to be shaped for an LSTM: (batch, timesteps, features)
        # Here, we assume 1 timestep per observation.
        inputs = tf.keras.Input(shape=(1, self.state_size))
        
        # LSTM Layer
        x = tf.keras.layers.LSTM(64, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_size, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores a transition in the memory buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        """
        Decides an action using the epsilon-greedy policy.

        Args:
            state (np.ndarray): The processed state vector.

        Returns:
            int: The index of the action to take.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Random action (exploration)
        
        # Reshape state for the LSTM: (1, 1, state_size)
        formatted_state = np.reshape(state, [1, 1, self.state_size])
        q_values = self.model.predict(formatted_state, verbose=0)
        return np.argmax(q_values[0]) # Best action (exploitation)

    def replay(self, batch_size: int):
        """Trains the network with a batch of transitions from memory."""
        if len(self.memory) < batch_size:
            return # Don't train if memory doesn't have enough samples

        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Reshape states for the LSTM
            state_fmt = np.reshape(state, [1, 1, self.state_size])
            next_state_fmt = np.reshape(next_state, [1, 1, self.state_size])
            
            target = reward
            if not done:
                # Bellman equation for the target Q-value
                target = reward + self.gamma * np.amax(self.model.predict(next_state_fmt, verbose=0)[0])
            
            target_q = self.model.predict(state_fmt, verbose=0)
            target_q[0][action] = target
            
            self.model.fit(state_fmt, target_q, epochs=1, verbose=0)

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay