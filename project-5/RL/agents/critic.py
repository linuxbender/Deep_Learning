from keras import layers, models, optimizers
from keras import backend as K

class Critic:

    def __init__(self, state_size, action_size, LEARNING_RATE):
        self.state_size = state_size
        self.action_size = action_size
        self.LEARNING_RATE = LEARNING_RATE

        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        net_states = layers.Dense(units=128, activation=None)(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=128, activation=None)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)

        net_actions = layers.Dense(units=128, activation=None)(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)        

        net_actions = layers.Dense(units=128, activation=None)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)        

        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('sigmoid')(net)

        Q_values = layers.Dense(units=1, name='q_values')(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = optimizers.Adam(lr=self.LEARNING_RATE)
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(Q_values, actions)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)