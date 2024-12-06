import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm
from keras import ops
from helper_functions import test_bot

#Define Battleship 
class BattleshipEnvironment:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.total_steps = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))  
        self.ships = []  
        
        ship_sizes = [2, 3, 3, 4, 5] 
        
        for i in range(5):
            ship_size = ship_sizes[i]
            
            ship_placement_row = np.random.randint(0, self.board_size - ship_size + 1) 
            ship_placement_column = np.random.randint(0, self.board_size - ship_size + 1)
            
            for n in range(ship_size):
                self.board[ship_placement_row][ship_placement_column + n] = 1
                self.ships.append((ship_placement_row, ship_placement_column + n))		
                
        return np.copy(self.board)

    def step(self, action):
        reward = 0
        done = False
        row, col = int(action[0]), int(action[1])
        
        if self.board[row][col] == 1:  
            reward = 4 
            self.board[row][col] = 2  
            if not any(1 in col for col in self.board):  
                done = True  
        elif self.board[row][col] == 0:
            reward = -1
            self.board[row][col] = -1  
        self.total_steps += 1
        return np.copy(self.board), reward, done

    def is_valid(self, coord):
        if self.board[coord[0]][coord[1]] == 0 or self.board[coord[0]][coord[1]] == 1:
            return True
        else:
            return False


#Build the DQN Model
@keras.saving.register_keras_serializable()
class DQNModel(tf.keras.Model):
    def __init__(self, num_actions=100, **kwargs):
        super(DQNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(1, 10, 1, activation='relu', padding='same')
        self.maxPool1 = tf.keras.layers.MaxPool2D(2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(1, 5, 1, activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(25, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        inputs = inputs[:,None,:]
        x = self.conv1(inputs)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
    
class ReplayBuffer:
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.buffer = []

	def add(self, experience):
		self.buffer.append(experience)
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)

	def sample(self, batch_size):
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		return [self.buffer[idx] for idx in indices]

class DQNTrainer:
	def __init__(self, model, target_model, replay_buffer, batch_size=16, gamma=0.01):
		self.model = model
		self.model_target = target_model
		self.replay_buffer = replay_buffer
		self.batch_size = batch_size
		self.gamma = gamma
		self.optimizer = tf.keras.optimizers.Adam()

	def train_step(self):
		batch = self.replay_buffer.sample(self.batch_size)
		states, actions, rewards, next_state, dones = zip(*batch)

		states = np.array(states)
		next_state = np.array(next_state)
		q_vals = self.model(states)
		next_q_values = self.model_target(next_state)

		targets = np.copy(q_vals)
  
		for i in range(self.batch_size):
			target = rewards[i]
			if not dones[i]:
				target += self.gamma * np.max(next_q_values[i])
			targets[i][actions[i]] = target

		with tf.GradientTape() as tape:
			predictions = self.model(states)
			loss = tf.reduce_mean(tf.square(targets - predictions))
   
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

	def update_target_net(self):
		self.model_target.set_weights(self.model.get_weights())

class DQNAgent:
	def __init__(self, model, num_actions):
		self.model = model
		self.num_actions = num_actions

	def attack(self, state, valid_moves):
		q_values = self.model(np.expand_dims(state, axis=0)).numpy().flatten()
		valid_q_values = [q_values[i] if i in valid_moves else float('-inf') for i in range(self.num_actions)]
		action = np.argmax(valid_q_values)
		return action

def train_model():
    env = BattleshipEnvironment()
    num_actions = env.board_size ** 2 
    model = DQNModel(num_actions)
    target_model = DQNModel(num_actions)
    replay_buffer = ReplayBuffer(buffer_size=1000)

    trainer = DQNTrainer(model, target_model, replay_buffer)
    num_epochs= 3000
    epsilon = 0.2  
    update_steps = 100  
    agent = DQNAgent(model, num_actions)

    top = float('inf')

    for epoch in tqdm(range(num_epochs)):
        state = env.reset()
        done = False
        reward_total = 0
        shots_total = 0
        while not done:
            
            if np.random.rand() < epsilon:
                action = np.random.choice([i for i in range(num_actions) if env.is_valid((i // env.board_size, i % env.board_size))])
            else:
                action = agent.attack(state, [i for i in range(num_actions) if env.is_valid((i // env.board_size, i % env.board_size))])

            next_state, reward, done = env.step((action / env.board_size, action % env.board_size))
            replay_buffer.add((state, action, reward, next_state, done))
            reward_total += reward
            shots_total += 1
            state = next_state

            if len(replay_buffer.buffer) >= trainer.batch_size:
                trainer.train_step()

            if env.total_steps % update_steps == 0:
                trainer.update_target_net()

        if shots_total < top:
            top = shots_total
        print("Reward:", reward_total, "Shots:", shots_total)
        
        
        try:
            if epoch == 150:
                model.save("bot_dqn_updated_150.keras")
            
            if epoch % 500 == 0:
                model.save(f"bot_dqn_updated_{epoch}.keras")
        except:
            pass
         
        model.save("bot_dqn_final.keras")   

    print("Model Trained!\n")

def test_model(env, agent):
    num_actions = env.board_size ** 2
    model = keras.models.load_model("bot_dqn.keras")
    agent = DQNAgent(model, num_actions)
    total_shots = []
    for _ in tqdm(range(100)):
        state = env.reset()
        done = False
        shots = 0
        while not done:
            action = agent.attack(state, [i for i in range(num_actions) if env.is_valid_move((i // env.board_size, i % env.board_size))])
            state, _, done = env.step((action // env.board_size, action % env.board_size))
            shots += 1
        total_shots.append(shots)
    return total_shots

class DQNBot: #for testing
    def __init__(self, board):
        self.board = board
        self.env = BattleshipEnvironment(board_size=board.get_size())
        self.num_actions = self.env.board_size ** 2
        self.model = keras.models.load_model("bot_dqn_final.keras")
        self.agent = DQNAgent(self.model, self.num_actions)
        self.state = self.env.reset()
        self.valid_moves = set(range(self.num_actions))  

    def attack(self):
        action = self.agent.attack(self.state, list(self.valid_moves))
        row, col = divmod(action, self.env.board_size)

        self.board.attack(row, col)

        self.state, _, _ = self.env.step((row, col))

        self.valid_moves.discard(action)

if __name__ == '__main__':
    # train_model()
    
    # model = keras.models.load_model("bot_dqn.keras")
    
    total, mean, median, max_, min_, std, avg_hits_at_move, avg_moves_for_hit = test_bot(100, 10, [2, 3, 3, 4, 5], Bot=DQNBot)
    print(f"Mean: {mean}, Median: {median}, Max: {max_}, Min: {min_}, Std: {std}, avg_hits_at_move: {avg_hits_at_move}, avg_moves_for_hit: {avg_moves_for_hit}")