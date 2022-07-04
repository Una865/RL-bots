import keras.layers
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import adam_v2
from collections import deque
import numpy as np
import random
import pyautogui
import pytesseract
from PIL import Image
import base64
from tqdm import tqdm
from PIL import Image, ImageOps
EPISODES = 20000
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

import time
import os

REPLAY_MEMORY_SIZE = 10000
MODEL_NAME = 'v1'
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
TARGET_STEP = 5
gamma = 0.99

epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
MIN_REWARD = -200

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

ep_rewards = [-200]
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)

output_dir = 'model_output/game'

if not os.path.isdir('models'):
    os.makedirs('models')


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_fir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key,value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()


        #self._write_logs(stats, self.step)


class Agent:

    def __init__(self, input_shape, action_space_size):

        self.input_shape = input_shape
        self.action_space_size = action_space_size
        self.target_update_counter = 0
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.model = self.create_model()
        self.target_model = self.create_model()

        self.target_model.set_weights(self.model.get_weights())


    def create_model(self):
        model = Sequential()
        model.add(keras.layers.Input(shape = self.input_shape))
        model.add(Conv2D(256, (5, 5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(32,activation='linear'))
        model.add(Dense(self.action_space_size,activation="linear"))
        model.compile(loss = "mse", optimizer=adam_v2.Adam(lr =0.001))

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

    def get_qs(self, state):
        preds = np.array(state).reshape(-1, state.shape[0],state.shape[1],state.shape[2])
        return self.model.predict(preds)[0]

    def train(self,terminal_state,step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)
        curr_states = np.array([transition[0] for transition in minibatch])
        curr_qs_val = self.model.predict(curr_states)

        new_curr_states = np.array([transition[3] for transition in minibatch])
        future_qs_vals = self.target_model.predict(new_curr_states)

        x = []
        y = []

        for idx, (curr_state, action, reward, new_curr_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_vals[idx])
                new_q = reward + gamma * max_future_q
            else:
                new_q = reward

            curr_qs = curr_qs_val[idx]
            curr_qs[action] = new_q

            x.append(curr_state)
            y.append(curr_qs)


        self.model.fit(np.array(x),np.array(y),batch_size=MINIBATCH_SIZE,verbose=0,shuffle= False,callbacks=[self.tensorboard] if terminal_state else None)
        if terminal_state:
            self.target_update_counter += 1
        if self.target_update_counter > TARGET_STEP:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0





def test_agent_output_shape():
    img = np.zeros((5, 300, 300, 1))
    ag = Agent(img[0].shape, 7)
    pred = np.zeros((5,7))
    assert pred.shape == ag.model.predict(img).shape
    print("Model gives desired output shape")

#test_agent_output_shape()

def try_img_to_str():
    img = pyautogui.screenshot(region=(0, 30, 450, 800))
    img2 = img.resize((150,300))
    img.save('img1.jpg')
    img2.save('img2.jpg')

    text = pytesseract.image_to_string(Image.open('img1.jpg'))
    print(text)

class  GameEnv:
    size = 256
    observation_space_values = (2*size,size,1)
    action_space_size = 5

    def process_img(self, img):

        img = ImageOps.grayscale(img)
        img = img.resize((self.observation_space_values[1],self.observation_space_values[0]))
        img.save('img2.jpg')
        a = np.array(img)
        a = a / 255
        a = np.reshape(a,(self.observation_space_values))

        return a


    def reset(self):

        self.episode_step = 0
        while not self.check():
            pass
        pyautogui.click(x=300, y=600)
        time.sleep(2)
        img = pyautogui.screenshot(region=(0, 30, 450, 800))
        return self.process_img(img)

    def check(self):

        img = pyautogui.screenshot(region=(0, 30, 450, 800))
        text = pytesseract.image_to_string(img)

        if 'Tap to Play' in text:
            return True

        '''print("Enter 1 when game is ready to play")
        num = int(input())
        if(num == 1):
            return True
        else:
            return False'''
        return False

    def step(self,action):

        self.episode_step += 1
        self.do_action(action)

        done = False
        img = pyautogui.screenshot(region=(0, 30, 450, 800))
        text = pytesseract.image_to_string(img)
        img = ImageOps.grayscale(img)
        if 'Continue' in text:
            done = True
            reward = -20
            print("una")
            pyautogui.press('space')
            pyautogui.click(x = 30,y = 760, button = 'left')
        else:
            reward = 10

        return self.process_img(img),  reward, done


    def do_action(self,action):
        if (action == 0):  # left
            pyautogui.press('left')
        elif (action == 1):
            pyautogui.press('right')
        elif (action == 2):
            pyautogui.press('up')
        elif (action == 3):
            pyautogui.press('down')
        elif (action == 4):
            time.sleep(0.3)




def StartTrain():
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        episode_reward = 0
        step = 1
        agent.tensorboard.step = episode
        curr_state = env.reset()
        done = False
        global epsilon
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(curr_state))
            else:
                action = np.random.randint(0, agent.action_space_size)

            new_state, reward, done = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((curr_state, action, reward, new_state, done))
            agent.train(done, step)
            curr_state = new_state
            step += 1

            ep_rewards.append(episode_reward)

            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=epsilon)

                # Save model, but only when min reward is greater or equal a set value
                if average_reward >= MIN_REWARD:
                    agent.model.save(
                        f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


env = GameEnv()
#try_img_to_str()
input_shape = env.observation_space_values
action_size = env.action_space_size
agent = Agent(input_shape,action_size)

StartTrain()







