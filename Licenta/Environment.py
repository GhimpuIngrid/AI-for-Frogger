import pygame
import numpy as np
from pygame.locals import *
from frogger import App
from config import g_vars

class FroggerEnv:
    def __init__(self):
        self.lives_before = 1
        self.gameApp = App()
        self.action_space = 5
        self.last_action = None
        self.observation_space = (g_vars['width'], g_vars['height'], 3)
        self.highest_lane = None
        self.reset()

    def reset(self):
        self.gameApp.init()
        self.gameApp.state = "PLAYING"
        self.gameApp.draw()
        state = self.get_state()
        self.highest_lane = 0
        return state

    def step(self, action):
        self.lives_before = self.gameApp.score.lives
        self.last_action = action
        pygame.event.pump()
        if action == 0:
            self.gameApp.frog.move(0, -1)  # Sus
        elif action == 1:
            self.gameApp.frog.move(0, 1)  # Jos
        elif action == 2:
            self.gameApp.frog.move(-1, 0)  # Stânga
        elif action == 3:
            self.gameApp.frog.move(1, 0)  # Dreapta
        elif action == 4:
            pass  # Sta pe loc

        self.gameApp.update()
        self.gameApp.draw()
        next_state = self.get_state()
        reward = self.get_reward()
        done = self.is_done()

        return next_state, reward, done, {}

    def render(self):
        self.gameApp.draw()
        self.gameApp.clock.tick(g_vars['fps'])

    def get_state(self):
        state = pygame.surfarray.array3d(g_vars['window'])
        state = np.transpose(state, (1, 0, 2))
        return state

    def get_reward(self):
        reward = 0
        current_lane = self.gameApp.frog.y

        if self.gameApp.score.lives < self.lives_before:
            reward -= 50  # Penalizare mare pentru pierderea unei vieți

        elif self.gameApp.current_lane == 6:
            reward += 50

        elif self.gameApp.current_lane == 12:
            reward += 100

        elif self.last_action == 0:
            if self.gameApp.current_lane > self.highest_lane:
                self.highest_lane = self.gameApp.current_lane
                reward += 10 * self.gameApp.current_lane
            reward += 2  # Recompensă mare pentru progres

        elif self.last_action == 1:
            if self.gameApp.current_lane == 7:
                reward -= 50
            else:
                reward -= 2  # Penalizam pentru intoarcere

        elif self.last_action == 4:
            reward -= 0  # Penalizare mică pentru inactivitate

        elif self.last_action in [2, 3]:
            reward -= 0  # Penalizare mică pentru mișcare fără progres

        return reward

    def is_done(self):
        return self.gameApp.score.lives == 0

    def close(self):
        pygame.quit()
