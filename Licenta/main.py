# main.py
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

import frogger
from Environment import FroggerEnv
from Agent import Agent

import torch


# Funcție principală de antrenament
def train(num_actions, num_episodes, scores, env, agent, scor_final, avg_scores):

    max_lane = 0
    min_lane = 10000
    avg_lane = 0
    lanes = []

    for episode in range(num_episodes):

        state = env.reset()
        env.gameApp.state = "PLAYING"
        env.gameApp.draw()

        total_reward = 0
        lane = 1

        counter = 0
        while True:
            counter += 1

            action = agent.choose_action(agent.prepare_input(state))
            if action == 0:
                lane += 1
            elif action == 1 and lane > 1:
                lane -= 1
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            agent.store_transition(agent.prepare_input(state), action, reward, agent.prepare_input(next_state), done)

            agent.train()
            state = next_state

            if done:
                break

        lanes.append(lane)

        if lane > max_lane:
            max_lane = lane
        if lane < min_lane:
            min_lane = lane
        avg_lane += lane

        scores.append(env.highest_lane)

        average_reward = np.mean(scores[-100:])
        avg_scores.append(average_reward)
        agent.update_target_nn()
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Avarage reward: {average_reward} Epsilon: {agent.eps}")

    print(scor_final.high_score, "max lane: ", max_lane, "min lane: ", min_lane, "avg lane: ", avg_lane)

    with open("date_ddql.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(lanes)

    agent.save_model()


def test(env, agent):

    test_rewards = []

    for i in range(5):
        agent.load_model()
        state = env.reset()
        total_reward = 0

        env.gameApp.draw()
        env.gameApp.state = "PLAYING"

        while True:

            action = agent.choose_action_test(agent.prepare_input(state))
            print("Am facut actiunea: ", action)
            next_state, reward, done, _ = env.step(action)
            print("Am murit: ", done)
            print("am primit: ", reward)
            time.sleep(0.5)
            env.gameApp.draw()

            state = next_state
            total_reward += reward

            if done:
                print(f"Total Reward: {total_reward}")
                break

        test_rewards.append(total_reward)
        print(f"Episode: {i + 1}, Total Reward: {total_reward}")

    av_reward = sum(test_rewards) / len(test_rewards)
    print("Avarage reward: ", av_reward)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    env = FroggerEnv()

    num_actions = 5
    num_episodes = 5000
    scores = []
    avg_scores = []

    agent = Agent(num_actions, input_dims=[1, 84, 84], batch_size=128)
    scor_final = frogger.Score()

    train(num_actions, num_episodes, scores, env, agent, scor_final, avg_scores)

    test(env, agent)

    plt.figure(figsize=(20, 10))
    plt.plot(scores, label="Reward per Episode")
    plt.plot(avg_scores, label="Average of last 100 episodes", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Agent Performance Over Time")
    plt.legend()
    plt.show()

