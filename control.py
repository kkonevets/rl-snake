"""
This is an implementation of some RL control algorithms
from "AG Barto, RS Sutton" book for snake game
"""

from collections import defaultdict
import time
import pickle
import itertools
import os
import numpy as np
import sys
import pygame

from models import Snake, plot_game


def epsilon_soft_choices(n, m, epsilon):
    """choose n random integers 0<=x<m from epsilon-soft distribution,
    first element has the biggest probability"""
    p_soft = np.full(m, epsilon / m)
    p_soft[0] = 1 - epsilon + epsilon / m
    return np.random.choice(m, size=n, p=p_soft)


def swap_positions(lst: list, pos1, pos2):
    """swap list elements"""
    lst[pos1], lst[pos2] = lst[pos2], lst[pos1]
    return lst


def ddf():
    "need global function to be able to pickle defaultdict"
    return defaultdict(float)


class Control:
    "Base class for optimal control algorithms"

    def __init__(self, game, env, epsilon=0.1):
        self.game, self.env, self.epsilon = game, env, epsilon
        self.pi, self.Q = {}, defaultdict(ddf)
        # prechoose big number of random numbers 0<=x<3 (optimization)
        # 3 available actions: [forward, left, right]
        self.choices = epsilon_soft_choices(1000 * (env.x + env.y), 3, epsilon)

    def epsilon_soft_action(self, actions: list, step):
        "Choose action according to epsilon-soft distribution"
        if step > self.choices.size - 1:
            raise RuntimeError(
                "too many random walk steps: ", self.choices.size
            )
        ix = self.choices[step]  # explore in train mode
        return actions[ix]

    def greedy_actions(self, state, qs):
        """Select action with max value and set it to be first in action list"""
        actions = self.env.available_actions(state)
        a_max = max(qs, key=qs.get)
        return swap_positions(actions, 0, actions.index(a_max))

    def run_episode(self, snake):
        raise NotImplementedError("Abstarct method, override in subclass")

    def loadQ(self):
        if not os.path.isfile("Q.pkl"):
            print("Error: file Q.pkl not found, did you train the snake?")
            sys.exit(-1)
        with open("Q.pkl", "rb") as f:
            epi, self.Q = pickle.load(f)
            for state, qs in self.Q.items():
                if len(qs):
                    self.pi[state] = self.greedy_actions(state, qs)
            self.print_stat(epi)
        return epi

    def follow(self, delay=1):
        """follow policy and visualize steps"""

        try:
            start = self.loadQ()
            for epi in itertools.count(start):  # while True
                snake = Snake(self.env)
                plot_game(self.game, self.env, snake)
                for step in self.count(0, snake):  # while not game over
                    _, state = self.env.state(snake)

                    actions = self.pi.get(state)
                    if actions is None:
                        actions = self.env.available_actions(state, True)

                    snake.move(actions[0])  # take greedy action

                    plot_game(self.game, self.env, snake)
                    time.sleep(delay)
        except KeyboardInterrupt:
            pass

    def train(self, load=False):
        start = self.loadQ() if load else 0

        try:
            for epi in itertools.count(start):  # while True
                snake = Snake(self.env)
                self.run_episode(snake)
                if epi % 1000 == 0:
                    self.print_stat(epi)
        except KeyboardInterrupt:
            print("\r")
            with open("Q.pkl", "wb") as f:
                pickle.dump((epi, self.Q), f)

    def count(self, start, snake):
        "a wrapper to count steps and check game over conditions"
        for step in itertools.count(start):
            yield step

            if self.env.is_game_over(snake):
                self.env.score = 0
                break

    def print_stat(self, epi):
        print(
            "Episode %i, # of states %i, # of state-value pairs: %i\r"
            % (
                epi,
                len(self.Q),
                sum((len(v) for k, v in self.Q.items())),
            ),
            end="",
        )


class MonteCarlo(Control):
    def __init__(self, game, env, epsilon=0.1):
        super().__init__(game, env, epsilon)
        self.Returns = defaultdict(lambda: [0, 0])

    def run_episode(self, snake):
        episode = []
        for step in self.count(0, snake):  # while not game over
            state_vec, state = self.env.state(snake)

            if step == 0:  # exploring start
                actions = self.env.available_actions(state, True)
            else:
                actions = self.pi.get(state)
                if actions is None:
                    actions = self.env.available_actions(state, True)

            action = self.epsilon_soft_action(actions, step)
            reward = self.env.reward(snake, state, state_vec, action)
            episode.append((state, action, reward))

            snake.move(action)

        self.backtrace(episode)

    def backtrace(self, episode):
        "backpropagate cumulative reward G"
        G = 0
        for i, (state, action, reward) in enumerate(episode[::-1]):
            G = G + reward
            found = (
                1
                for s, a, _ in episode[: -i - 1]
                if state == s and action == a
            )
            if next(found, None) is None:
                ret = self.Returns[(state, action)]
                ret[0] += G
                ret[1] += 1
                qs = self.Q[state]
                qs[action] = ret[0] / ret[1]
                self.pi[state] = self.greedy_actions(state, qs)


class TemporalDifference(Control):
    def __init__(self, game, env, epsilon=0.1, alpha=0.05):
        super().__init__(game, env, epsilon)
        self.alpha = alpha

    def epsilon_greedy_action(self, qs, state, step):
        if step == 0 or len(qs) == 0:  # exploring start
            actions = self.env.available_actions(state, True)
        else:
            actions = self.greedy_actions(state, qs)
        return self.epsilon_soft_action(actions, step)

    def state_actions(self, snake):
        state_vec, state = self.env.state(snake)
        return state_vec, state, self.Q[state]


class Sarsa(TemporalDifference):
    def __init__(self, game, env, **kwargs):
        super().__init__(game, env, **kwargs)

    def run_episode(self, snake):
        state_vec1, state1, qs1 = self.state_actions(snake)
        action1 = self.epsilon_greedy_action(qs1, state1, 0)

        for step in self.count(1, snake):  # while not game over
            reward = self.env.reward(snake, state1, state_vec1, action1)

            snake.move(action1)

            state_vec2, state2, qs2 = self.state_actions(snake)
            action2 = self.epsilon_greedy_action(qs2, state2, step)

            q1 = qs1[action1]
            qs1[action1] = q1 + self.alpha * (reward + qs2[action2] - q1)

            state_vec1, state1, qs1, action1 = state_vec2, state2, qs2, action2


class QLearning(TemporalDifference):
    def __init__(self, game, env, **kwargs):
        super().__init__(game, env, **kwargs)

    def run_episode(self, snake):
        state_vec1, state1, qs1 = self.state_actions(snake)

        for step in self.count(0, snake):  # while not game over
            action = self.epsilon_greedy_action(qs1, state1, step)
            reward = self.env.reward(snake, state1, state_vec1, action)

            snake.move(action)
            state_vec2, state2, qs2 = self.state_actions(snake)

            q1 = qs1[action]
            q2_max = max(qs2.values(), default=0)
            qs1[action] = q1 + self.alpha * (reward + q2_max - q1)

            state_vec1, state1, qs1 = state_vec2, state2, qs2


def debug(game, env):
    """move snake by hand and print variables"""
    key = None
    snake = Snake(env)

    print("INIT")
    _, state = env.state(snake)
    print("{0:b}".format(state)[::-1])

    plot_game(game, env, snake)

    def game_over():
        pygame.quit()
        sys.exit()

    # Main logic
    while not env.is_game_over(snake):
        event = pygame.event.wait()
        # Whenever a key is pressed down
        if event.type == pygame.QUIT:
            game_over()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                game_over()
            elif pygame.K_RIGHT <= event.key <= pygame.K_UP:
                key = event.key
            else:
                continue
        else:
            continue

        snake.move(key)

        plot_game(game, env, snake)

        state_vec, state = env.state(snake)
        print(
            "state %s, Reward if moving (LEFT, RIGHT, UP, DOWN): (%i, %i, %i, %i)"
            % (
                "{0:b}".format(state)[::-1],
                env.reward(snake, state, state_vec, pygame.K_LEFT),
                env.reward(snake, state, state_vec, pygame.K_RIGHT),
                env.reward(snake, state, state_vec, pygame.K_UP),
                env.reward(snake, state, state_vec, pygame.K_DOWN),
            ),
        )
