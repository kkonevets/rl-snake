from collections import defaultdict
import time
import pickle
import itertools
import os
import numpy as np
import sys
import pygame

from models import Snake, Environment, plot_game


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
    return defaultdict(float)


class Control:
    "Base class for optimal control algorithms"

    def __init__(self, game, env, epsilon=0.1):
        self.game, self.env, self.epsilon = game, env, epsilon
        self.pi, self.Q = {}, defaultdict(ddf)
        # prechoose big number of random numbers 0<=x<3 (optimization)
        # 3 available actions: [forward, left, right]
        self.choices = epsilon_soft_choices(1000 * (env.x + env.y), 3, epsilon)

    def epsilon_soft_action(self, actions: list, step, train):
        "Choose action according to epsilon-soft distribution"
        if train:
            if step > self.choices.size - 1:
                raise RuntimeError(
                    "too many random walk steps: ", self.choices.size
                )
            ix = self.choices[step]  # explore in train mode
        else:
            ix = 0  # always select the best action in test mode

        return actions[ix]

    def greedy_actions(self, state, qs):
        """Select action with max value and set it to be first in action list"""
        actions = Environment.available_actions(self.env.direction(state))
        a_max = max(qs, key=qs.get)
        return swap_positions(actions, 0, actions.index(a_max))

    def run_episode(self, snake, train=True, delay=1):
        raise NotImplementedError("Abstarct method, override in subclass")

    def run(self, train=True, delay=1):
        if not train:
            if not os.path.isfile("Q.pkl"):
                print("Error: file Q.pkl not found, did you train the snake?")
                sys.exit(-1)
            with open("Q.pkl", "rb") as f:
                epi, self.Q = pickle.load(f)
                for state, qs in self.Q.items():
                    self.pi[state] = self.greedy_actions(state, qs)
                self.print_stat(epi)

        try:
            for epi in itertools.count():  # while True
                snake = Snake(self.env)
                if self.game:
                    plot_game(self.game, self.env, snake)

                self.run_episode(snake, train, delay)
                if train and epi % 1000 == 0:
                    self.print_stat(epi)

        except KeyboardInterrupt:
            print("\r")
            if train:
                with open("Q.pkl", "wb") as f:
                    pickle.dump((epi, self.Q), f)

    def is_game_over(self, snake):
        "game over conditions"
        return (
            not self.env.check_borders(snake)
            or len(snake.body) == self.env.x * self.env.y
        )

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

    def run_episode(self, snake, train=True, delay=1):
        episode = []
        for step in itertools.count():  # while True
            state_vec, state = self.env.state(snake)
            direction = self.env.direction(state)

            if step == 0 and train:  # exploring start
                actions = Environment.available_actions(direction, True)
            else:
                actions = self.pi.get(state)
                if actions is None:
                    actions = Environment.available_actions(direction, True)

            action = self.epsilon_soft_action(actions, step, train)
            if train:
                reward = self.env.reward(snake, state, state_vec, action)
                episode.append((state, action, reward))

            snake.move(action)

            if self.is_game_over(snake):
                self.env.score = 0
                break

            if self.game:
                plot_game(self.game, self.env, snake)
                time.sleep(delay)

        if train:
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

    def epsilon_greedy_action(self, qs, state, step, train: bool):
        if (step == 0 and train) or len(qs) == 0:  # exploring start
            actions = Environment.available_actions(
                self.env.direction(state), True
            )
        else:
            actions = self.greedy_actions(state, qs)
        return self.epsilon_soft_action(actions, step, train)

    def state_actions(self, snake):
        state_vec, state = self.env.state(snake)
        return state_vec, state, self.Q[state]


class Sarsa(TemporalDifference):
    def __init__(self, game, env, epsilon=0.1, alpha=0.05):
        super().__init__(game, env, epsilon, alpha)

    def run_episode(self, snake, train=True, delay=1):
        state_vec1, state1, qs1 = self.state_actions(snake)
        action1 = self.epsilon_greedy_action(qs1, state1, 0, train)

        for step in itertools.count(1):  # while True
            if train:
                reward = self.env.reward(snake, state1, state_vec1, action1)

            snake.move(action1)

            state_vec2, state2, qs2 = self.state_actions(snake)
            action2 = self.epsilon_greedy_action(qs2, state2, step, train)

            if train:
                q1 = qs1[action1]
                qs1[action1] = q1 + self.alpha * (reward + qs2[action2] - q1)

            state_vec1, state1, qs1, action1 = state_vec2, state2, qs2, action2

            if self.is_game_over(snake):
                self.env.score = 0
                break

            if self.game:
                plot_game(self.game, self.env, snake)
                time.sleep(delay)


class QLearning(TemporalDifference):
    def __init__(self, game, env, epsilon=0.1, alpha=0.05):
        super().__init__(game, env, epsilon, alpha)

    def run_episode(self, snake, train=True, delay=1):
        state_vec1, state1, qs1 = self.state_actions(snake)

        for step in itertools.count(0):  # while True
            action = self.epsilon_greedy_action(qs1, state1, step, train)
            if train:
                reward = self.env.reward(snake, state1, state_vec1, action)

            snake.move(action)
            state_vec2, state2, qs2 = self.state_actions(snake)

            if train:
                q1 = qs1[action]
                q2_max = max(qs2.values(), default=0)
                qs1[action] = q1 + self.alpha * (reward + q2_max - q1)

            state_vec1, state1, qs1 = state_vec2, state2, qs2

            if self.is_game_over(snake):
                self.env.score = 0
                break

            if self.game:
                plot_game(self.game, self.env, snake)
                time.sleep(delay)


def debug(game, env):
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
    while True:
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

        # Game Over conditions
        if not env.check_borders(snake) or len(snake.body) == env.x * env.y:
            game_over()

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
