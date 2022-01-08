import pygame
import sys
import random
import numpy as np
from collections import defaultdict
import time
import pickle
import itertools
import argparse


class Color:
    "Colors (R, G, B)"
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)


class Snake:
    def __init__(self, head, grow=False):
        self.head, self.body, self.grow = head, [head], grow
        # get random direction: LEFT, RIGHT, UP, DOWN
        self.direction = random.randrange(pygame.K_RIGHT, pygame.K_UP + 1)

    def l1(self, food_pos):
        "Return L1 distance to a food point"
        dx = abs(self.head[0] - food_pos[0])
        dy = abs(self.head[1] - food_pos[1])
        return dx + dy

    def move_pos(self, pos, key):
        if sorted((key, self.direction)) in (
            [pygame.K_RIGHT, pygame.K_LEFT],
            [pygame.K_DOWN, pygame.K_UP],
        ):
            return False  # can't move in opposite direction

        if key == pygame.K_UP:
            pos[1] -= 1
        elif key == pygame.K_LEFT:
            pos[0] -= 1
        elif key == pygame.K_DOWN:
            pos[1] += 1
        elif key == pygame.K_RIGHT:
            pos[0] += 1
        else:
            raise NotImplementedError(key)

        return True

    def move(self, key, env):
        if not self.move_pos(self.head, key):
            return

        self.direction = key

        # Snake body growing mechanism
        if self.grow:
            self.body.insert(0, list(self.head))
        if self.head == env.food_pos:
            env.score += 1
            env.food_pos = env.gen_point()
            while env.food_pos in self.body:
                env.food_pos = env.gen_point()
        elif self.grow:
            self.body.pop()


class Environment:
    def __init__(self, x=10, y=10, brick=20, grow=False):
        # Window size
        self.x, self.y, self.brick, self.grow = x, y, brick, grow

        self.food_pos = self.gen_point()  # food position
        self.score = 0

        self.state_size = 12 + grow * (self.x * self.y)

        self._direction_map = {
            0b1000: pygame.K_LEFT,
            0b10000: pygame.K_RIGHT,
            0b100000: pygame.K_UP,
            0b1000000: pygame.K_DOWN,
        }

        # Checks for errors encountered
        check_errors = pygame.init()
        # second number in tuple gives number of errors
        if check_errors[1] > 0:
            print(
                f"[!] Had {check_errors[1]} errors when initialising game, exiting..."
            )
            sys.exit(-1)

    def state(self, snake: Snake):
        """
        The state is a set of (12 + number of cells) bits as an integer number,
        representing:
            - Danger one step ahead
            - Danger on the left
            - Danger on the right
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side
            - The food is in one step distance
            - Boolean indicator if cell is empty for each cell
        """
        state = [0] * self.state_size
        x, y = snake.head

        if snake.direction == pygame.K_LEFT:
            state[0] = x < 1
            state[1] = y > self.y - 2
            state[2] = y < 1
        elif snake.direction == pygame.K_RIGHT:
            state[0] = x > self.x - 2
            state[1] = y < 1
            state[2] = y > self.y - 2
        elif snake.direction == pygame.K_UP:
            state[0] = y < 1
            state[1] = x < 1
            state[2] = x > self.x - 2
        elif snake.direction == pygame.K_DOWN:
            state[0] = y > self.y - 2
            state[1] = x > self.x - 2
            state[2] = x < 1
        else:
            raise NotImplementedError(snake.direction)

        state[3] = snake.direction == pygame.K_LEFT
        state[4] = snake.direction == pygame.K_RIGHT
        state[5] = snake.direction == pygame.K_UP
        state[6] = snake.direction == pygame.K_DOWN
        state[7] = self.food_pos[0] < x
        state[8] = self.food_pos[0] > x
        state[9] = self.food_pos[1] < y
        state[10] = self.food_pos[1] > y
        state[11] = snake.l1(self.food_pos) == 1

        if self.grow:
            # encode every grid cell state
            for i, pos in enumerate(snake.body):
                state[12 + pos[0] + pos[1] * self.x] = True

        shash = 0
        for i, b in enumerate(state):
            if b:
                shash |= 1 << i  # set i-th bit

        # print("{0:b}".format(shash)[::-1], state, snake.direction)
        return state, shash

    def gen_point(self):
        "Generate random point"
        return [random.randrange(0, self.x), random.randrange(0, self.y)]

    def check_borders(self, snake: Snake):
        return (
            (0 <= snake.head[0] < self.x)
            and (0 <= snake.head[1] < self.y)
            and snake.head not in snake.body[1:]  # Snake self intersection
        )

    def direction(self, state):
        return self._direction_map[state & 0b1111000]

    def direction_neigbs(direction):
        if direction == pygame.K_UP:
            return pygame.K_LEFT, pygame.K_RIGHT
        elif direction == pygame.K_RIGHT:
            return pygame.K_UP, pygame.K_DOWN
        elif direction == pygame.K_DOWN:
            return pygame.K_RIGHT, pygame.K_LEFT
        elif direction == pygame.K_LEFT:
            return pygame.K_DOWN, pygame.K_UP
        else:
            raise NotImplementedError(direction)

    def reward(self, snake, state, state_vec, action):
        next_pos = [snake.head[0], snake.head[1]]  # copy head
        if not snake.move_pos(next_pos, action):  # move copied head
            return 0

        direction = self.direction(state)
        l, r = Environment.direction_neigbs(direction)

        if state_vec[0] and action == direction:  # danger ahead
            return -1
        elif state_vec[1] and action == l:  # danger left
            return -1
        elif state_vec[2] and action == r:  # danger right
            return -1
        elif next_pos in snake.body[1:-1]:  # self intersection
            return -1
        elif state_vec[11]:  # one step to a food
            if state_vec[10] and action == pygame.K_DOWN:
                return 1  # food down
            elif state_vec[9] and action == pygame.K_UP:
                return 1  # food up
            elif state_vec[7] and action == pygame.K_LEFT:
                return 1  # food left
            elif state_vec[8] and action == pygame.K_RIGHT:
                return 1  # food right

        return 0

    def available_actions(direction, shuffle=False):
        l, r = Environment.direction_neigbs(direction)
        actions = [direction, l, r]
        if shuffle:
            np.random.shuffle(actions)
        return actions


def show_score(game, score):
    score_font = pygame.font.SysFont("consolas", 17)
    score_surface = score_font.render("Score : %d" % score, True, Color.WHITE)
    score_rect = score_surface.get_rect()
    frame_size_x, frame_size_y = game.get_size()
    score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.2)
    game.blit(score_surface, score_rect)


def plot_game(game, env, snake):
    game.fill(Color.BLACK)  # GFX

    def my_rect(xy):
        return pygame.Rect(xy[0] * brick, xy[1] * brick, brick, brick)

    brick = env.brick
    pygame.draw.rect(game, Color.WHITE, my_rect(snake.head))
    for pos in snake.body[1:]:
        pygame.draw.rect(game, Color.GREEN, my_rect(pos))

    # Snake food
    pygame.draw.rect(game, Color.RED, my_rect(env.food_pos))

    show_score(game, env.score)
    pygame.display.update()  # Refresh game screen


def epsilon_soft_distribution(n, eps):
    p = np.full(n, eps / n)
    p[0] = 1 - eps + eps / n
    return p


class MonteCarloEpsilonGreedy:
    def __init__(self, game, env, eps=0.1):
        self.game = game
        self.env = env
        self.eps = eps
        self.pi, self.Q = {}, {}
        self.Returns = defaultdict(lambda: [0, 0])
        self.p_soft = epsilon_soft_distribution(3, eps)
        # prechoose big number of random numbers 0<=x<3 (optimization)
        self.choices = np.random.choice(
            self.p_soft.size,
            size=1000 * (env.x + env.y),
            p=self.p_soft,
        )

    def run_episode(self, train=True, visual=True, delay=1):
        snake = Snake(self.env.gen_point(), self.env.grow)
        if visual:
            plot_game(self.game, self.env, snake)

        episode = []
        for step in itertools.count():  # while True
            state_vec, state = self.env.state(snake)
            direction = env.direction(state)

            if step == 0 and train:  # exploring start
                actions = Environment.available_actions(direction, True)
            else:
                actions = self.pi.get(
                    state, Environment.available_actions(direction, True)
                )

            if train:
                if step > self.choices.size - 1:
                    raise RuntimeError(
                        "too many random walk steps: ", self.choices.size
                    )
                ix = self.choices[step]  # explore in train mode
            else:
                ix = 0  # always select the best action in test mode
            action = actions[ix]

            snake.move(action, self.env)

            if train:
                reward = self.env.reward(snake, state, state_vec, action)
                episode.append((state, action, reward))

            # Episode Over conditions
            if not self.env.check_borders(snake):
                self.env.score = 0
                break

            if visual:
                plot_game(self.game, self.env, snake)
                time.sleep(delay)

        if train:
            self.backtrace(episode)

    def greedy_pi(self, q, state):
        a_star = max(q, key=q.get)
        actions = Environment.available_actions(env.direction(state))
        res = [a_star]
        res.extend(a for a in actions if a != a_star)
        self.pi[state] = res

    def backtrace(self, episode):
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
                q = self.Q.setdefault(state, {})
                q[action] = ret[0] / ret[1]
                self.greedy_pi(q, state)

    def run(self, train=True, visual=True, delay=1):
        if not train:
            with open("Q.pkl", "rb") as f:
                self.Q = pickle.load(f)
                for state, q in self.Q.items():
                    self.greedy_pi(q, state)
                self.print_stat(0)

        try:
            for epi in itertools.count():  # while True
                self.run_episode(train, visual, delay)
                if train and epi % 1000 == 0:
                    self.print_stat(epi)

        except KeyboardInterrupt:
            print("\r")
            if train:
                with open("Q.pkl", "wb") as f:
                    pickle.dump(self.Q, f)

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


def debug(game, env):
    key = None
    snake = Snake(env.gen_point(), env.grow)

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

        snake.move(key, env)

        # Game Over conditions
        if not env.check_borders(snake):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        dest="train",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="train and save Q.pkl in current directory, load it when not `train`",
    )
    parser.add_argument(
        "--visual",
        dest="visual",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--delay",
        dest="delay",
        default=0.1,
        type=float,
        help="controls snake speed in visual mode",
    )
    parser.add_argument(
        "--brick",
        dest="brick",
        default=30,
        type=int,
        help="size of a grid cell in pixels",
    )
    parser.add_argument(
        "--x", dest="x", default=4, type=int, help="frame `x` size in bricks"
    )
    parser.add_argument(
        "--y", dest="y", default=4, type=int, help="frame `y` size in bricks"
    )
    parser.add_argument(
        "--grow",
        dest="grow",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="whether to grow snake on eating a target",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="move snake by7 hand and see it's state",
    )

    args = parser.parse_args()

    env = Environment(args.x, args.y, args.brick, args.grow)

    # Initialize game window
    pygame.display.set_caption("Snake")
    game = pygame.display.set_mode((args.x * args.brick, args.y * args.brick))

    if args.debug:
        debug(game, env)
    else:
        alg = MonteCarloEpsilonGreedy(game, env, eps=0.1)
        alg.run(train=args.train, visual=args.visual, delay=args.delay)
