import pygame
import sys
import random
import numpy as np
from collections import defaultdict
import time
from operator import itemgetter
import pickle
from pprint import pprint

BRICK = 20  # size of unit snake element


class Color:
    "Colors (R, G, B)"
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)


class Snake:
    def __init__(self, head):
        self.head = head
        self.body = [head]
        # get random direction: LEFT, RIGHT, UP, DOWN
        self.direction = random.randrange(pygame.K_RIGHT, pygame.K_UP + 1)

    def l1(self, food_pos):
        "Return L1 distance to a food point"
        dist = abs(self.head[0] - food_pos[0]) + abs(
            self.head[1] - food_pos[1]
        )
        return int(dist / BRICK)

    def move(self, key, env):
        if sorted((key, self.direction)) in (
            [pygame.K_RIGHT, pygame.K_LEFT],
            [pygame.K_DOWN, pygame.K_UP],
        ):
            return

        self.direction = key
        if key == pygame.K_UP:
            self.head[1] -= BRICK
        elif key == pygame.K_LEFT:
            self.head[0] -= BRICK
        elif key == pygame.K_DOWN:
            self.head[1] += BRICK
        elif key == pygame.K_RIGHT:
            self.head[0] += BRICK

        # Snake body growing mechanism
        self.body.insert(0, list(self.head))
        if self.head == env.food_pos:
            env.score += 1
            env.food_pos = env.gen_point()
            while env.food_pos in self.body:
                env.food_pos = env.gen_point()
        else:
            self.body.pop()


def isKthBitSet(n, k) -> bool:
    return n & (1 << (k - 1)) != 0


class Environment:
    def __init__(self, frame_size_x=10 * BRICK, frame_size_y=10 * BRICK):
        # Window size
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y

        self.food_pos = self.gen_point()  # food position
        self.score = 0

        self.state_size = 12

        # Checks for errors encountered
        check_errors = pygame.init()
        # second number in tuple gives number of errors
        if check_errors[1] > 0:
            print(
                f"[!] Had {check_errors[1]} errors when initialising game, exiting..."
            )
            sys.exit(-1)
        else:
            print("[+] Game successfully initialised")

    def state(self, snake: Snake):
        """
        The state is a set of (12) bits, representing:
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
        """

        state = np.zeros(self.state_size, dtype="I")
        x, y = snake.head

        if snake.direction == pygame.K_LEFT:
            if x < BRICK:
                state[0] = 1
            if y > self.frame_size_y - 2 * BRICK:
                state[1] = 1
            if y < BRICK:
                state[2] = 1
        elif snake.direction == pygame.K_RIGHT:
            if x > self.frame_size_x - 2 * BRICK:
                state[0] = 1
            if y < BRICK:
                state[1] = 1
            if y > self.frame_size_y - 2 * BRICK:
                state[2] = 1
        elif snake.direction == pygame.K_UP:
            if y < BRICK:
                state[0] = 1
            if x < BRICK:
                state[1] = 1
            if x > self.frame_size_x - 2 * BRICK:
                state[2] = 1
        elif snake.direction == pygame.K_DOWN:
            if y > self.frame_size_y - 2 * BRICK:
                state[0] = 1
            if x > self.frame_size_x - 2 * BRICK:
                state[1] = 1
            if x < BRICK:
                state[2] = 1

        state[3] = snake.direction == pygame.K_LEFT
        state[4] = snake.direction == pygame.K_RIGHT
        state[5] = snake.direction == pygame.K_UP
        state[6] = snake.direction == pygame.K_DOWN
        state[7] = self.food_pos[0] < x
        state[8] = self.food_pos[0] > x
        state[9] = self.food_pos[1] < y
        state[10] = self.food_pos[1] > y
        state[11] = snake.l1(self.food_pos) == 1

        shash = 0
        for i, b in enumerate(state):
            if b:
                shash |= 1 << i  # set i-th bit

        # print("{0:b}".format(shash)[::-1], state, snake.direction)

        return shash

    def gen_point(self):
        "Generate random point"
        return [
            random.randrange(1, (self.frame_size_x // BRICK)) * BRICK,
            random.randrange(1, (self.frame_size_y // BRICK)) * BRICK,
        ]

    def check_borders(self, snake: Snake):
        if (
            snake.head[0] < 0
            or snake.head[0] > self.frame_size_x - BRICK
            or snake.head[1] < 0
            or snake.head[1] > self.frame_size_y - BRICK
            or snake.head in snake.body[1:]  # Snake self intersection
        ):
            return False
        return True

    def direction(self, state):
        if isKthBitSet(state, 4):
            return pygame.K_LEFT
        elif isKthBitSet(state, 5):
            return pygame.K_RIGHT
        elif isKthBitSet(state, 6):
            return pygame.K_UP
        elif isKthBitSet(state, 7):
            return pygame.K_DOWN
        else:
            raise NotImplemented

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
            raise NotImplemented

    def reward(self, state, action):
        direction = self.direction(state)
        dneigbs = Environment.direction_neigbs(direction)

        if isKthBitSet(state, 1) and action == direction:  # danger ahead
            return -1
        elif isKthBitSet(state, 2) and action == dneigbs[0]:  # danger left
            return -1
        elif isKthBitSet(state, 3) and action == dneigbs[1]:  # danger right
            return -1
        elif isKthBitSet(state, 12):  # one step to a food
            if isKthBitSet(state, 11) and action == pygame.K_DOWN:
                return 1  # food down
            elif isKthBitSet(state, 10) and action == pygame.K_UP:
                return 1  # food up
            elif isKthBitSet(state, 8) and action == pygame.K_LEFT:
                return 1  # food left
            elif isKthBitSet(state, 9) and action == pygame.K_RIGHT:
                return 1  # food right

        return 0


def game_over():
    "Game Over"
    pygame.quit()
    sys.exit()


def show_score(game, score):
    score_font = pygame.font.SysFont("consolas", 20)
    score_surface = score_font.render(
        "Score : " + str(score), True, Color.WHITE
    )
    score_rect = score_surface.get_rect()
    frame_size_x, frame_size_y = game.get_size()
    score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.2)
    game.blit(score_surface, score_rect)


def plot_game(game, env, snake):
    # GFX
    game.fill(Color.BLACK)

    pygame.draw.rect(
        game,
        Color.WHITE,
        pygame.Rect(snake.head[0], snake.head[1], BRICK, BRICK),
    )
    for pos in snake.body[1:]:
        pygame.draw.rect(
            game, Color.GREEN, pygame.Rect(pos[0], pos[1], BRICK, BRICK)
        )

    # Snake food
    pygame.draw.rect(
        game,
        Color.RED,
        pygame.Rect(env.food_pos[0], env.food_pos[1], BRICK, BRICK),
    )

    show_score(game, env.score)
    # Refresh game screen
    pygame.display.update()


def rand_action(direction):
    while True:
        action = random.randrange(pygame.K_RIGHT, pygame.K_UP + 1)
        if direction == pygame.K_DOWN and action != pygame.K_UP:
            return action
        elif direction == pygame.K_UP and action != pygame.K_DOWN:
            return action
        elif direction == pygame.K_RIGHT and action != pygame.K_LEFT:
            return action
        elif direction == pygame.K_LEFT and action != pygame.K_RIGHT:
            return action


def load_Q():
    with open("Q.pkl", "rb") as f:
        Q = pickle.load(f)
        print("size:", len(Q))
        pprint(Q)


def MonteCarloES(game, env, visual=True, delay=1):
    pi = {}
    Q = {}
    Returns = defaultdict(lambda: [0, 0])

    def learning_loop():
        snake = Snake(env.gen_point())
        if visual:
            plot_game(game, env, snake)

        episode = []
        step_i = 0
        while True:
            state = env.state(snake)
            if step_i == 0:
                action = rand_action(snake.direction)
            else:
                action = pi.get(state, rand_action(snake.direction))

            step_i += 1
            snake.move(action, env)

            episode.append((state, action, env.reward(state, action)))

            # Episode Over conditions
            if not env.check_borders(snake):
                env.score = 0
                break

            if visual:
                plot_game(game, env, snake)
                time.sleep(delay)

        # print(Q)

        G = 0
        for i, (state, action, reward) in enumerate(episode[::-1]):
            G = G + reward
            found = filter(
                lambda sa: state == sa[0] and action == sa[1],
                episode[: -i - 1],
            )
            if not list(found):
                ret = Returns[(state, action)]
                ret[0] += G
                ret[1] += 1
                Q[(state, action)] = ret[0] / ret[1]
                pi[state] = max(
                    [(a, q) for (s, a), q in Q.items() if s == state],
                    key=itemgetter(1),
                )[0]

        # print(G)

    try:
        while True:
            learning_loop()
    except:
        with open("Q.pkl", "wb") as f:
            pickle.dump(Q, f)


def by_hand(game, env):
    key = None
    snake = Snake(env.gen_point())

    print("INIT")
    print("{0:b}".format(env.state(snake))[::-1])

    plot_game(game, env, snake)

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

        state = env.state(snake)
        print(
            env.reward(state, pygame.K_LEFT),
            env.reward(state, pygame.K_RIGHT),
            env.reward(state, pygame.K_UP),
            env.reward(state, pygame.K_DOWN),
        )


if __name__ == "__main__":
    env = Environment(10 * BRICK, 10 * BRICK)

    # Initialize game window
    pygame.display.set_caption("Snake")
    game = pygame.display.set_mode((env.frame_size_x, env.frame_size_y))

    # MonteCarloES(game, env, 0, 1)
    # by_hand(game, env)
    load_Q()
