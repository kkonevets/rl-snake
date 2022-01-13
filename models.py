import sys
import random
import numpy as np
import pygame


class Color:
    "Colors (R, G, B)"
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)


class Snake:
    def __init__(self, env):
        self.env = env

        # head should differ from food
        self.head = env.gen_point()
        while self.head == env.food_pos:
            self.head = env.gen_point()

        self.body = [list(self.head)]
        # get random direction: LEFT, RIGHT, UP, DOWN
        self.direction = random.randrange(pygame.K_RIGHT, pygame.K_UP + 1)

    def food_dist(self):
        "Return L1 distance to a food point"
        dx = abs(self.head[0] - self.env.food_pos[0])
        dy = abs(self.head[1] - self.env.food_pos[1])
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

    def move(self, key):
        if not self.move_pos(self.head, key):
            return

        self.direction = key
        env = self.env

        # Snake body growing mechanism
        self.body.insert(0, list(self.head))
        eat = self.head == env.food_pos
        if not env.grow or not eat:
            self.body.pop()

        if eat:
            env.score += 1
            env.food_pos = env.gen_point()
            while (
                env.food_pos in self.body and len(self.body) != env.x * env.y
            ):
                env.food_pos = env.gen_point()


class Environment:
    def __init__(self, x=10, y=10, brick=20, grow=False):
        # Window size
        self.x, self.y, self.brick, self.grow = x, y, brick, grow

        self.food_pos = self.gen_point()  # food position
        self.score = 0

        self.state_size = 12 + grow * 9

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
                f"[!] Had {check_errors[1]} errors\
                when initialising game, exiting..."
            )
            sys.exit(-1)

    def state(self, snake: Snake):
        """
        The state is a set of (12 + 9 cells around a head) bits
        as an integer number, representing:
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
            - Boolean indicator if cell is empty for each cell around a head
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
        state[11] = snake.food_dist() == 1

        if self.grow:
            # encode every grid cell state aroud a head
            ix = 12
            for i in (x - 1, x, x + 1):
                for j in (y - 1, y, y + 1):
                    if (0 <= i < self.x) and (0 <= j < self.y):
                        state[ix] = [i, j] in snake.body
                    ix += 1

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
