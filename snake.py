import pygame
import sys
import random
import numpy as np

BRICK = 15  # size of unit snake element


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
        self.direction = -1

    def move(self, key, env):
        if sorted((key, self.direction)) in (
            [pygame.K_DOWN, pygame.K_UP],
            [pygame.K_RIGHT, pygame.K_LEFT],
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
        snake.body.insert(0, list(snake.head))
        if snake.head == env.food_pos:
            env.score += 1
            env.food_pos = env.gen_food()
            while env.food_pos in self.body:
                env.food_pos = env.gen_food()
        else:
            snake.body.pop()


class Environment:
    def __init__(self, frame_size_x=350, frame_size_y=350):
        # Window size
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y

        self.food_pos = self.gen_food()  # food position
        self.score = 0

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

    def get_state(self, snake: Snake):
        """
        Return the state.
        The state is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the left
            - Danger 1 OR 2 steps on the right
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side
        """
        state = np.zeros(11)
        x, y = snake.head

        if snake.direction == pygame.K_LEFT:
            if x < 2 * BRICK:
                state[0] = 1
            if y > self.frame_size_y - 3 * BRICK:
                state[1] = 1
            if y < 2 * BRICK:
                state[2] = 1
        elif snake.direction == pygame.K_RIGHT:
            if x > self.frame_size_x - 3 * BRICK:
                state[0] = 1
            if y < 2 * BRICK:
                state[1] = 1
            if y > self.frame_size_y - 3 * BRICK:
                state[2] = 1
        elif snake.direction == pygame.K_UP:
            if y < 2 * BRICK:
                state[0] = 1
            if x < 2 * BRICK:
                state[1] = 1
            if x > self.frame_size_x - 3 * BRICK:
                state[2] = 1
        elif snake.direction == pygame.K_DOWN:
            if y > self.frame_size_y - 3 * BRICK:
                state[0] = 1
            if x > self.frame_size_x - 3 * BRICK:
                state[1] = 1
            if x < 2 * BRICK:
                state[2] = 1

        state[3] = snake.direction == pygame.K_LEFT
        state[4] = snake.direction == pygame.K_RIGHT
        state[5] = snake.direction == pygame.K_UP
        state[6] = snake.direction == pygame.K_DOWN
        state[7] = self.food_pos[0] < x
        state[8] = self.food_pos[0] > x
        state[9] = self.food_pos[1] < y
        state[10] = self.food_pos[1] > y

        def set_bit(value, bit):
            return value | (1 << bit)

        def state_hash(state):
            state_hash = 0
            for i, b in enumerate(state):
                if b:
                    state_hash = set_bit(state_hash, i)
            return state_hash

        return state, state_hash(state)

    def gen_food(self):
        "Generate food"
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
    score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.1)
    game.blit(score_surface, score_rect)


def plot_game(game, env, snake):
    # GFX
    game.fill(Color.BLACK)
    for pos in snake.body:
        # .draw.rect(play_surface, color, xy-coordinate)
        # xy-coordinate -> .Rect(x, y, size_x, size_y)
        pygame.draw.rect(
            game, Color.GREEN, pygame.Rect(pos[0], pos[1], BRICK, BRICK), 1
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


if __name__ == "__main__":
    env = Environment(20 * BRICK, 20 * BRICK)

    # Initialise game window
    pygame.display.set_caption("Snake")
    game = pygame.display.set_mode((env.frame_size_x, env.frame_size_y))

    key = None
    snake = Snake([5 * BRICK, 5 * BRICK])

    plot_game(game, env, snake)

    # Main logic
    while True:
        event = pygame.event.wait()
        # Whenever a key is pressed down
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                game_over()
            else:
                key = event.key
        else:
            continue

        snake.move(key, env)

        # Game Over conditions
        if not env.check_borders(snake):
            game_over()

        plot_game(game, env, snake)
