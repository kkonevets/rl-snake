import pygame
import sys
import random


class Color:
    "Colors (R, G, B)"
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)


class Snake:
    head = [100, 50]
    body = [head]
    direction = pygame.K_RIGHT

    def __init__(self, key):
        self.direction = key

    def move(self, key, env):
        if (key == pygame.K_UP and self.direction != pygame.K_DOWN) or (
            key == pygame.K_DOWN and self.direction == pygame.K_UP
        ):
            self.direction = pygame.K_UP
            self.head[1] -= 10
        elif (key == pygame.K_DOWN and self.direction != pygame.K_UP) or (
            key == pygame.K_UP and self.direction == pygame.K_DOWN
        ):
            self.direction = pygame.K_DOWN
            self.head[1] += 10
        elif (key == pygame.K_LEFT and self.direction != pygame.K_RIGHT) or (
            key == pygame.K_RIGHT and self.direction == pygame.K_LEFT
        ):
            self.direction = pygame.K_LEFT
            self.head[0] -= 10
        elif (key == pygame.K_RIGHT and self.direction != pygame.K_LEFT) or (
            key == pygame.K_LEFT and self.direction == pygame.K_RIGHT
        ):
            self.direction = pygame.K_RIGHT
            self.head[0] += 10

        # Snake body growing mechanism
        snake.body.insert(0, list(snake.head))
        if snake.head == env.food_pos:
            env.score += 1
            env.food_pos = env.gen_food()
        else:
            snake.body.pop()


class Environment:
    # Window size
    frame_size_x = 0
    frame_size_y = 0

    food_pos = [0, 0]  # food position
    score = 0

    def __init__(self, frame_size_x=350, frame_size_y=350):
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y

        self.food_pos = self.gen_food()

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

    def gen_food(self):
        "Generate food"
        return [
            random.randrange(1, (self.frame_size_x // 10)) * 10,
            random.randrange(1, (self.frame_size_y // 10)) * 10,
        ]

    def check_borders(self, snake: Snake):
        if (
            snake.head[0] < 0
            or snake.head[0] > self.frame_size_x - 10
            or snake.head[1] < 0
            or snake.head[1] > self.frame_size_y - 10
            or snake.head in snake.body[1:]  # Snake self intersection
        ):
            return False
        return True


def game_over(game):
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
    score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.05)
    game.blit(score_surface, score_rect)


if __name__ == "__main__":
    env = Environment(350, 350)

    # Initialise game window
    pygame.display.set_caption("Snake Eater")
    game = pygame.display.set_mode((env.frame_size_x, env.frame_size_y))

    # frames per second controller
    fps_controller = pygame.time.Clock()

    key = pygame.K_RIGHT
    snake = Snake(key)

    # Main logic
    while True:
        for event in pygame.event.get():
            # Whenever a key is pressed down
            if event.type == pygame.KEYDOWN:
                key = event.key

        snake.move(key, env)

        # Game Over conditions
        if not env.check_borders(snake):
            game_over(game)

        # GFX
        game.fill(Color.BLACK)
        for pos in snake.body:
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
            pygame.draw.rect(
                game, Color.GREEN, pygame.Rect(pos[0], pos[1], 10, 10), 1
            )

        # Snake food
        pygame.draw.rect(
            game,
            Color.RED,
            pygame.Rect(env.food_pos[0], env.food_pos[1], 10, 10),
        )

        show_score(game, env.score)
        # Refresh game screen
        pygame.display.update()
        # Refresh rate
        fps_controller.tick(10)
