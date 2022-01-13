import pygame
import argparse
from argparse import RawTextHelpFormatter

import control
from models import Environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--train",
        dest="train",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Train and save Q.pkl in current directory,\
load it when not `train`. \nTo stop training press `Ctrl-C`.",
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
        help="Controls snake speed in visual mode",
    )
    parser.add_argument(
        "--brick",
        dest="brick",
        default=30,
        type=int,
        help="Size of a grid cell in pixels",
    )
    parser.add_argument(
        "--x", dest="x", default=4, type=int, help="Frame `x` size in bricks"
    )
    parser.add_argument(
        "--y", dest="y", default=4, type=int, help="Frame `y` size in bricks"
    )
    parser.add_argument(
        "--grow",
        dest="grow",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to grow snake on eating a target",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Move snake by7 hand and see it's state",
    )
    parser.add_argument(
        "--algo",
        default="sarsa",
        const="sarsa",
        nargs="?",
        choices=["mc", "sarsa", "ql"],
        help="algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "--epsilon",
        dest="epsilon",
        default=0.1,
        type=float,
        help="Exploration strength",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        default=0.05,
        type=float,
        help="temporal difference step size",
    )

    args = parser.parse_args()
    if not args.train:
        args.visual = True

    env = Environment(args.x, args.y, args.brick, args.grow)

    game = None
    if args.visual:
        # Initialize game window
        pygame.display.set_caption("Snake")
        game = pygame.display.set_mode(
            (args.x * args.brick, args.y * args.brick)
        )

    if args.debug:
        control.debug(game, env)
    else:
        if args.algo == "mc":
            alg = control.MonteCarlo(game, env, epsilon=args.epsilon)
        elif args.algo == "sarsa":
            alg = control.Sarsa(
                game, env, epsilon=args.epsilon, alpha=args.alpha
            )
        elif args.algo == "ql":
            alg = control.QLearning(
                game,
                env,
                epsilon=args.epsilon,
                alpha=args.alpha,
            )
        else:
            raise NotImplementedError(args.algo)

        alg.run(train=args.train, delay=args.delay)
