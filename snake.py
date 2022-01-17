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
        "--load",
        dest="load",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Continue training after loading action-value file",
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
        help="Move snake by hand and see it's state",
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
        help="Temporal difference step size",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        default=4,
        type=int,
        help="Number of steps for temporal difference method (n-step sarsa)",
    )

    args = parser.parse_args()
    env = Environment(args.x, args.y, args.brick, args.grow)

    game = None
    if not args.train:
        # Initialize game window
        pygame.display.set_caption("Snake")
        game = pygame.display.set_mode(
            (args.x * args.brick, args.y * args.brick)
        )

    if args.debug:
        control.debug(game, env)
    elif args.train:
        if args.algo == "mc":
            alg = control.MonteCarlo(game, env, epsilon=args.epsilon)
        elif args.algo == "sarsa":
            alg = control.Sarsa(
                game,
                env,
                n=args.steps,
                epsilon=args.epsilon,
                alpha=args.alpha,
            )
        elif args.algo == "ql":
            alg = control.QLearning(
                game,
                env,
                n=args.steps,
                epsilon=args.epsilon,
                alpha=args.alpha,
            )
        else:
            raise NotImplementedError(args.algo)

        alg.train(args.load)
    else:
        ctrl = control.Control(game, env, epsilon=args.epsilon)
        ctrl.follow(delay=args.delay)
