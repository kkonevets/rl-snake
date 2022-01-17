# rl-snake
Some reinforcement learning algorithms to play snake game taken from Sutton's book

## Usage

```bash
$ python snake.py --help
```

## Algorithms

### Tabular

1. Monte Carlo
2. one-step Q-learning
3. n-step SARSA

To stop training press `Ctrl-C` and `Q.pkl` file will be saved in current directory. Then it can be used to continue training or to follow learned policy with visualization.

### Non-growing snake
```python
python snake.py --train --x=5 --y=5 --algo=mc
```

It is enough to train non-growing snake on a 5x5 grid to be able to use it on arbitrary large grid. \
The more you train the better it becomes. Test it:
```python
python snake.py --x=10 --y=10
```

### Growing snake
```python
python snake.py --train --x=5 --y=5 --grow --algo=sarsa --step=4
```

Additional 9 boolean indicators are added to a snake's state for each cell around a head, indicating if cell belongs to a snake. So snake is myopic in terms of what it can see. Adding all grid cells is not tractable due to enourmouse ammount of possible states.

![](etc/head-state.png)

```python
python snake.py --x=5 --y=5 --grow --delay=0.3
```

### Compare

Left to right: Monte Carlo, 1-step SARSA, 4-step SARSA, 1-step Q-learning

<img src="etc/mc-single.gif" title="Monte Carlo (single)" width="200"/> <img src="etc/1-sarsa-single.gif" title="1-step SARSA (single)" width="200"/> <img src="etc/4-sarsa-single.gif" title="4-step SARSA (single)" width="200"/> <img src="etc/1-ql-single.gif" title="1-step Q-learning (single)" width="200"/>

<img src="etc/mc-growing.gif" title="Monte Carlo (growing)" width="200"/> <img src="etc/1-sarsa-growing.gif" title="1-step SARSA (growing)" width="200"/> <img src="etc/4-sarsa-growing.gif" title="4-step SARSA (growing)" width="200"/> <img src="etc/1-ql-growing.gif" title="1-step Q-learning (growing)" width="200"/>

### Parameters

|             | Monte Carlo | 1-SARSA | 4-SARSA | Q-learning |
|-------------|-------------|---------|---------|------------|
| **SINGLE**  |             |         |         |            |
| epsilon     | 0.5         | 0.3     | 0.3     | 0.5        |
| alpha       | ---         | 0.05    | 0.05    | 0.005      |
| episodes    | 1080k       | 2044k   | 253k    | 1030k      |
|-------------|-------------|---------|---------|------------|
| **GROWING** |             |         |         |            |
| epsilon     | 0.05        | 0.05    | 0.05    | 0.1        |
| alpha       | ---         | 0.05    | 0.05    | 0.0005     |
| episodes    | 4152k       | 1559k   | 1559k   | 2023k      |
