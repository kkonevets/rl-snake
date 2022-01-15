# rl-snake
Some reinforcement learning algorithms to play snake game taken from Sutton's book

## Usage

```bash
$ python snake.py --help
```

## Algorithms

### Tabular

1. Monte Carlo
2. SARSA
3. Q-learning

### Non-growing snake
```python
python snake.py --train --x=5 --y=5 --algo=sarsa
```

It is enough to train non-growing snake on a 5x5 grid to be able to use it on arbitrary large grid. \
The more you train the better it becomes. Test it:
```python
python snake.py --visual --x=10 --y=10 --algo=sarsa
```

### Growing snake
```python
python snake.py --train --x=5 --y=5 --grow --algo=sarsa
```

Additional 9 boolean indicators are added to a snake's state for each cell around a head, indicating if cell belongs to a snake. So snake is myopic in terms of what it can see. Adding all grid cells is not tractable due to enourmouse ammount of possible states.

![](etc/head-state.png)

```python
python snake.py --visual --x=5 --y=5 --grow --delay=0.3 --algo=sarsa
```

### Compare

Left to right: Monte Carlo, SARSA, QLearning

<img src="etc/mc-single.gif" title="Monte Carlo (single)" width="200"/> <img src="etc/sarsa-single.gif" title="SARSA (single)" width="200"/> <img src="etc/ql-single.gif" title="QLearning (single)" width="200"/>

<img src="etc/mc-growing.gif" title="Monte Carlo (growing)"/> <img src="etc/sarsa-growing.gif" title="SARSA (growing)"/> <img src="etc/ql-growing.gif" title="QLearning (growing)"/>


### Parameters

|             | Monte Carlo | SARSA | QLearning |
|-------------|-------------|-------|-----------|
| **SINGLE**  |             |       |           |
| epsilon     | 0.5         | 0.3   | 0.5       |
| alpha       | ---         | 0.05  | 0.005     |
| episodes    | 1080k       | 2044k | 1030k     |
|-------------|-------------|-------|-----------|
| **GROWING** |             |       |           |
| epsilon     | 0.05        | 0.05  | 0.1       |
| alpha       | ---         | 0.05  | 0.0005    |
| episodes    | 4152k       | 1559k | 2023k     |
