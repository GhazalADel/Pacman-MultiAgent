# Pacman Multi-Agent AI

**Course**: Fundamentals and Applications of Artificial Intelligence — Spring 1404  
**University**: Amirkabir University of Technology  

## Overview

Designing AI agents for classic Pacman with ghosts, using adversarial and probabilistic search algorithms.

## Implemented Agents

| Agent | Algorithm |
|-------|-----------|
| `ReflexAgent` | Improved evaluation function using food/ghost distances |
| `MinimaxAgent` | Minimax search to arbitrary depth with multiple ghosts |
| `AlphaBetaAgent` | Minimax with alpha-beta pruning |
| `ExpectimaxAgent` | Expectimax for probabilistic ghost modeling |
| `betterEvaluationFunction` | Improved state evaluation for use with Expectimax |

## Modified Files

- `multiAgents.py` — all agent and evaluation function implementations

## Running

```bash
# Reflex agent
python pacman.py -p ReflexAgent -l testClassic

# Minimax
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

# Alpha-Beta
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

# Expectimax
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

# Run autograder
python autograder.py
python autograder.py -q q2 --no-graphics
```

## Base Project

Based on the [UC Berkeley CS188 Multi-Agent Pacman Project](http://ai.berkeley.edu/multiagent.html).
