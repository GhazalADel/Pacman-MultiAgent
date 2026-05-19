# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print(newFood)

        score = successorGameState.getScore()
        if newFood:
            foodList = newFood.asList()
            minDistance = math.inf
            for i in range(len(foodList)):
                food = foodList[i]
                distance = manhattanDistance(food, newPos)
                if distance < minDistance:
                    minDistance = distance

            if minDistance == 0:
                score += 2
            else:
                score += 1.0 / minDistance

        if newGhostStates:
            for i in range(len(newGhostStates)):
                ghost = newGhostStates[i]
                ghostDistance = manhattanDistance(ghost.getPosition(), newPos)

                scaredTime = newScaredTimes[i]
                if scaredTime > 0:
                    score += (2.0 * scaredTime) / (ghostDistance + 1)
                else:
                    if ghostDistance > 5:
                        score += 1.0 / (ghostDistance + 1)
                    else:
                        score -= 20

        if action == Directions.STOP:
            score -= 3

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getNextAgentAndDepth(self, agentIndex, depth, state):
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth+=1
        return nextAgent, nextDepth

    def minimax(self, agentIndex, state, depth=0):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        values = []
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            nextAgent, nextDepth = self.getNextAgentAndDepth(agentIndex, depth, state)
            value = self.minimax(nextAgent, successor, nextDepth)
            values.append(value)

        if agentIndex == 0:
            return max(values)
        else:
            return min(values)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # pacman
        highest_score = -math.inf
        best_action = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            res = self.minimax(1, successor)
            if res > highest_score:
                highest_score = res
                best_action = action

        return best_action
class AlphaBetaAgent(MultiAgentSearchAgent):
    def getNextAgentAndDepth(self, agentIndex, depth, state):
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth += 1
        return nextAgent, nextDepth

    def alphaBeta(self, agentIndex, state, alpha, beta, depth=0):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        values = []
        for action in state.getLegalActions(agentIndex):
            successor   = state.generateSuccessor(agentIndex, action)
            nextAgent, nextDepth = self.getNextAgentAndDepth(agentIndex, depth, state)
            value = self.alphaBeta(nextAgent, successor, alpha, beta, nextDepth)
            values.append(value)

            if agentIndex == 0:
                if value > beta:
                    return value
                alpha = max(alpha, value)
            else:
                if value < alpha:
                    return value
                beta = min(beta, value)

        if agentIndex == 0:
            return max(values)
        else:
            return min(values)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        highest_score = -math.inf
        best_action = None
        alpha = -math.inf
        beta = math.inf

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            res = self.alphaBeta(1, successor, alpha, beta)
            if res > highest_score:
                highest_score = res
                best_action = action
            alpha = max(alpha, res)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getNextAgentAndDepth(self, agentIndex, depth, state):
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth += 1
        return nextAgent, nextDepth

    def expectimax(self, agentIndex, state, depth=0):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        nextAgent, nextDepth = self.getNextAgentAndDepth(agentIndex, depth, state)
        legal = state.getLegalActions(agentIndex)

        if agentIndex == 0:
            _max_score = -math.inf
            for action in legal:
                successor = state.generateSuccessor(agentIndex, action)
                res = self.expectimax(nextAgent, successor, nextDepth)
                _max_score = max(_max_score, res)
            return _max_score

        else:
            total = 0
            for action in legal:
                successor = state.generateSuccessor(agentIndex, action)
                total += self.expectimax(nextAgent, successor, nextDepth)
            return total / len(legal)


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        highest_score = -math.inf
        best_action = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(1, successor)
            if score > highest_score:
                highest_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    if foods:
        foodList = foods.asList()
        minDistance = math.inf
        for i in range(len(foodList)):
            food = foodList[i]
            distance = manhattanDistance(food, pacmanPosition)
            if distance < minDistance:
                minDistance = distance

        if minDistance == 0:
            score += 2
        else:
            score += 1.0 / minDistance

    for i in range(len(ghostPositions)):
        ghostPosition = ghostPositions[i]
        scaredTime = scaredTimers[i]
        distance = manhattanDistance(pacmanPosition, ghostPosition)
        if scaredTime > 0:
            score += (2.0 * scaredTime) / (distance + 1)
        else:
            if distance >= 3:
                score += 1.0 / (distance + 1)
            else:
                score -= 10

    return score





# Abbreviation
better = betterEvaluationFunction
