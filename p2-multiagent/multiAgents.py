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


from util import manhattanDistance
from game import Directions
import random, util, collections

from game import Agent

# Some functions used in the evaluationFunction:

def nearestFood(state):
    """
    Returns the manhattan distance to the nearest food.
    Takes a GameState as argument.
    """
    pacmanPos = state.getPacmanPosition();
    dist = []
    for food in state.getFood().asList():
        dist.append(manhattanDistance(pacmanPos, food))
    if len(dist) == 0:
        return 0
    return min(dist)

def closeToGhost(state, tolerance):
    """
    Returns whether Pacman is within "tolerance" of a scary ghost
    Takes a GameState as argument.
    """
    pacmanPos = state.getPacmanPosition();
    dist = []
    for ghost in state.getGhostStates():
        # Don't worry about scared ghosts
        if ghost.scaredTimer == 0:
            dist.append(manhattanDistance(pacmanPos, ghost.getPosition()))
    if len(dist) == 0:
        return False
    if min(dist) < tolerance:
        return True
    return False

def bonus(score, percent):
    change = abs(score) * percent / 100
    return score + change

def penalty(score, percent):
    change = abs(score) * percent / 100
    return score - change


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        # The strategy here is to apply a series of bonuses and penalties to the score
        score = successorGameState.getScore()

        # If Pacman is moving towards food, apply a bonus
        nearestFoodThen = nearestFood(currentGameState)
        nearestFoodNow = nearestFood(successorGameState)
        if nearestFoodNow < nearestFoodThen:
            score = bonus(score, 20)

        # If Pacman is close to a scary ghost apply a penalty
        if closeToGhost(successorGameState, 3):
            score = penalty(score, 30)

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
        """

        # Recursive minimax function which returns the action to take
        def minimaxTraversal(gameState, depth, agent):
            scores = []
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            actions = gameState.getLegalActions(agent)
            # If agent is 0 then we are picking the highest score
            if agent == 0:
                for action in actions:
                    next_state = gameState.generateSuccessor(agent, action)
                    # get the scores that min chose so we can choose the highest of them
                    scores.append(minimaxTraversal(next_state, depth, agent+1)[0])
                highest_score = max(scores)
                for i in range(len(scores)):
                    if scores[i] == highest_score:
                        action_index = i
                return highest_score, actions[action_index]
            # If agent is not 0 then we are a ghost agent and we are choosing the lowest score
            else:
                for action in actions:
                    next_state = gameState.generateSuccessor(agent, action)
                    # if this is the last ghost we are calculating scores for, swap to max's turn
                    if agent == gameState.getNumAgents() - 1:
                        scores.append(minimaxTraversal(next_state, depth - 1, 0)[0])
                    # else we have more ghosts to calculate scores for
                    else:
                        scores.append(minimaxTraversal(next_state, depth, agent+1)[0])
                lowest_score = min(scores)

                return lowest_score, None

        action = minimaxTraversal(gameState, self.depth, 0)[1]
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaTraversal(gameState, depth, agent, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agent)
            bestScore = None
            # If we are Pacman, choose the highest score
            if agent == 0:
                for action in actions:
                    childState = gameState.generateSuccessor(agent, action)
                    childScore = alphaBetaTraversal(
                            childState, depth, agent + 1, alpha, beta)
                    if bestScore == None or childScore > bestScore:
                        bestScore = childScore
                    if bestScore > alpha:
                        alpha = bestScore
                    if beta < alpha:
                        break
            # If we are a ghost, choose the lowest score
            else:
                for action in actions:
                    childState = gameState.generateSuccessor(agent, action)
                    # If we are the last ghost then the next move is Pacman's
                    if agent + 1 == gameState.getNumAgents():
                        childScore = alphaBetaTraversal(
                                childState, depth - 1, 0, alpha, beta)
                    else:
                        childScore = alphaBetaTraversal(
                                childState, depth, agent + 1, alpha, beta)
                    if bestScore == None or childScore < bestScore:
                        bestScore = childScore
                    if bestScore < beta:
                        beta = bestScore
                    if beta < alpha:
                        break
            return bestScore

        # Return Pacman's best move
        actions = gameState.getLegalActions(0)
        bestScore = None
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        for action in actions:
            childState = gameState.generateSuccessor(0, action)
            childScore = alphaBetaTraversal(
                    childState, self.depth, 1, alpha, beta)
            if bestScore == None or childScore > bestScore:
                bestScore = childScore
                bestAction = action
            if bestScore > alpha:
                alpha = bestScore
            if beta < alpha:
                break
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def expectimaxTraversal(gameState, depth, agent):
            scores = []
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            actions = gameState.getLegalActions(agent)
            # If agent is 0 then we are picking the highest score, exactly as a max agent
            if agent == 0:
                for action in actions:
                    next_state = gameState.generateSuccessor(agent, action)
                    # get the scores that min chose so we can choose the highest of them
                    scores.append(expectimaxTraversal(next_state, depth, agent+1)[0])
                highest_score = max(scores)
                for i in range(len(scores)):
                    if scores[i] == highest_score:
                        action_index = i
                return highest_score, actions[action_index]
            # If agent is not 0 then we are a ghost agent and we are calculating the average of all possible states
            else:
                for action in actions:
                    next_state = gameState.generateSuccessor(agent, action)
                    # if this is the last ghost we are calculating scores for, swap to max's turn
                    if agent == gameState.getNumAgents() - 1:
                        scores.append(expectimaxTraversal(next_state, depth - 1, 0)[0])
                    # else we have more ghosts to calculate scores for
                    else:
                        scores.append(expectimaxTraversal(next_state, depth, agent+1)[0])
                total = 0.0
                for score in scores:
                    total += score
                avg = total / float(len(scores))
                return avg, None

        action = expectimaxTraversal(gameState, self.depth, 0)[1]
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

