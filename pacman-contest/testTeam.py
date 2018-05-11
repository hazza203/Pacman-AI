# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'TestAgent', second = 'TestAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class TestAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    # beliefs: a list of beliefs about each agent's position indexed
    # by agent index
    # Each element is a Counter of probabilities with board coordinates
    # as the key
    width = gameState.getWalls().width
    height = gameState.getWalls().height
    initProb = 1.0 / (width * height)
    self.beliefs = []
    for agent in range(gameState.getNumAgents()):
      self.beliefs.append(util.Counter())
      for x in range(width):
        for y in range(height):
          if gameState.hasWall(x, y):
            self.beliefs[agent][(x,y)] = 0.0
          else:
            self.beliefs[agent][(x,y)] = initProb

  def nearestOpponent(self, gameState):
    '''
    If we can see an opponent, this will return the maze distance to that
    agent, otherwise it will return the maze distance to where we think the
    nearest opponent might be
    '''
    selfPos = gameState.getAgentPosition(self.index)
    if self.canSeeOpponent(gameState):
      positions = [gameState.getAgentPosition(opp) for opp in self.getOpponents(gameState)]
    else:
      positions = [self.bestGuess(opp) for opp in self.getOpponents(gameState)]
    nearestPos = None
    minDist = 9999
    for pos in positions:
      if not pos:
        continue
      dist = self.getMazeDistance(selfPos, pos)
      if dist < minDist:
        minDist = dist
        nearestPos = pos
    self.debugClear()
    self.debugDraw(nearestPos, [1.0, 0, 0])
    return minDist

  def canSeeOpponent(self, gameState):
    "Returns True if there is an opponent in sight"
    for opp in self.getOpponents(gameState):
      if gameState.getAgentPosition(opp):
        return True
    return False

    self.updateBeliefs(gameState)
    self.displayDistributionsOverPositions(self.beliefs)

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    nextState = gameState.generateSuccessor(self.index, action)
    features['score'] = self.getScore(nextState)
    features['nearestOpponent'] = self.nearestOpponent(nextState)
    if nextState.getAgentState(self.index).isPacman:
      features['isDefending'] = 0
    else:
      features['isDefending'] = 1
    return features

  def getWeights(self, gameState, action):
    return {
        'score': 1.0,
        'nearestOpponent': -10.0,
        'isDefending': 100
        }

  def getAdjacentPositions(self, gameState, pos):
    '''
    Returns a list of adjacent positions (including the same position)
    May be less than 5 if the position is by the edge or a wall
    '''
    width = gameState.getWalls().width
    height = gameState.getWalls().height
    ret = [pos]
    x = pos[0] - 1
    y = pos[1]
    if x >= 0 and not gameState.hasWall(x, y):
      ret.append((x, y))
    x = pos[0] + 1
    y = pos[1]
    if x < width and not gameState.hasWall(x, y):
      ret.append((x, y))
    x = pos[0]
    y = pos[1] - 1
    if y >= 0 and not gameState.hasWall(x, y):
      ret.append((x, y))
    x = pos[0]
    y = pos[1] + 1
    if y < height and not gameState.hasWall(x, y):
      ret.append((x, y))
    return ret

  def updateBeliefs(self, gameState):
    '''
    Update the probabilities of each agent being in each position
    using Bayesian Inference
    '''
    thisAgentPos = gameState.getAgentPosition(self.index)
    fuzzyReadings = gameState.getAgentDistances()
    for agent in range(gameState.getNumAgents()):
      # The agent may have moved!
      # For each position we marginalise over the positions the agent may
      # have been last turn
      # Assuming the agent is moving randomly the probability of making a particular
      # move is 1/(number of adjacent positions)
      # (The agent may remain stationary so the same position is considered adjacent)
      newBelief = util.Counter()
      for pos in self.beliefs[agent].keys():
        # The agents can't move into walls
        if gameState.hasWall(pos[0], pos[1]):
          continue
        adjPositions = self.getAdjacentPositions(gameState, pos)
        prob = 1.0 / len(adjPositions)
        newBelief[pos] = self.beliefs[agent][pos] * prob
        for adj in adjPositions:
          newBelief[pos] += self.beliefs[agent][adj] * prob
      self.beliefs[agent] = newBelief
      # PofE: the marginal probability of getting this fuzzy reading
      # (E for 'Evidence')
      # We get this by marginalising over the probability of getting
      # the reading for each position
      # i.e. where E is the reading and Xn is the position:
      # P(E) = P(E|X1)P(X1) + P(E|X2)P(X2) + ...
      # This is the denominator in Baye's rule
      PofE = 0
      for pos in self.beliefs[agent].keys():
        trueDist = util.manhattanDistance(thisAgentPos, pos)
        # PofEgivenPos: the probability we would get this reading
        # given the agent is in this position, i.e.:
        # P(E|X)
        PofEgivenPos = gameState.getDistanceProb(trueDist, fuzzyReadings[agent])
        PofE += PofEgivenPos * self.beliefs[agent][pos]
      # Avoid division by zero
      if not PofE:
        PofE = 0.001
      # Now we update with Baye's rule:
      # P(X|E) = (P(E|X) * P(X)) / P(E)
      for pos in self.beliefs[agent].keys():
        trueDist = util.manhattanDistance(thisAgentPos, pos)
        PofEgivenPos = gameState.getDistanceProb(trueDist, fuzzyReadings[agent])
        self.beliefs[agent][pos] = (PofEgivenPos * self.beliefs[agent][pos]) / PofE

  def bestGuess(self, agent):
    '''
    Returns where the given agent most likely is
    '''
    topProb = max(self.beliefs[agent].values())
    locs = [loc for (loc, prob) in self.beliefs[agent].items() if prob == topProb]
    ret = random.choice(locs)
    return ret

  def chooseAction(self, gameState):
    self.updateBeliefs(gameState)
    self.displayDistributionsOverPositions(self.beliefs)

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    choice = random.choice(bestActions)
    print 'evaluations:'
    for (a, v) in zip(actions, values):
      print a, ': ', v
    print 'choosing ', choice

    return choice

# vi: sw=2
