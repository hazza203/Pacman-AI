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

# bits and pieces in this file are from baselineTeam.py

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game


def getClosestOpp(agent, gameState):
  closest_dist = 99999
  closest_pos = None
  opps = agent.getOpponents(gameState)
  for opp in opps:
    opp_pos = gameState.getAgentPosition(opp)
    if opp_pos is not None:
      dist = agent.getMazeDistance(gameState.getAgentPosition(agent.index), opp_pos)
      if dist < closest_dist:
        closest_dist = dist
        closest_pos = opp_pos
  return closest_pos, closest_dist

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
    first = 'OffensiveAgent', second = 'DefensiveAgent'):

  # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class BaseAgent(CaptureAgent):

  def registerInitialState(self, gameState):
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
    self.start = gameState.getAgentPosition(self.index)
    self.init()
    # beliefs: a list of beliefs about each agent's position indexed
    # by agent index
    # Each element is a Counter of probabilities with board coordinates
    # as the key
    walls = gameState.getWalls()
    self.mazeWidth = walls.width
    self.mazeHeight = walls.height
    self.beliefs = []
    self.ourSide = (walls.width / 2) - 1 if gameState.isOnRedTeam(self.index) else (walls.width / 2) + 1

    for agent in range(gameState.getNumAgents()):
      self.beliefs.append(util.Counter())
      self.resetBelief(gameState, agent)

  def chooseAction(self, gameState):
    self.updateBeliefs(gameState)
    if self.index == 0:
      self.displayDistributionsOverPositions(self.beliefs)
    self.updateState(gameState)

    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getAdjacentPositions(self, gameState, pos):
    '''
    Returns a list of adjacent positions (including the same position)
    '''
    ret = [pos]
    x = pos[0] - 1
    y = pos[1]
    if x >= 0 and not gameState.hasWall(x, y):
      ret.append((x, y))
    x = pos[0] + 1
    y = pos[1]
    if x < self.mazeWidth and not gameState.hasWall(x, y):
      ret.append((x, y))
    x = pos[0]
    y = pos[1] - 1
    if y >= 0 and not gameState.hasWall(x, y):
      ret.append((x, y))
    x = pos[0]
    y = pos[1] + 1
    if y < self.mazeHeight and not gameState.hasWall(x, y):
      ret.append((x, y))
    return ret

  def resetBelief(self, gameState, agent):
    initProb = 1.0 / (self.mazeWidth * self.mazeHeight)
    for x in range(self.mazeWidth):
      for y in range(self.mazeHeight):
        if gameState.hasWall(x, y):
          self.beliefs[agent][(x,y)] = 0.0
        else:
          self.beliefs[agent][(x,y)] = initProb

  def updateBeliefs(self, gameState):
    '''
    Update the probabilities of each agent being in each position
    using Bayesian Inference
    '''
    lastState = self.getPreviousObservation()
    thisAgentPos = gameState.getAgentPosition(self.index)
    fuzzyReadings = gameState.getAgentDistances()
    for agent in range(gameState.getNumAgents()):
      # if we can actually see the agent set their current position
      # as the only location
      actualPos = gameState.getAgentPosition(agent)
      if actualPos:
        for pos in self.beliefs[agent]:
          self.beliefs[agent][pos] = 0.0
        self.beliefs[agent][actualPos] = 1.0
        continue
      # if we cannot see the agent but we could see them last turn
      # then they were most likely killed and we no longer know
      # where they are
      elif lastState and lastState.getAgentPosition(agent) != None:
        self.resetBelief(gameState, agent)
        continue
      # The agent may have moved!
      # For each position we marginalise over the positions the agent may
      # have been last turn
      # Assuming the agent is moving randomly the probability of making a particular
      # move is 1/(number of adjacent positions)
      # (The agent may remain stationary so the same position is considered adjacent)
      newBelief = util.Counter()
      for pos in self.beliefs[agent]:
       adjPositions = self.getAdjacentPositions(gameState, pos)
       prob = 1.0 / len(adjPositions)
       for adj in adjPositions:
        newBelief[adj] += self.beliefs[agent][pos] * prob
      self.beliefs[agent] = newBelief
      # PofE: the marginal probability of getting this fuzzy reading
      # (E for 'Evidence')
      # We get this by marginalising over the probability of getting
      # the reading for each position
      # i.e. where E is the reading and Xn is the position:
      # P(E) = P(E|X1)P(X1) + P(E|X2)P(X2) + ...
      # This is the denominator in Baye's rule
      PofE = 0
      for pos in self.beliefs[agent]:
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
      for pos in self.beliefs[agent]:
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

  def nearestOpponent(self, gameState):
    '''
    This will return the maze distance to where we think the nearest
    opponent most likely is
    '''
    selfPos = gameState.getAgentPosition(self.index)
    positions = [self.bestGuess(opp) for opp in self.getOpponents(gameState)]
    nearestPos = None
    minDist = 9999
    for pos in positions:
      if not pos:
        continue
      # getMazeDistance seems to complain about positions being off the grid
      # after a pacman has been eaten
      try:
        dist = self.getMazeDistance(selfPos, pos)
      except Exception:
        continue
      if dist < minDist:
        minDist = dist
        nearestPos = pos
    return minDist

  def nearestInvader(self, gameState):
    selfPos = gameState.getAgentPosition(self.index)
    positions = [self.bestGuess(opp) for opp in self.getOpponents(gameState)]
    minDist = 9999
    for pos in positions:
      if not pos:
        continue
      x,_ = pos
      if gameState.isOnRedTeam(self.index):
        if x > self.ourSide:
          continue
      else:
        if x < self.ourSide:
          continue
      try:
        dist = self.getMazeDistance(selfPos, pos)
      except Exception:
        continue
      if dist < minDist:
        minDist = dist
    return minDist

  def canSeeOpponent(self, gameState):
    "Returns True if there is an opponent in sight"
    for opp in self.getOpponents(gameState):
      if gameState.getAgentPosition(opp):
        return True
    return False

  def getLastAction(self, previousState, currentState):
    oldPos = previousState.getAgentPosition(self.index)
    curPos = currentState.getAgentPosition(self.index)

    x,y = oldPos
    nx, ny = curPos
    if nx > x:
      return Directions.EAST
    elif nx < x:
      return Directions.WEST
    elif ny > y:
      return Directions.NORTH
    elif ny < y:
      return Directions.SOUTH
    else:
      return Directions.STOP


class DefensiveAgent(BaseAgent):

  def init(self):
    pass

  def updateState(self, gameState):
    pass

  def getFeatures(self, gameState, action):
    features = util.Counter()
    nextState = gameState.generateSuccessor(self.index, action)
    features['score'] = self.getScore(nextState)
    if self.nearestInvader(gameState) > self.nearestOpponent(gameState) and self.nearestInvader(gameState) != 9999:
      features['nearestOpponent'] = self.nearestInvader(nextState)
    else:
      features['nearestOpponent'] = self.nearestOpponent(nextState)
    if nextState.getAgentState(self.index).isPacman:
      features['onHomeSide'] = 0
    else:
      features['onHomeSide'] = 1
    enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if action == Directions.STOP: features['stop'] = 1

    return features

  def getWeights(self, gameState, action):
    return {
        'score': 1.0,
        'nearestOpponent': -10.0,
        'onHomeSide': 100.0,
        'numInvaders': -1000.0,
        'stop': -100.0
        }

class OffensiveAgent(BaseAgent):

  def init(self):
    self.foodCarried = 0
    self.onHomeSide = True
    self.headingHome = False
    self.scaredTime = 0

  def updateState(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    foodList = self.getFood(gameState).asList()
    foodThisTurn = len(foodList)
    lastState = self.getPreviousObservation()
    opps = self.getOpponents(gameState)
    self.scaredTime = 0
    for opp in opps:
      if gameState.getAgentState(opp).scaredTimer > 0:
        self.scaredTime = gameState.getAgentState(opp).scaredTimer
    if lastState:
      foodLastTurn = len(self.getFood(lastState).asList())
      if foodThisTurn < foodLastTurn:
        self.foodCarried += 1
    if gameState.getAgentState(self.index).isPacman:
      self.onHomeSide = False
    else:
      self.onHomeSide = True
    if self.foodCarried > 3 or len(foodList) == 2:
      self.headingHome = True
    if self.scaredTime > 5:
      self.headingHome = False
    if self.foodCarried > 0 and self.onHomeSide:
      self.foodCarried = 0
      self.headingHome = False

  def getFeatures(self, gameState, action):
    nextState = gameState.generateSuccessor(self.index, action)
    myPos = nextState.getAgentState(self.index).getPosition()
    foodList = self.getFood(nextState).asList()
    
    features = util.Counter()
    features['score'] = self.getScore(nextState)
    nearestopp = self.nearestOpponent(nextState)
    if self.scaredTime == 0:
      if nearestopp < 4:
        if len(nextState.getLegalActions(self.index)) == 2:
          features['trapped'] = 1
        if nearestopp != 0:
          features['enemyNearby'] = 3 / nearestopp
          if nearestopp < 6 and self.onHomeSide:
            features['enemyNearby'] = 6 / nearestopp
        else:
          features['enemyNearby'] = 1
      caps = self.getCapsules(gameState)
      if len(caps) > 0:
        nearestCapsule = min([self.getMazeDistance(myPos, cap) for cap in caps])
        if nearestCapsule < 4:
          features['nearestCapsule'] = 1
          if len(self.getCapsules(nextState)) == 0:
            features['eatenCapsule'] = 1
    features['foodLeft'] = len(foodList)
    if len(foodList) > 0:
      features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in foodList])
    features['distanceFromStart'] = self.getMazeDistance(myPos, self.start)
    if action == Directions.STOP: features['stop'] = 1
    if nextState.getAgentPosition(self.index) == self.start and not self.onHomeSide:
      features['died'] = 1

    return features

  def getWeights(self, gameState, action):

    weights = util.Counter()
    weights['score'] = 1.0
    weights['enemyNearby'] = -200.0
    weights['nearestCapsule'] = 100.0
    weights['eatenCapsule'] = 1000
    weights['foodLeft'] = -100.0
    weights['distanceToFood'] = -2.0
    weights['trapped'] = -100
    weights['died'] = -10000
    if self.headingHome:
      weights['distanceFromStart'] = -5.0
    weights['stop'] = -200.0
    return weights

# vi: sw=2
