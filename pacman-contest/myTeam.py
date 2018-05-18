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

import cProfile
import pstats

debugOpt = False

def getAdjacentPositions(gameState, pos):
  '''
  Returns a list of adjacent positions (including the same position)
  '''
  walls = gameState.getWalls()
  mazeWidth = walls.width
  mazeHeight = walls.height
  ret = [pos]
  x = pos[0] - 1
  y = pos[1]
  if x >= 0 and not gameState.hasWall(x, y):
    ret.append((x, y))
  x = pos[0] + 1
  y = pos[1]
  if x < mazeWidth and not gameState.hasWall(x, y):
    ret.append((x, y))
  x = pos[0]
  y = pos[1] - 1
  if y >= 0 and not gameState.hasWall(x, y):
    ret.append((x, y))
  x = pos[0]
  y = pos[1] + 1
  if y < mazeHeight and not gameState.hasWall(x, y):
    ret.append((x, y))
  return ret

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
    first = 'OffensiveAgent', second = 'DefensiveAgent',
    debug = False):

  global debugOpt
  debugOpt = debug

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
    # beliefs: a list of beliefs about each agent's position indexed
    # by agent index
    # Each element is a Counter of probabilities with board coordinates
    # as the key
    walls = gameState.getWalls()
    self.opponentsIndexes = self.getOpponents(gameState)
    self.mazeWidth = walls.width
    self.mazeHeight = walls.height
    self.beliefs = []
    self.oppPos = [0, 0, 0, 0]
    self.ourSide = (walls.width / 2) - 1 if gameState.isOnRedTeam(self.index) else (walls.width / 2) + 1

    for agent in range(gameState.getNumAgents()):
      self.beliefs.append(util.Counter())
      self.resetBelief(gameState, agent)
    # init subclass
    self.init()

  def chooseAction(self, gameState):
    if debugOpt:
      self.debugClear()
    self.updateBeliefs(gameState)
    if debugOpt and self.index == 0:
      self.displayDistributionsOverPositions(self.beliefs)
    # update subclass
    self.turnUpdate(gameState)
    for opp in self.getOpponents(gameState):
      if gameState.getAgentPosition(opp):
        if self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(opp)) < 5:
          if debugOpt == 'profile':
            pr = cProfile.Profile()
            pr.enable()
          _, action = self.expectiMax(gameState, 2, opp, self.index)
          if debugOpt == 'profile':
            pr.disable()
            pr.dump_stats('profile')
          return action
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def expectiMax(self, gameState, depth, opp, agent):
    actions = gameState.getLegalActions(agent)
    if depth == 0:
      self.updateState(gameState)
      return max(self.evaluate(gameState, a) for a in actions), None

    values = []
    if agent == self.index:
      for action in actions:
        nextState = gameState.generateSuccessor(self.index, action)
        values.append(self.expectiMax(nextState, depth, opp, opp)[0])
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      if Directions.STOP in bestActions and len(bestActions) > 1:
        bestActions.remove(Directions.STOP)
      return maxValue, random.choice(bestActions)
    if agent == opp:
      for action in actions:
        nextState = gameState.generateSuccessor(opp, action)
        values.append(self.expectiMax(nextState, depth - 1, opp, self.index)[0])
      total = 0
      for value in values:
        total += value
      avg = total / float(len(values))
      return avg, None

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

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
        adjPositions = getAdjacentPositions(gameState, pos)
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
    # finally update our best guesses at where the agents are
    for agent in range(gameState.getNumAgents()):
      self.updateLocation(agent)

  def updateLocation(self, agent):
    'Update our best guess at where this agent is'
    topProb = max(self.beliefs[agent].values())
    locs = [loc for (loc, prob) in self.beliefs[agent].items() if prob == topProb]
    self.oppPos[agent] = random.choice(locs)
    if debugOpt:
      self.debugDraw(self.oppPos[agent], [1,0,0])

  def bestGuess(self, agent):
    '''
    Returns where the given agent most likely is
    '''
    return self.oppPos[agent]

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

  def turnUpdate(self, gameState):
    self.updateState(gameState)

  def updateState(self, gameState):
    pass

  def getFeatures(self, gameState, action):
      features = util.Counter()
      nextState = gameState.generateSuccessor(self.index, action)
      features['score'] = self.getScore(nextState)
      nearestInvader = self.nearestInvader(nextState)
      nearestOpponent = self.nearestOpponent(nextState)
      if nearestInvader > nearestOpponent and nearestInvader != 9999:
          features['nearestOpponent'] = nearestInvader
      else:
          features['nearestOpponent'] = nearestOpponent
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
    self.plannedPos = None
    self.scaredTime = 0
    self.foodList = []

  def turnUpdate(self, gameState):
    self.updateState(gameState)
    foodLastTurn = len(self.foodList)
    self.foodList = self.getFood(gameState).asList()
    foodThisTurn = len(self.foodList)
    lastState = self.getPreviousObservation()
    if lastState:
      if foodThisTurn < foodLastTurn:
        self.foodCarried += 1
        if debugOpt:
          print 'now I have ', self.foodCarried, ' food'
    if self.foodCarried > 3 or len(self.foodList) == 2:
      self.headingHome = True
      if debugOpt:
        print "I'm heading home"
    if self.scaredTime > 5:
      self.headingHome = False
    if self.foodCarried > 0 and self.onHomeSide:
      self.foodCarried = 0
      self.headingHome = False
    self.plannedPos = self.planPath(gameState)

  def updateState(self, gameState):
    myPos = gameState.getAgentState(self.index).getPosition()
    opps = self.getOpponents(gameState)
    self.scaredTime = 0
    for opp in opps:
      if gameState.getAgentState(opp).scaredTimer > 0:
        self.scaredTime = gameState.getAgentState(opp).scaredTimer
    if gameState.getAgentState(self.index).isPacman:
      self.onHomeSide = False
    else:
      self.onHomeSide = True

  def getFeatures(self, gameState, action):
    nextState = gameState.generateSuccessor(self.index, action)
    myPos = nextState.getAgentState(self.index).getPosition()

    features = util.Counter()
    features['score'] = self.getScore(nextState)
    nearestopp = self.nearestOpponent(nextState)
    if self.scaredTime <= 4:
      if nearestopp < 4:
        if len(nextState.getLegalActions(self.index)) == 2:
          features['trapped'] = 1
        if nearestopp != 0:
          features['enemyNearby'] = 3 / nearestopp
          if nearestopp < 6 and self.onHomeSide:
            features['enemyNearby'] = 5 / nearestopp
        else:
          features['enemyNearby'] = 1
      caps = self.getCapsules(gameState)
      if len(caps) > 0:
        nearestCapsule = min([self.getMazeDistance(myPos, cap) for cap in caps])
        if nearestCapsule < 4:
          if nearestCapsule != 0:
            features['nearestCapsule'] = 3 / nearestCapsule
          else:
            features['eatenCapsule'] = 1

    elif self.scaredTime > 4:
      features['scary'] = 1
    features['distanceFromStart'] = self.getMazeDistance(myPos, self.start)
    if action == Directions.STOP: features['stop'] = 1
    if myPos == self.plannedPos:
      features['isOnPath'] = 1
    if nextState.getAgentPosition(self.index) == self.start and not self.onHomeSide:
      features['died'] = 1
    #features['foodLeft'] = len(self.foodList)
    #if len(self.foodList) > 0:
      #features['distanceToFood'] = min([self.getMazeDistance(myPos, food) for food in self.foodList])
    #features['positionHasFood'] = myPos in self.foodList

    return features

  def getWeights(self, gameState, action):

    weights = util.Counter()
    weights['score'] = 1.0
    weights['enemyNearby'] = -200.0
    weights['nearestCapsule'] = 200.0
    weights['eatenCapsule'] = 1000.0
    weights['trapped'] = -100.0
    weights['died'] = -10000.0
    weights['scary'] = 1000.0
    if self.headingHome:
      weights['distanceFromStart'] = -10.0
      weights['isOnPath'] = 0
    else:
      weights['distanceFromStart'] = 0
      weights['isOnPath'] = 10.0
    weights['stop'] = -200.0
    #weights['foodLeft'] = -100.0
    #weights['distanceToFood'] = -2.0
    #weights['positionHasFood'] = 100
    return weights


  def planPath(self, gameState):
    '''
    Finds a path to food with Djikstra's algorithm
    Positions nearer to enemies are more costly to encourage avoiding enemies
    Returns the position adjacent to this agent which is the first step
    on the path
    '''

    def distanceToEnemy(pos):
      oppPositions = [self.bestGuess(opp) for opp in self.getOpponents(gameState)]
      minDist = 9999
      for oppPos in oppPositions:
        if not oppPos:
          continue
        dist = util.manhattanDistance(pos, oppPos)
        if dist < minDist:
          minDist = dist
      return minDist

    def posCost(pos):
      AVOIDANCE_FACTOR = 100
      dist = distanceToEnemy(pos)
      # avoid division by zero
      if dist == 0:
        dist = 0.01
      return 1.0 + (1.0 / dist) * AVOIDANCE_FACTOR

    class SearchNode:
      '''
      A node in the path planning search tree.
      When creating, pass in the parent of this node and the position of this node
      '''
      def __init__(self, parent, pos):
        self.parent = parent
        self.pos = pos
        if parent:
          self.cost = parent.cost + posCost(pos)
        else:
          self.cost = posCost(pos)

    selfPos = gameState.getAgentPosition(self.index)
    frontier = util.PriorityQueue()
    food = self.getFood(gameState)
    node = SearchNode(None, selfPos)
    visited = [node.pos]
    while not food[node.pos[0]][node.pos[1]]:
      adj = getAdjacentPositions(gameState, node.pos)
      for pos in adj:
        newNode = SearchNode(node, pos)
        if pos not in visited:
          frontier.push(newNode, newNode.cost)
      # If the frontier becomes empty there is no food
      if frontier.isEmpty():
        return None
      node = frontier.pop()
      visited.append(node.pos)
    # rewind to find the first step of the path
    while node.parent != None:
      if debugOpt:
        self.debugDraw(node.pos, [0, 0, 0.5])
      nextNode = node
      node = node.parent
    return nextNode.pos

# vi: sw=2
