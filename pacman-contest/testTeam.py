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

  def __init__(self, index):
      CaptureAgent.__init__(self, index)
      self.distribs = []
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

  def nearestOpponent(self, gameState):
    "Returns the fuzzy distance to the nearest opponent"
    return min([gameState.getAgentDistances()[opp]
      for opp in self.getOpponents(gameState)])

  def canSeeOpponent(self, gameState):
    "Returns True if there is an opponent in sight"
    for opp in self.getOpponents(gameState):
      if gameState.getAgentPosition(opp):
        return True
    return False

  def nearestVisibleOpponent(self, gameState):
    '''
    Returns the position of the nearest opponent
    or None if none are visible
    '''
    # a list of opponent positions, if they are visible
    opps = [gameState.getAgentPosition(opp)
        for opp in self.getOpponents(gameState)
        if gameState.getAgentPosition(opp)]
    mindist = 9999
    nearest = None
    for opp in opps:
      dist = util.manhattanDistance(gameState.getAgentPosition(self.index), opp)
      if dist < mindist:
        mindist = dist
        nearest = opp
    return nearest

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    features['nearestOpponent'] = self.nearestOpponent(gameState)
    return features

  def getWeights(self, gameState, action):
    return {'nearestOpponent': -1.0}

  def getDistributions(self, gameState):
    selfPos = gameState.getAgentPosition(self.index)
    opps = self.getOpponents(gameState)
    fuzzyDists = gameState.getAgentDistances()
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    distribs = []
    numover = 0
    for opp in opps:
      fuzzyDist = fuzzyDists[opp]
      print fuzzyDist
      distrib = util.Counter()
      for x in range(width):

        for y in range(height):
          pos = (x, y)
          distance = util.manhattanDistance(selfPos, pos)
          prob = gameState.getDistanceProb(distance, fuzzyDist)
          if prob > 0.07:
              numover+=1
          distrib[pos] = prob
      distribs.append(distrib)
    self.distribs.append(distribs)
    print "normal num over = ", numover
    return distribs

  def getNormalizedDists(self, gameState):

    #Removing earliest not needed reading
    distribs = []
    for x in range(3):
      distrib = util.Counter()
      distribs.append(distrib)
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    if len(self.distribs) == 3:
      self.distribs = self.distribs[1:]

    for distrib in self.distribs:
      for agent in distrib:
        z = 0
        for x in range(width):
          for y in range(height):
            pos = (x, y)
            if pos not in distribs[z]:
              distribs[z][pos] = agent[pos]
            else:
              distribs[z][pos] += agent[pos]
        z += 1
    numover = 0
    for distrib in distribs:
        for x in range(width):
          for y in range(height):
            pos = (x, y)
            distrib[pos] = distrib[pos] / 3
            if distrib[pos] > 0.076:
                numover += 1

    print "numover = ", numover
    return distribs

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    dists = self.getDistributions(gameState)
    dists = self.getNormalizedDists(gameState)
    self.displayDistributionsOverPositions(dists)
    #self.displayDistributionsOverPositions([None, dists[1], None, None])
    testMap = util.Counter()
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    for x in range(width):
      for y in range(height):
        testMap[(x,y)] = float(x) / width
    #self.displayDistributionsOverPositions([testMap])

    print 'fuzzy readings: ', gameState.getAgentDistances()
    print 'unique probs: ', set(dists[1].values())
    return random.choice(bestActions)

# vi: sw=2
