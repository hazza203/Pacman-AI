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

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):

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

class OffensiveAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.distances = dict()
        self.scaredTimer = 40
        self.scary = False
        self.collected_food = 0
        self.coordinates = dict()
        self.at_center = False
        self.at_top = False
        self.at_bottom = False

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

        self.coordinates['height'] = len(gameState.getWalls()[0])

        width = 0
        for wall in gameState.getWalls().asList():
            if wall[1] == 0:
                width += 1

        self.coordinates['halfway'] = width / 2
        width = (width / 2) - 4 if gameState.isOnRedTeam(self.index) else (width / 2) + 4

        self.coordinates['width'] = width
        self.coordinates['center'] = width, self.coordinates['height'] / 2
        self.coordinates['top_center'] = width, self.coordinates['center'][1] + self.coordinates['height'] / 3
        self.coordinates['bottom_center'] = width, self.coordinates['center'][1] - self.coordinates['height'] / 3

        while gameState.hasWall(self.coordinates['center'][0], self.coordinates['center'][1]):
            width = width - 1 if gameState.isOnRedTeam(self.index) else width + 1
            self.coordinates['center'] = width, self.coordinates['height'] / 2

        while gameState.hasWall(self.coordinates['top_center'][0], self.coordinates['top_center'][1]):
            width = width - 1 if gameState.isOnRedTeam(self.index) else width + 1
            self.coordinates['top_center'] = width, self.coordinates['center'][1] + self.coordinates['height'] / 3

        while gameState.hasWall(self.coordinates['bottom_center'][0], self.coordinates['bottom_center'][1]):
            width = width - 1 if gameState.isOnRedTeam(self.index) else width + 1
            self.coordinates['bottom_center'] = width, self.coordinates['center'][1] - self.coordinates['height'] / 3

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

    def chooseAction(self, gameState):

        if self.getScore(gameState) <= 0 and self.collected_food == 0:
            if self.getScore(gameState) > 0:
                self.collected_food = 0
            return self.getActionToFood(gameState)
        else:
            actions = gameState.getLegalActions(self.index)
            best_action = None
            opps = self.getOpponents(gameState)
            for opp in opps:
                opp_pos = gameState.getAgentPosition(opp)
                if opp_pos is not None:
                    if not gameState.getAgentState(opp).isPacman:
                        return Directions.STOP
                    dist = self.getMazeDistance(gameState.getAgentPosition(self.index), opp_pos)
                    for action in actions:
                        nextState = gameState.generateSuccessor(self.index, action)
                        new_dist = self.getMazeDistance(nextState.getAgentPosition(self.index), opp_pos)
                        if new_dist < dist:
                            if gameState.getAgentState(self.index).isPacman:
                                continue
                            best_action = action

            if best_action is not None:
                return best_action

            if not self.at_center:
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['center'])
                nextState = gameState.generateSuccessor(self.index, best_action)
                if nextState.getAgentPosition(self.index) == self.coordinates['center']:
                    self.at_center = True
                    self.at_bottom = False
                return best_action if best_action is not None else random.choice(actions)

            else:
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['bottom_center'])
                nextState = gameState.generateSuccessor(self.index, best_action)
                if nextState.getAgentPosition(self.index) == self.coordinates['bottom_center']:
                    self.at_center = False
                    self.at_bottom = True
                return best_action if best_action is not None else random.choice(actions)

            return random.choice(actions)


    def get_best_action_for_pos(self, gameState, actions, pos):
        min = 9999
        best_action = None
        for action in actions:
            nextState = gameState.generateSuccessor(self.index, action)
            dist = self.getMazeDistance(nextState.getAgentPosition(self.index), pos)
            if dist < min:
                min = dist
                best_action = action
        return best_action

    def getActionToFood(self, gameState):
        actions = gameState.getLegalActions(self.index)
        food_pos, food_dist = self.getClosestFood(gameState)

        if food_pos is None:
            return None

        for action in actions:
            next_state = gameState.generateSuccessor(self.index, action)
            new_dist = self.getMazeDistance(next_state.getAgentPosition(self.index), food_pos)
            if new_dist < food_dist:
                if next_state.getAgentPosition(self.index) == food_pos:
                    self.collected_food = 1
                    print "BLAH", self.collected_food
                return action

    def getActionIfScary(self, gameState, closest_opp, dist):
        actions = gameState.getLegalActions(self.index)
        for action in actions:
            nextState = gameState.generateSuccessor(self.index, action)
            new_dist = nextState.getMazeDistance(nextState.getAgentPosition(self.index), closest_opp)
            if new_dist < dist:
                return action

    def getActionIfScared(self, gameState, closest_opp, dist):
        actions = gameState.getLegalActions(self.index)
        cap, cap_dist = self.getClosestCapsule(gameState)

        # If capsule is closer than ghost go for capsule
        if cap_dist <= dist:
            for action in actions:
                nextState = gameState.generateSuccessor(self.index, action)
                new_dist = self.getMazeDistance(nextState.getAgentPosition(self.index), cap)
                if new_dist > cap_dist:
                    return action

        if gameState.isOnRedTeam:
            if Directions.EAST in actions:
                actions.remove(Directions.EAST)
        else:
            if Directions.WEST in actions:
                actions.remove(Directions.WEST)

        for action in actions:
            nextState = gameState.generateSuccessor(self.index, action)
            new_dist = self.getMazeDistance(nextState.getAgentPosition(self.index), closest_opp)
            if new_dist > dist:
                return action

        # Trapped
        return self.goHome(gameState)

    def getClosestFood(self, gameState):
        food_list = self.getFood(gameState).asList()
        our_pos = gameState.getAgentPosition(self.index)
        closest_dist = 99999
        closest_pos = None
        if len(food_list) == 0:
            return closest_pos, closest_dist

        for food in food_list:
            if (our_pos, food) not in self.distances:
                dist = self.getMazeDistance(our_pos, food)
                self.distances[(our_pos, food)] = dist
            else:
                dist = self.distances[(our_pos, food)]
            if dist < closest_dist:
                closest_dist = dist
                closest_pos = food

        return closest_pos, closest_dist

    def getClosestCapsule(self, gameState):
        capsules = self.getCapsules(gameState)
        our_pos = gameState.getAgentPosition(self.index)
        closest_dist = 99999
        closest_pos = None
        for cap in capsules:
            if (our_pos, cap) not in self.distances:
                dist = self.getMazeDistance(our_pos, cap)
                self.distances[(our_pos, cap)] = dist
            else:
                dist = self.distances[(our_pos, cap)]
            if dist < closest_dist:
                closest_dist = dist
                closest_pos = cap
        return closest_pos, closest_dist

    def goHome(self, gameState):
        actions = gameState.getLegalActions(self.index)
        our_pos = gameState.getAgentPosition(self.index)
        x, y = our_pos

        x = 1 if gameState.isOnRedTeam(self.index) \
            else self.coordinates['width'] - 1

        while gameState.hasWall(x, y):
            x = x + 1 if gameState.isOnRedTeam(self.index) else x - 1

        dist = self.getMazeDistance(our_pos, (x, y))

        for action in actions:
            next_state = gameState.generateSuccessor(self.index, action)
            new_dist = self.getMazeDistance(next_state.getAgentPosition(self.index), (x, y))
            if new_dist < dist:
                return action

##########
# Agents #
##########

class DefensiveAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.coordinates = dict()
        self.at_center = False
        self.at_top = False
        self.at_bottom = False
        self.noisyDists = dict()
        self.avgNoisyDists = dict()

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
        self.coordinates['height'] = len(gameState.getWalls()[0])

        width = 0
        for wall in gameState.getWalls().asList():
            if wall[1] == 0:
                width += 1

        width = (width / 2) - 4 if gameState.isOnRedTeam(self.index) else (width / 2) + 4

        self.coordinates['width'] = width
        self.coordinates['center'] = width, self.coordinates['height'] / 2
        self.coordinates['top_center'] = width, self.coordinates['center'][1] + self.coordinates['height'] / 3
        self.coordinates['bottom_center'] = width, self.coordinates['center'][1] - self.coordinates['height'] / 3

        while gameState.hasWall(self.coordinates['center'][0], self.coordinates['center'][1]):
            width = width - 1 if gameState.isOnRedTeam(self.index) else width + 1
            self.coordinates['center'] = width, self.coordinates['height'] / 2

        while gameState.hasWall(self.coordinates['top_center'][0], self.coordinates['top_center'][1]):
            width = width - 1 if gameState.isOnRedTeam(self.index) else width + 1
            self.coordinates['top_center'] = width, self.coordinates['center'][1] + self.coordinates['height'] / 3

        while gameState.hasWall(self.coordinates['bottom_center'][0], self.coordinates['bottom_center'][1]):
            width = width - 1 if gameState.isOnRedTeam(self.index) else width + 1
            self.coordinates['bottom_center'] = width, self.coordinates['center'][1] - self.coordinates['height'] / 3

        # self.coordinates['']

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

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        self.updateNoisyAvg(gameState)

        actions = gameState.getLegalActions(self.index)
        best_action = None
        opps = self.getOpponents(gameState)
        for opp in opps:
            opp_pos = gameState.getAgentPosition(opp)
            if opp_pos is not None:
                if not gameState.getAgentState(opp).isPacman:
                    return Directions.STOP
                dist = self.getMazeDistance(gameState.getAgentPosition(self.index), opp_pos)
                for action in actions:
                    nextState = gameState.generateSuccessor(self.index, action)
                    new_dist = self.getMazeDistance(nextState.getAgentPosition(self.index), opp_pos)
                    if new_dist < dist:
                        if gameState.getAgentState(self.index).isPacman:
                            continue
                        best_action = action

        if best_action is not None:
            return best_action

        if self.getScore(gameState) <= 0:
            if not self.at_center:
                self.at_center = True if gameState.getAgentPosition(self.index) == self.coordinates['center'] else False
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['center'])
                return best_action if best_action is not None else random.choice(actions)

            elif not self.at_top:
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['top_center'])
                nextState = gameState.generateSuccessor(self.index, best_action)
                if nextState.getAgentPosition(self.index) == self.coordinates['top_center']:
                    self.at_top = True
                    self.at_bottom = False
                return best_action if best_action is not None else random.choice(actions)

            elif not self.at_bottom:
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['bottom_center'])
                nextState = gameState.generateSuccessor(self.index, best_action)
                if nextState.getAgentPosition(self.index) == self.coordinates['bottom_center']:
                    self.at_top = False
                    self.at_bottom = True
                return best_action if best_action is not None else random.choice(actions)
        else:
            if not self.at_center:
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['center'])
                nextState = gameState.generateSuccessor(self.index, best_action)
                if nextState.getAgentPosition(self.index) == self.coordinates['center']:
                    self.at_center = True
                    self.at_top = False
                return best_action if best_action is not None else random.choice(actions)

            else:
                best_action = self.get_best_action_for_pos(gameState, actions, self.coordinates['top_center'])
                nextState = gameState.generateSuccessor(self.index, best_action)
                if nextState.getAgentPosition(self.index) == self.coordinates['top_center']:
                    self.at_top = True
                    self.at_center = False
                return best_action if best_action is not None else random.choice(actions)

        return random.choice(actions)

    def getClosestNoisyDist(self, gameState):
        opps = self.getOpponents(gameState)
        noisy_dists = gameState.getAgentDistances()
        smallest = 50

        for opp in opps:
            if noisy_dists[opp] < smallest:
                smallest = noisy_dists[opp]

        return smallest

    def updateNoisyAvg(self, gameState):
        opps = self.getOpponents(gameState)
        noisy_dists = gameState.getAgentDistances()

        for opp in opps:
            gamma = 0.5
            total = 0
            if opp not in self.noisyDists:
                self.noisyDists[opp] = []
                self.noisyDists[opp].append(abs(noisy_dists[opp]))
            else:
                if len(self.noisyDists[opp]) == 5:
                    self.noisyDists[opp] = self.noisyDists[opp][1:]
                    self.noisyDists[opp].append(abs(noisy_dists[opp]))
                else:
                    self.noisyDists[opp].append(abs(noisy_dists[opp]))

            for x in range(len(self.noisyDists[opp]) - 1, -1, -1):
                total += gamma * self.noisyDists[opp][x]
                gamma = gamma * gamma

    def get_best_action_for_pos(self, gameState, actions, pos):
        min = 9999
        best_action = None
        for action in actions:
            nextState = gameState.generateSuccessor(self.index, action)
            dist = self.getMazeDistance(nextState.getAgentPosition(self.index), pos)
            if dist < min:
                min = dist
                best_action = action
        return best_action

