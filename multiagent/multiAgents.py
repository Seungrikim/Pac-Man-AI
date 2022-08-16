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

        "*** YOUR CODE HERE ***"
        #print(newPos)
        #print(newFood)
        #print(newGhostStates)
        #print(newScaredTimes[0])
        #print(successorGameState)
        #print(len(newGhostStates))
        #score= successorGameState.getScore()
        #print(score)
        score = successorGameState.getScore()
        for ghost in range(len(newGhostStates)):
        	distOfGhost = util.manhattanDistance(newPos, newGhostStates[ghost].getPosition())
        	distOfFood = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        	if distOfGhost <= newScaredTimes[ghost]:
        		score += distOfGhost
        	elif distOfGhost < 2:
        		score -= distOfGhost
        	if newPos in newFood.asList():
        		score += 0.1 * min(distOfFood)
        	if distOfFood:
        		score -= 0.1 * min(distOfFood)
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        def maxValue(state, depth, agentIndex):
        	if state.isWin() or state.isLose() or depth == self.depth:
        		return self.evaluationFunction(state)
        	maxSize = -float("inf")
        	legalActions = state.getLegalActions(agentIndex)
        	for act in legalActions:
        		maxSize = max(maxSize, minValue(state.generateSuccessor(agentIndex, act), depth, 1))
        	return maxSize

        def minValue(state, depth, agentIndex):
        	if state.isWin() or state.isLose() or depth == self.depth:
        		return self.evaluationFunction(state)
        	legalAction = state.getLegalActions(agentIndex)
        	maxSize = float("inf")
        	for act in legalAction:
        		if agentIndex + 1 < gameState.getNumAgents():
        			maxSize = min(maxSize, minValue(state.generateSuccessor(agentIndex, act), depth, agentIndex + 1))
        		else:
        			maxSize = min(maxSize, maxValue(state.generateSuccessor(agentIndex, act), depth + 1, 0))
        	return maxSize

        legalAction = gameState.getLegalActions()
        maxSize = -float("inf")
        for act in legalAction:
        	value = minValue(gameState.generateSuccessor(0, act), 0, 1)
        	if value > maxSize:
        		maxSize = value
        		action = act
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
        def maxValue(state, depth, agentIndex, alpha, beta):
        	if state.isWin() or state.isLose() or depth == self.depth:
        		return self.evaluationFunction(state)
        	maxSize = -float("inf")
        	legalActions = state.getLegalActions(agentIndex)
        	for act in legalActions:
        		maxSize = max(maxSize, minValue(state.generateSuccessor(agentIndex, act), depth, 1, alpha, beta))
        		if maxSize > beta:
        			return maxSize
        		alpha = max(alpha, maxSize)	
        	return maxSize

        def minValue(state, depth, agentIndex, alpha, beta):
        	if state.isWin() or state.isLose() or depth == self.depth:
        		return self.evaluationFunction(state)
        	legalAction = state.getLegalActions(agentIndex)
        	maxSize = float("inf")
        	for act in legalAction:
        		if agentIndex + 1 < gameState.getNumAgents():
        			maxSize = min(maxSize, minValue(state.generateSuccessor(agentIndex, act), depth, agentIndex + 1, alpha, beta))
        		else:
        			maxSize = min(maxSize, maxValue(state.generateSuccessor(agentIndex, act), depth + 1, 0, alpha, beta))
        		if maxSize < alpha:
        			return maxSize
        		beta = min(beta, maxSize)
        	return maxSize

        legalAction = gameState.getLegalActions()
 
        maxSize = -float("inf")
        a = -float("inf")
        b = float("inf")     
        for act in legalAction:
        	value = minValue(gameState.generateSuccessor(0, act), 0, 1, a, b)
        	if value > maxSize:
        		maxSize = value
        		action = act
        		a = value
        return action

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
        "*** YOUR CODE HERE ***"
        def maxValue(state, depth, agentIndex):
        	if state.isWin() or state.isLose() or depth == self.depth:
        		return self.evaluationFunction(state)
        	maxSize = -float("inf")
        	legalActions = state.getLegalActions(agentIndex)
        	for act in legalActions:
        		maxSize = max(maxSize, expValue(state.generateSuccessor(agentIndex, act), depth, 1))
        	return maxSize

        def expValue(state, depth, agentIndex):
        	if state.isWin() or state.isLose() or depth == self.depth:
        		return self.evaluationFunction(state)
        	legalAction = state.getLegalActions(agentIndex)
        	maxSize = 0
        	for act in legalAction:
        		if agentIndex + 1 < gameState.getNumAgents():
        			maxSize += expValue(state.generateSuccessor(agentIndex, act), depth, agentIndex + 1)
        		else:
        			maxSize += maxValue(state.generateSuccessor(agentIndex, act), depth + 1, 0)
        	return maxSize / len(legalAction)

        legalAction = gameState.getLegalActions()
        maxSize = -float("inf")
        for act in legalAction:
        	value = expValue(gameState.generateSuccessor(0, act), 0, 1)
        	if value > maxSize:
        		maxSize = value
        		action = act
        return action
        
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    It is basically same implementation with evaluationfunction that I impleted on q1,
    and I define different reward in each situations. For example, 
    using manhattanDistance to define distance of each ghost, then while pacman in scaredTime,
    make try to chase ghost and eat it. It has huge reward, and also not in scaredTime, 
    then try to run away from ghost. Also using manhattanDistance to define distance of each food
    then find closest food then give reward of it.  
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    for ghost in range(len(newGhostStates)):
        distOfGhost = util.manhattanDistance(newPos, newGhostStates[ghost].getPosition())
       	distOfFood = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
       	if distOfGhost <= newScaredTimes[ghost] and distOfGhost > 0:
       		score += 100 / distOfGhost
       	elif distOfGhost < 2 and distOfGhost > 0:
       		score -= 10 / distOfGhost
       	if newPos in newFood.asList():
       		score += 10 / min(distOfFood)
       	if distOfFood:        		
       		score += 10 / min(distOfFood)
    return score

# Abbreviation
better = betterEvaluationFunction
