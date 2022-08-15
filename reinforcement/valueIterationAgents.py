# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
          values = util.Counter()
          for state in states:
            if self.mdp.isTerminal(state):
              values[state] = 0.0
            else:
              actions = self.mdp.getPossibleActions(state)
              maxVal = max([self.getQValue(state, act) for act in actions])
              values[state] = maxVal
          self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return 0.0
        qvalue = 0.0
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        for trans in transition:
          nextState = trans[0]
          prob = trans[1]
          qvalue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return 0.0
        action = self.mdp.getPossibleActions(state)
        value = -float("inf")
        for act in action:
          qval = self.getQValue(state, act)
          if qval > value:
            value = qval
            computedAction = act
        return computedAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        maxVal = 0.0
        for i in range(self.iterations):
          newState = states[i % len(states)]
          values = util.Counter()
          if self.mdp.isTerminal(newState):
            values[newState] = 0.0
          else:
            actions = self.mdp.getPossibleActions(newState)
            maxVal = max([self.getQValue(newState, act) for act in actions])
            self.values[newState] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        values = util.Counter()
        pq = util.PriorityQueue()
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
          predecessors[state] = set()
        for state in states:
          for action in self.mdp.getPossibleActions(state):
            transition = self.mdp.getTransitionStatesAndProbs(state, action)
            for trans in transition:
              nextState = trans[0]
              prob = trans[1]
              if prob != 0:
                predecessors[nextState].add(state)
            #print("prob val", prob)
          if not self.mdp.isTerminal(state):
            maxVal = -float("inf")
            actions = self.mdp.getPossibleActions(state)
            maxVal = max([self.getQValue(state, act) for act in actions])
            diff = abs(self.values[state] - maxVal)
            #print("diff val", diff)
            pq.push(state, -diff)

        for i in range(self.iterations):
          if not pq.isEmpty():
            state = pq.pop()
            #print("state", state)
            if not self.mdp.isTerminal(state):
              actions = self.mdp.getPossibleActions(state)
              maxVal = max([self.getQValue(state, act) for act in actions])
              self.values[state] = maxVal
            for pre in predecessors[state]:
              actions = self.mdp.getPossibleActions(pre)
              maxVal = max([self.getQValue(pre, act) for act in actions])
              diff = abs(self.values[pre] - maxVal)
              if diff > self.theta:
                pq.update(pre, -diff)