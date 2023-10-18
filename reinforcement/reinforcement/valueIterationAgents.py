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
        self.actionDict = {}
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #("RUNNING VALUE ITERATION")
        #("Get States:",self.mdp.getStates())
        #print("Possible Actions:", self.mdp.getPossibleActions((3,2)))
        #print("Transtion States:", self.mdp.getTransitionStatesAndProbs((3,2), 'exit'))
        #print("GetReward", self.mdp.getReward((3,2), 'exit', 'TERMINAL_STATE'))
        #print("is terminal", self.mdp.isTerminal('TERMINAL_STATE'))

        #print("Iterations:",self.iterations)
        
        k=0
        while k < self.iterations:
            #print(self.values)
            kPlus1Dict = {}
            for state in self.mdp.getStates():
                
                maxAction = float('-inf')
                actionToTake = None
                for action in self.mdp.getPossibleActions(state):

                    sum = 0

                    for transitionState in self.mdp.getTransitionStatesAndProbs(state, action):
                        sPrimeState=transitionState[0]
                        sum+= transitionState[1] * (self.mdp.getReward(state, action, sPrimeState) + self.discount * self.values[sPrimeState])
                        #since there is no ordering on which state gos first, a state can be looking at a sprime state that already had its self.values updated this iteration

                    if maxAction < sum:
                        maxAction = sum
                        actionToTake = action

                if maxAction != float('-inf'):
                    kPlus1Dict[state] = maxAction
                    self.actionDict[state] = actionToTake

            k+=1
            for key in kPlus1Dict:

                self.values[key] = kPlus1Dict[key]

        #print("SELF>VALUES AT THE END OF RUN:",self.values) 
        #print(self.actionDict)



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
        
        if action == 'exit':

            return self.mdp.getReward(state, action, self.mdp.getTransitionStatesAndProbs(state,action)[0])
        sum=0
        #print("State", state, "action", action, self.mdp.getTransitionStatesAndProbs(state,action))
        for tuple in self.mdp.getTransitionStatesAndProbs(state,action):

            sum += tuple[1] * (self.mdp.getReward(state, action, tuple[0]) + self.discount*self.values[tuple[0]])
        
        return sum

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #print(state)
        if self.mdp.isTerminal(state):
            #print("returning none")
            return None

        if state not in self.actionDict:
            return None

        return self.actionDict[state]


        util.raiseNotDefined()

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

        currIter = 0
        currState=0
        while currIter < self.iterations:
            state=self.mdp.getStates()[currState]
            if state != 'TERMINAL_STATE':

                maxAction = float('-inf')
                actionToTake = None
                for action in self.mdp.getPossibleActions(state):

                    sum = 0

                    for transitionState in self.mdp.getTransitionStatesAndProbs(state, action):
                        sPrimeState=transitionState[0]
                        sum+= transitionState[1] * (self.mdp.getReward(state, action, sPrimeState) + self.discount * self.values[sPrimeState])
                        #since there is no ordering on which state gos first, a state can be looking at a sprime state that already had its self.values updated this iteration

                    if maxAction < sum:
                        maxAction = sum
                        actionToTake = action

                if maxAction != float('-inf'):
                    self.values[state] = maxAction
                    self.actionDict[state] = actionToTake       

            currState+=1
            if currState == len(self.mdp.getStates()):
                currState=0

            currIter+=1

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

