import random
import sys
import copy
import operator
from Observation import *
from Reward import *
from Action import *
from Environment import *
from random import Random





class Agent:
	# Random generator
	randGenerator=Random()

	# Remember last action
	lastAction=Action()

	# Remember last observation (state)
	lastObservation=Observation()
	
	# Q-learning stuff: Step size, epsilon, gamma, learning rate
	stepsize = 0.1
	epsilon = 0.5
	gamma = 0.9
	learningRate = 0.5

	# Value table
	v_table = None

	# The environment
	gridEnvironment = None
	
	#Initial observation
	initialObs = None
	
	#Current observation
	currentObs = None
	
	# The environment will run for no more than this many steps
	numSteps = 500
	
	# Total reward
	totalReward = 0.0
	
	# action trace
	trace = []
	
	# agent memory. A list of traces. Memory is not ever reset.
	memory = []
	
	# Print debugging statements
	verbose = True
	
	# Number of actions in the environment
	numActions = 5

	# Constructor, takes a reference to an Environment
	def __init__(self, env):

		# Initialize value table
		self.v_table={}
		
		# Set dummy action and observation
		self.lastAction=Action()
		self.lastObservation=Observation()
		
		# Set the environment
		self.gridEnvironment = env
		
		# Get first observation and start the environment
		self.initialObs = self.gridEnvironment.env_start()
		self.initializeInitialObservation(env)
		
	def initializeInitialObservation(self, env):
		if self.calculateFlatState(self.initialObs.worldState) not in self.v_table.keys():
			self.v_table[self.calculateFlatState(self.initialObs.worldState)] = self.numActions*[0.0]
        
	# Once learning is done, use this to run the agent
	# observation is the initial observation
	def executePolicy(self, observation):
		# Start the counter
		count = 0
		# reset total reward
		self.totalReward = 0.0
		# Copy the initial observation
		self.workingObservation = self.copyObservation(observation)
		
		# Make sure the value table has the starting observation
		if self.calculateFlatState(self.workingObservation.worldState) not in self.v_table.keys():
			self.v_table[self.calculateFlatState(self.workingObservation.worldState)] = self.numActions*[0.0]

		if self.verbose:
			print("START")
		
		# While a terminal state has not been hit and the counter hasn't expired, take the best action for the current state
		while not self.workingObservation.isTerminal and count < self.numSteps:
			newAction = Action()
			# Get the best action for this state
			newAction.actionValue = self.greedy(self.workingObservation)
			# Store the action
			self.trace.append((self.workingObservation, newAction))

			if self.verbose == True:
				print self.gridEnvironment.actionToString(newAction.actionValue)

			# execute the step and get a new observation and reward
			currentObs, reward = self.gridEnvironment.env_step(newAction)
			# update the value table
			if self.calculateFlatState(currentObs.worldState) not in self.v_table.keys():
				self.v_table[self.calculateFlatState(currentObs.worldState)] = self.numActions*[0.0]
			self.totalReward = self.totalReward + reward.rewardValue
			self.workingObservation = copy.deepcopy(currentObs)

			# increment counter
			count = count + 1
        
		if self.verbose:
			print("END")

	# find a trace in memory that starts with the given observation, replay it
	def replay(self, observation):
		# copy the initial observation
		activeTrace = None
	
		# Find something in memory that matches the initial observation
		for trace in self.memory:
			if trace[0][0].worldState == observation.worldState:
				activeTrace = trace
				print "trace found"
				break

		if activeTrace is not None:
			self.replayMemory(observation, activeTrace)
		else:
			print "trace not found"
			
	# replay a specific memory trace
	def replayMemory(self, observation, activeTrace):
		# copy the initial observation
		self.workingObservation = self.copyObservation(observation)
		self.totalReward = 0.0
		count = 0
		lastAction = -1
		while not self.workingObservation.isTerminal and count < self.numSteps:
			# Get the next action from the memory trace
			currentTraceItem = activeTrace.pop(0)
			nextTraceItem = None
			if len(activeTrace) > 0:
				nextTraceItem = activeTrace[0] #if this is the end of the trace, there is no next
			newAction = currentTraceItem[1]
			print "action", newAction.actionValue
			lastAction = newAction.actionValue
			# Get the new state and reward from the environment
			currentObs, reward = self.gridEnvironment.env_step(newAction)
			# if new observation doesn't match the expected next observation, terminate
			if nextTraceItem is not None and currentObs.worldState != nextTraceItem[0].worldState:
				print "replay failed", currentObs.worldState, "!=", nextTraceItem[0].worldState
				return
			rewardValue = reward.rewardValue
			#update value table
			if self.calculateFlatState(currentObs.worldState) not in self.v_table.keys():
				self.v_table[self.calculateFlatState(currentObs.worldState)] = self.numActions*[0.0]
			lastFlatState = self.calculateFlatState(self.workingObservation.worldState)
			newFlatState = self.calculateFlatState(currentObs.worldState)
			if not currentObs.isTerminal:
				Q_sa=self.v_table[lastFlatState][newAction.actionValue]
				Q_sprime_aprime=self.v_table[newFlatState][self.returnMaxIndex(currentObs)]
				new_Q_sa=Q_sa + self.stepsize * (rewardValue + self.gamma * Q_sprime_aprime - Q_sa)
				self.v_table[lastFlatState][lastAction]=new_Q_sa
			else:
				Q_sa=self.v_table[lastFlatState][lastAction]
				new_Q_sa=Q_sa + self.stepsize * (rewardValue - Q_sa)
				self.v_table[lastFlatState][lastAction] = new_Q_sa
			# increment counter
			count = count + 1
			self.workingObservation = self.copyObservation(currentObs)
			# increment total reward
			self.totalReward = self.totalReward + reward.rewardValue
									
		# Done learning, reset environment
		self.gridEnvironment.env_reset()

	# q-learning implementation
	# observation is the initial observation
	def qLearn(self, observation):
		# copy the initial observation
		self.workingObservation = self.copyObservation(observation)
		
		# start the counter
		count = 0

		lastAction = -1
		
		# reset total reward
		self.totalReward = 0.0
		
		# while terminal state not reached and counter hasn't expired, use epsilon-greedy search
		while not self.workingObservation.isTerminal and count < self.numSteps:
			
			# Take the epsilon-greedy action
			newAction = Action()
			newAction.actionValue = self.egreedy(self.workingObservation)
			lastAction = newAction.actionValue

			# Get the new state and reward from the environment
			currentObs, reward = self.gridEnvironment.env_step(newAction)
			rewardValue = reward.rewardValue
			
			# update the value table
			if self.calculateFlatState(currentObs.worldState) not in self.v_table.keys():
				self.v_table[self.calculateFlatState(currentObs.worldState)] = self.numActions*[0.0]
			lastFlatState = self.calculateFlatState(self.workingObservation.worldState)
			newFlatState = self.calculateFlatState(currentObs.worldState)
			if not currentObs.isTerminal:
				Q_sa=self.v_table[lastFlatState][newAction.actionValue]
				Q_sprime_aprime=self.v_table[newFlatState][self.returnMaxIndex(currentObs)]
				new_Q_sa=Q_sa + self.stepsize * (rewardValue + self.gamma * Q_sprime_aprime - Q_sa)
				self.v_table[lastFlatState][lastAction]=new_Q_sa
			else:
				Q_sa=self.v_table[lastFlatState][lastAction]
				new_Q_sa=Q_sa + self.stepsize * (rewardValue - Q_sa)
				self.v_table[lastFlatState][lastAction] = new_Q_sa
			
			# increment counter
			count = count + 1
			self.workingObservation = self.copyObservation(currentObs)
		
			# increment total reward
			self.totalReward = self.totalReward + reward.rewardValue


		# Done learning, reset environment
		self.gridEnvironment.env_reset()


	def returnMaxIndex(self, observation):
		flatState = self.calculateFlatState(observation.worldState)
		actions = observation.availableActions
		qValueArray = []
		qValueIndexArray = []
		for i in range(len(actions)):
			qValueArray.append(self.v_table[flatState][actions[i]])
			qValueIndexArray.append(actions[i])

		return qValueIndexArray[qValueArray.index(max(qValueArray))]

	# Return the best action according to the policy, or a random action epsilon percent of the time
	def egreedy(self, observation):
		maxIndex=0
		actualAvailableActions = []
		for i in range(len(observation.availableActions)):
			actualAvailableActions.append(observation.availableActions[i])

		if self.randGenerator.random() < self.epsilon:
			randNum = self.randGenerator.randint(0,len(actualAvailableActions)-1)
			return actualAvailableActions[randNum]

		else:
			v_table_values = []
			flatState = self.calculateFlatState(observation.worldState)
			for i in actualAvailableActions:
				v_table_values.append(self.v_table[flatState][i])
			return actualAvailableActions[v_table_values.index(max(v_table_values))]

	# Return the best action according to the policy
	def greedy(self, observation):
        
		actualAvailableActions = []
		for i in range(len(observation.availableActions)):
			actualAvailableActions.append(observation.availableActions[i])
		v_table_values = []
		flatState = self.calculateFlatState(observation.worldState)
		for i in actualAvailableActions:
			v_table_values.append(self.v_table[flatState][i])
		return actualAvailableActions[v_table_values.index(max(v_table_values))]
	

	# Reset the agent
	def agent_reset(self):
		self.lastAction = Action()
		self.lastObservation = Observation()
		self.initialObs = self.gridEnvironment.env_start()
		self.trace = []

	# Create a copy of the observation
	def copyObservation(self, obs):
		returnObs =  Observation()
		if obs.worldState != None:
			returnObs.worldState = obs.worldState[:]
            
		if obs.availableActions != None:
			returnObs.availableActions = obs.availableActions[:]
        
		if obs.isTerminal != None:
			returnObs.isTerminal = obs.isTerminal
            
		return returnObs
	
	# Turn the state into a tuple for bookkeeping
	def calculateFlatState(self, theState):
		return tuple(theState)
