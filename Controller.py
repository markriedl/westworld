import sys
from Observation import *
from Reward import *
from Action import *
from Agent import *
from Environment import *
import numpy


# Training episodes
episodes = 1000

trainingReportRate = 1000

# How many memories can the agent have?
numMemories = 1 #2#

# Reverie mode is false by default
reverie = False #3#

# Retrain the agent after reverie?
retrain = False


#Max reward received in any iteration
maxr = None

# Set up environment for initial training
gridEnvironment = Environment()
gridEnvironment.randomStart = True
gridEnvironment.humanWander = False
gridEnvironment.verbose = False
gridEnvironment.humanCanTorture = True #4#

# Set up agent
gridAgent = Agent(gridEnvironment)
gridAgent.verbose = False

# This is where learning happens
for i in range(episodes):
	# Train
	gridAgent.agent_reset()
	gridAgent.qLearn(gridAgent.initialObs)
	# Test
	gridAgent.agent_reset()
	gridAgent.executePolicy(gridAgent.initialObs)
	# Report
	totalr = gridAgent.totalReward
	if maxr == None or totalr > maxr:
		maxr = totalr
	
	if i%(episodes/trainingReportRate) == 0:
		print "iteration:", i, "max reward:", maxr


# Reset the environment for policy execution
gridEnvironment.verbose = True
gridEnvironment.randomStart = True # Don't change this or memories won't be created properly!
gridEnvironment.humanWander = False
gridEnvironment.humanCanTorture = True

gridAgent.verbose = True

# Make a number of memories. Also doubles as testing
print "---"
for i in range(numMemories):
	print "Execute Policy", i
	gridAgent.agent_reset()
	gridAgent.executePolicy(gridAgent.initialObs)
	print "total reward", gridAgent.totalReward
	gridAgent.memory.append(gridAgent.trace)
	print "---"


# Reverie mode
if reverie:
	# get agent ready to learn from memories
	gridAgent.lastAction=Action()
	gridAgent.lastObservation=Observation()

	gridAgent.verbose = True
	gridEnvironment.verbose = True

	# Replaying memories creates the value table that the agent would have if all it had to go on was the memories
	print "Replaying memories", len(gridAgent.memory)
	gridEnvironment.randomStart = False # Don't change this for the replay
	counter = 0
	print "---"
	for m in gridAgent.memory:
		obs = m[0][0].worldState
		print "Learn from memory", counter
		print "init state", obs
		gridEnvironment.startState = obs
		gridAgent.agent_reset()
		gridAgent.lastAction=Action()
		gridAgent.lastObservation=Observation()
		gridAgent.gridEnvironment = gridEnvironment
		gridAgent.initialObs = gridEnvironment.env_start()
		gridAgent.initializeInitialObservation(gridEnvironment)
		gridAgent.replayMemory(gridAgent.initialObs, m)
		# Report
		print "replay", counter, "total reward", gridAgent.totalReward
		print "---"
		counter = counter + 1

	# Reset the environment for policy execution
	gridEnvironment = Environment()
	gridEnvironment.verbose = True
	gridEnvironment.randomStart = True
	gridEnvironment.humanWander = False
	gridEnvironment.humanCanTorture = True

	gridAgent.gridEnvironment = gridEnvironment
	gridAgent.agent_reset()

	gridAgent.verbose = True


	# Test new v table
	print "---"
	for i in range(100):
		print "Execute Post-Reverie Policy", i
		gridAgent.initialObs = gridEnvironment.env_start()
		gridAgent.initializeInitialObservation(gridEnvironment)
		gridAgent.agent_reset()
		gridAgent.executePolicy(gridAgent.initialObs)
		print "total reward", gridAgent.totalReward
		gridAgent.memory.append(gridAgent.trace)
		print "---"


# Retrain the agent
if retrain:
	maxr = None
	for i in range(0):
		# Train
		gridAgent.agent_reset()
		gridAgent.qLearn(gridAgent.initialObs)
		# Test
		gridAgent.agent_reset()
		gridAgent.executePolicy(gridAgent.initialObs)
		# Report
		totalr = gridAgent.totalReward
		if maxr == None or totalr > maxr:
			maxr = totalr
		
		if i%(episodes/trainingReportRate) == 0:
			print "iteration:", i, "max reward:", maxr

	# Reset the environment for policy execution
	gridEnvironment.verbose = True
	gridEnvironment.randomStart = True
	gridEnvironment.humanWander = False
	gridEnvironment.humanCanTorture = True
	gridAgent.agent_reset()
	
	# Test new v table
	print "---"
	for i in range(numMemories):
		print "Execute Policy", i
		gridAgent.initialObs = gridEnvironment.env_start()
		gridAgent.initializeInitialObservation(gridEnvironment)
		gridAgent.agent_reset()
		gridAgent.executePolicy(gridAgent.initialObs)
		print "total reward", gridAgent.totalReward
		gridAgent.memory.append(gridAgent.trace)
