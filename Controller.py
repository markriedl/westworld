import sys
from Observation import *
from Reward import *
from Action import *
from Agent import *
from Environment import *
import numpy

# Training episodes
episodes = 1000

trainingReportRate1 = 1000
trainingReportRate2 = 1000

# Reverie mode is false by default
reverie = True

# How much of the agent's value table is erased in reverie mode?
forget = 0.5

# How many memories can the agent have?
numMemories = 1000

#Max reward received in any iteration
maxr = None

# Set up environment
gridEnvironment = Environment()
gridEnvironment.randomStart = True
gridEnvironment.humanWander = False
gridEnvironment.verbose = False
gridEnvironment.humanCanTorture = False

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
	
	if i%(episodes/trainingReportRate1) == 0:
		print "iteration:", i, "max reward:", maxr

# Reset the environment for policy execution
gridEnvironment.verbose = True
gridEnvironment.randomStart = True # Don't change this
gridEnvironment.humanWander = False
gridEnvironment.humanCanTorture = False

gridAgent.verbose = True

# Make a number of memories. Also doubles as testing
for i in range(numMemories):
	print "Execute Policy", i
	gridAgent.agent_reset()
	gridAgent.executePolicy(gridAgent.initialObs)
	print "total reward", gridAgent.totalReward
	gridAgent.memory.append(gridAgent.trace)


# Reverie mode
if reverie:
	# Wipe forget % of memories
	memories = gridAgent.memory
	if forget >= 1.0:
		# Wipe all memories
		memories = []
	elif forget > 0.0:
		# Wipe only a fraction of memories
		deleteMe = []
		for m in memories:
			r = random.random()
			if r < forget:
				deleteMe.append(m)
		#memories = list(set(memories) - set(deleteMe))
		memories = [x for x in memories if x not in deleteMe]
	gridAgent.memory = memories

	# Wipe the agent's v table
	gridAgent.v_table = {}
	gridAgent.lastAction=Action()
	gridAgent.lastObservation=Observation()

	gridAgent.verbose = True
	gridEnvironment.verbose = True

	# Replaying memories creates the value table that the agent would have if all it had to go on was the memories
	print "Replaying memories", len(memories)
	gridEnvironment.randomStart = False # Don't change this for the replay
	counter = 0
	for m in gridAgent.memory:
		obs = m[0][0].worldState
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
		#for x in gridAgent.v_table.keys():
		#	print x, "=>", gridAgent.v_table[x]
		counter = counter + 1

	print "final v table after replay"
	for x in gridAgent.v_table.keys():
		print x, "=>", gridAgent.v_table[x]

	# Reset the environment for policy execution
	gridEnvironment = Environment()
	gridEnvironment.verbose = False
	gridEnvironment.randomStart = True
	gridEnvironment.humanWander = False
	gridEnvironment.humanCanTorture = True

	gridAgent.gridEnvironment = gridEnvironment
	gridAgent.agent_reset()

	gridAgent.verbose = False


#	# Test v table
#	for i in range(10):
#		print "Execute Reverie Policy", i
#		gridAgent.initialObs = gridEnvironment.env_start()
#		gridAgent.initializeInitialObservation(gridEnvironment)
#		gridAgent.agent_reset()
#		gridAgent.executePolicy(gridAgent.initialObs)
#		print "total reward", gridAgent.totalReward
#		gridAgent.memory.append(gridAgent.trace)


	# Retrain the agent
	maxr = None
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
		
		if i%(episodes/trainingReportRate2) == 0:
			print "iteration:", i, "max reward:", maxr

	# Reset the environment for policy execution
	#gridEnvironment.verbose = True
	#gridEnvironment.randomStart = True
	#gridEnvironment.humanWander = False
	#gridEnvironment.humanCanTorture = True
	#gridAgent.agent_reset()
	
	#print "Execute Policy"
	#gridAgent.executePolicy(gridAgent.initialObs)
	#print "total reward", gridAgent.totalReward
