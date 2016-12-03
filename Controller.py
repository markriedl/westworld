import sys
from Observation import *
from Reward import *
from Action import *
from Agent import *
from Environment import *
import numpy

# Reverie mode is false by default
reverie = False

# How much of the agent's value table is erased in reverie mode?
forget = 0.0

# How many memories can the agent have?
numMemories = 50

#Max reward received in any iteration
maxr = None

# Set up environment
gridEnvironment = Environment()
gridEnvironment.randomStart = False
gridEnvironment.humanWander = False
gridEnvironment.verbose = False
gridEnvironment.humanCanTorture = True

# Set up agent
gridAgent = Agent(gridEnvironment)

# Training episodes
episodes = 10000

# This is where learning happens
for i in range(episodes):
	gridAgent.qLearn(gridAgent.initialObs)
	totalr = gridAgent.totalReward
	if maxr == None or totalr > maxr:
		maxr = totalr
	
	if i%100 == 0:
		print "iteration:", i, "max reward:", maxr

# Reset the environment for policy execution
gridEnvironment.verbose = True
gridEnvironment.randomStart = True
gridEnvironment.humanWander = False
gridEnvironment.humanCanTorture = True

# Make a number of memories
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
		memories = list(set(memories) - set(deleteMe))
	gridAgent.memory = memories

	# Wipe the agent's v table
	gridAgent.v_table = {}
	gridAgent.lastAction=Action()
	gridAgent.lastObservation=Observation()

	# Replaying memories creates the value table that the agent would have if all it had to go on was the memories
	print "Replaying memories", len(memories)
	gridEnvironment.randomStart = False # Don't change this for the replay
	counter = 0
	for m in gridAgent.memory:
		obs = m[0][0].worldState
		print "obs", obs
		gridEnvironment.startState = obs
		gridAgent.agent_reset()
		gridAgent.v_table = {}
		gridAgent.lastAction=Action()
		gridAgent.lastObservation=Observation()
		gridAgent.gridEnvironment = gridEnvironment
		gridAgent.initialObs = gridEnvironment.env_start()
		gridAgent.initializeInitialObservation(gridEnvironment)
		gridAgent.replayMemory(gridAgent.initialObs, m)
		print "replay", counter, "total reward", gridAgent.totalReward
		counter = counter + 1

	print "final v table"
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
		gridAgent.qLearn(gridAgent.initialObs)
		totalr = gridAgent.totalReward
		if maxr == None or totalr > maxr:
			maxr = totalr
		
		if i%100 == 0:
			print "iteration:", i, "max reward:", maxr

	# Reset the environment for policy execution
	gridEnvironment.verbose = True
	gridEnvironment.randomStart = True
	gridEnvironment.humanWander = False
	gridEnvironment.humanCanTorture = True
	gridAgent.agent_reset()
	
	print "Execute Policy"
	gridAgent.executePolicy(gridAgent.initialObs)
	print "total reward", gridAgent.totalReward


### What next?
### Reverie mode deletes x% of replays
### Replay memories
### Look at how fast to converge
### Torture to no-torture (normal reward)
### No-torture to torture (normal reward)
### Torture to no-torture (psycho reward)
### no-torture to torture (psycho reward)

#gridAgent.agent_reset()
#gridAgent.v_table = {}
#gridAgent.initializeInitialObservation(gridEnvironment)
#gridAgent.replay(gridAgent.initialObs)
#print "replay total reward", gridAgent.totalReward
#for x in gridAgent.v_table.keys():
#	print x, "=>", gridAgent.v_table[x]


### Reverie mode looks at how learning is different if the agent's value table isn't completely erased
if False:
	if forget >= 1.0:
		gridAgent.v_table = {}
		gridAgent.initializeInitialObservation(gridEnvironment)
	elif forget > 0.0:
		keys = gridAgent.v_table.keys()[:]
		for key in keys:
			r = random.random()
			if r < forget:
				del key
		gridAgent.initializeInitialObservation(gridEnvironment)



	# store a reference to the v_table
	old_v_table = gridAgent.v_table

	# Set up new environment
	gridEnvironment = Environment()
	gridEnvironment.randomStart = False
	gridEnvironment.humanWander = False
	gridEnvironment.verbose = False
	gridEnvironment.humanCanTorture = False

	# New agent
	gridAgent = Agent(gridEnvironment)
	gridAgent.v_table = old_v_table

	# Retrain the agent
	maxr = None
	for i in range(episodes):
		gridAgent.qLearn(gridAgent.initialObs)
		totalr = gridAgent.totalReward
		if maxr == None or totalr > maxr:
			maxr = totalr
	
		if i%100 == 0:
			print "iteration:", i, "max reward:", maxr

	# Reset the environment for policy execution
	gridEnvironment.verbose = True
	gridEnvironment.randomStart = False
	gridEnvironment.humanWander = False
	gridEnvironment.humanCanTorture = False
	gridAgent.agent_reset()

	print "Execute Policy"
	gridAgent.executePolicy(gridAgent.initialObs)
	print "total reward", gridAgent.totalReward