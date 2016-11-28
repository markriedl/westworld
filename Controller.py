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
forget = 0.5

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
#gridEnvironment.randomStart = False
#gridEnvironment.humanWander = False
gridEnvironment.humanCanTorture = True
gridAgent.agent_reset()

print "Execute Policy"
gridAgent.executePolicy(gridAgent.initialObs)
print "total reward", gridAgent.totalReward


### Reverie mode looks at how learning is different if the agent's value table isn't completely erased
if reverie:
	if forget >= 1.0:
		gridAgent.v_table = {}
	else:
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
	gridEnvironment.humanCanTorture = True

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
	#gridEnvironment.randomStart = False
	#gridEnvironment.humanWander = False
	gridEnvironment.humanCanTorture = True
	gridAgent.agent_reset()

	print "Execute Policy"
	gridAgent.executePolicy(gridAgent.initialObs)
	print "total reward", gridAgent.totalReward