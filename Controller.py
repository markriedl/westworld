import sys
from Observation import *
from Reward import *
from Action import *
from Agent import *
from Environment import *
import numpy

reverie = True
forget = 0.5
maxr = None

# Set up environment
gridEnvironment = Environment()
gridEnvironment.randomStart = False
gridEnvironment.humanWander = False
gridEnvironment.verbose = False

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
gridAgent.agent_reset()

print "Execute Policy"
gridAgent.executePolicy(gridAgent.initialObs)
print "total reward", gridAgent.totalReward

if reverie:
	if forget >= 1.0:
		gridAgent.v_table = {}
	else:
		keys = gridAgent.v_table.keys()[:]
		for key in keys:
			r = random.random()
			if r < forget:
				del key

	# store a reference to the v_table
	old_v_table = gridAgent.v_table

	# Set up new environment
	gridEnvironment = Environment()
	gridEnvironment.randomStart = False
	gridEnvironment.humanWander = False
	gridEnvironment.verbose = False

	# New agent
	gridAgent = Agent(gridEnvironment)
	gridAgent.v_table = old_v_table

	# Retrain the agent
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
	gridAgent.agent_reset()

	print "Execute Policy"
	gridAgent.executePolicy(gridAgent.initialObs)
	print "total reward", gridAgent.totalReward