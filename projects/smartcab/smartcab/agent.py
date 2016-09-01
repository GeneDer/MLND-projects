import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = ['forward', 'left', 'right', None]
        self.alpha = 0.2
        self.alphaC = 1 - self.alpha
        self.gamma = 0.8
        self.epsilon = 0.5
        self.decayRate = 0.05
        self.qTable = {}
        self.initialReward = None
        
        self.numDestReach = 0
        self.totalPosReward = 0
        self.totalNegReward = 0
        self.totalTimeUsed = 0
        self.totalDisplacements = 0
        
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.initialReward = None
        self.initialPosition = self.env.agent_states[self]['location']
        if self.epsilon > 0:
            self.epsilon -= self.decayRate

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = '0' if inputs['light'] == 'green' else '1'
        self.state += '0' if inputs['oncoming'] == None or inputs['oncoming'] == 'left' else '1'
        self.state += '0' if inputs['left'] == 'forward' else '1'
        if self.next_waypoint == 'forward':
            self.state += '0'
        elif self.next_waypoint == 'left':
            self.state += '1'
        else:
            self.state += '2'
        
        # TODO: Select action according to your policy
        if random.random() > self.epsilon and self.state in self.qTable:
            qValues = self.qTable[self.state]
            maxQV = max(qValues)
            idx = []
            for i in xrange(4):
                if qValues[i] == maxQV:
                    idx.append(i)
            action = self.actions[idx[random.randint(0,len(idx)-1)]]
        else:
            action = self.actions[random.randint(0,3)]
            
        # Execute action and get reward
        reward = self.env.act(self, action)
        if self.initialReward == None:
            self.initialReward = reward

        self.totalTimeUsed += 1
        if reward < 0:
            self.totalNegReward += reward
        elif reward >= 10:
            self.numDestReach += 1
            self.totalPosReward += reward
            endingLocation = self.env.agent_states[self]['location']
            self.totalDisplacements += self.env.compute_dist(self.initialPosition, endingLocation)
        else:
            self.totalPosReward += reward
        if deadline == 0:
            endingLocation = self.env.agent_states[self]['location']
            self.totalDisplacements += self.env.compute_dist(self.initialPosition, endingLocation)

        # TODO: Learn policy based on state, action, reward
        next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        postState = '0' if inputs['light'] == 'green' else '1'
        postState += '0' if inputs['oncoming'] == None or inputs['oncoming'] == 'left' else '1'
        postState += '0' if inputs['left'] != 'forward' else '1'
        if next_waypoint == 'forward':
            postState += '0'
        elif next_waypoint == 'left':
            postState += '1'
        else:
            postState += '2'

        if random.random() > self.epsilon and self.state in self.qTable:
            qValues = self.qTable[self.state]
            maxQV = max(qValues)
            idx = []
            for i in xrange(4):
                if qValues[i] == maxQV:
                    idx.append(i)
            postAction = self.actions[idx[random.randint(0,len(idx)-1)]]
        else:
            postAction = self.actions[random.randint(0,3)]

        if self.state not in self.qTable:
            self.qTable[self.state] = [self.initialReward]*4
        if postState not in self.qTable:
            self.qTable[postState] = [self.initialReward]*4
        
        idx = self.actions.index(action)
        postIdx = self.actions.index(postAction)
        self.qTable[self.state][idx] = self.alphaC*self.qTable[self.state][idx] + \
                                       self.alpha*(reward + self.gamma * \
                                                   self.qTable[postState][postIdx])
        
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        

def run():
    """Run the agent for a finite number of trials."""
    random.seed(1)
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "Destination reach rate: ", a.numDestReach/100.0
    print "Average Positive reward: ", a.totalPosReward/100.0
    print "Average Negative reward: ", a.totalNegReward/100.0
    print "Average time used: ", a.totalTimeUsed/100.0
    print "Average velocity: ", float(a.totalDisplacements)/a.totalTimeUsed
    print a.qTable

if __name__ == '__main__':
    run()
