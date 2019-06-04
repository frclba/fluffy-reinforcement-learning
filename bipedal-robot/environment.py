from bipedal import *
import sys

agent = Agent()

agent.load('weights.pkl')

# train for 100 iterations
agent.train(100)

# the pre-trained weights are saved into 'weights.pkl' which you can use.
agent.save('weights.pkl')

# play one episode
agent.play(1)
