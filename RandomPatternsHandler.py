from numpy import *

def makeRandomPatterns(numberOfPatterns, patternSize):

	# define a lambda function to calculate the result of a single neuron
	sgn = vectorize(lambda x: -1 if x<0.5 else +1)
	randomPatterns = sgn(random.rand(numberOfPatterns, patternSize))
	return randomPatterns

def degradePattern(pattern,noise):
    #This allows you to add noise to a pattern
    sgn=vectorize(lambda x: x*-1 if random.random()<noise else x)
    out=sgn(pattern)
    return out
def toggleNValuesInVector(numberOfChanges, pattern):

	for x in range(0,numberOfChanges):
		if (pattern[x] == -1):
			pattern[x] = 1
		else:
			pattern[x] = -1
	return pattern