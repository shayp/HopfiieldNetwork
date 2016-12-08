from HopfieldNetwork import *
from ImagesHandler import *
from RandomPatternsHandler import *
import matplotlib.pyplot as pil
from scipy import *
def main():
	# ****** Parans *******
	errorLimit = 0.138
	xSize = 20
	ySize = 20
	PATTERN_SIZE = xSize * ySize
	ImageToRecall = 0
	numberOfSteps = 2000
	errorsInInPattern = 0.25
	patternLimitAssumption = int(math.ceil(PATTERN_SIZE * errorLimit))
	#patternLimitAssumption = 

	images = getImageArrayFromFolder("/Users/shay-macbook/Desktop/Neuroscience/computationalNeuroscience/HopfieldNetwork/forest", 'jpg',xSize, ySize)
	print len(images)
	#numberOfPatterns = len(images)
	numberOfPatterns = 80
	#images[2].show()
	imagepatterns = fromImageArrayToPatternArray(images)

	randomPatternsToTrain = makeRandomPatterns(85, PATTERN_SIZE)
	TestPatternsChangeCapacity(randomPatternsToTrain, numberOfPatterns, 
							 patternLimitAssumption,
							 errorsInInPattern, 0, numberOfSteps)
	# For Images Use
	#TestPatternsChangeCapacity(imagepatterns, numberOfPatterns, 
	#						 patternLimitAssumption,
	#						 errorsInInPattern, ImageToRecall, numberOfSteps)


def TestPatternsChangeCapacity(patternsVector, numberOfPatterns,
							 patternsLimitAssumption, errorRateInPattern, indexToCheck,
							 numberOfSteps):

	#errorImageToTest = degradePattern(patternsVector[indexToCheck].copy(), errorRateInPattern)
	errorImageToTest = toggleNValuesInVector(int (errorRateInPattern * 400), patternsVector[indexToCheck].copy())
	#errorImageToTest = patternsVector[indexToCheck].copy()
	vector = range(numberOfPatterns + 1)
	steps = range(numberOfPatterns + 1)
	distance = range(numberOfPatterns + 1)
	for x in range(1,numberOfPatterns):
		print "$$$$$$$$ Capacity check for  " + str(x) + " pictures in the network $$$$$$$$$$$$"
		HopfieldNet= hopfieldNetwork(patternsVector[:x], numberOfSteps)
		vector[x], steps[x], distance[x] =  HopfieldNet.recall_async(errorImageToTest,patternsVector[indexToCheck])
		print x
		print  spatial.distance.euclidean(patternsVector[indexToCheck], vector[x])
		#display(vector[x], x);

	pil.figure(x + 1)
	pil.plot(steps[patternsLimitAssumption - 3], distance[patternsLimitAssumption - 3], 'mo',steps[patternsLimitAssumption - 2], distance[patternsLimitAssumption - 2], 'ro', steps[patternsLimitAssumption - 1], distance[patternsLimitAssumption - 1], 'go', steps[patternsLimitAssumption], distance[patternsLimitAssumption], 'bo')
	#pil.axis([-1,100.0000,-1,20])
	pil.xlabel('steps of single neuron')
	pil.ylabel('sqeuclidean distance')
	pil.grid(True)
	pil.figure(x + 2)
	for x in range(0, numberOfPatterns):
		pil.plot(x, spatial.distance.euclidean(patternsVector[indexToCheck], vector[x]), 'bo') 
		pil.xlabel('Number of patterns in the network')
		pil.ylabel('sqeuclidean distance at the last step')
	pil.figure(x + 3)
	for x in range(1, numberOfPatterns):
		pil.plot(x, len(steps[x]), 'bo')
		pil.xlabel('Number of patterns in the network')
		pil.ylabel('last step')
	pil.show()


if __name__ == "__main__":
	main()