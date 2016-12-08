from numpy import *
from random import *
from scipy import spatial
import glob

class hopfieldNetwork:

    def __init__(self, patterns, numOfSteps=10):
        self.patterns = patterns
        self.numOfSteps = numOfSteps
        self.NetorkWeights = self.train()
        self.syncUpdateParamsStep = []
        self.asyncUpdateParamsStep = []
        self.syncUpdateParamsDistance = []
        self.asyncUpdateParamsDistance = []

    def train(self):

        #get array dimensions
        row,column = self.patterns.shape

        #define and zeroize the net
        NetorkWeights = zeros((column, column))

        # Run for every patternt that we want to insert to the network
        for currentPattern in self.patterns:
            # Add to the network the numeric relation between the pattern parts(each bit)
            NetorkWeights = NetorkWeights + outer(currentPattern,currentPattern)

        # we zeroize the diagonal line because there isn't any relatin between a bit to himself
        NetorkWeights[diag_indices(column)] = 0

        # We return the normalization of the Network, dividing in the number of patterns
        return NetorkWeights/row

    def recall_sync(self, patternToRecall, wantedOutputPattern):

        self.syncUpdateParamsStep = []
        self.syncUpdateParamsDistance = []

        # define a lambda function to calculate the result of a single neuron
        sgn = vectorize(lambda x: -1 if x<0 else +1)

        # run as the number of the steps allowed to update
        for i in xrange(self.numOfSteps):      

            self.traceSyncUpdate(i, patternToRecall, wantedOutputPattern)
            # multiple the input vector with the matrix using dot product in order to update the pattern
            patternToRecall = sgn(dot(patternToRecall,self.NetorkWeights))

        self.traceSyncUpdate(i, patternToRecall, wantedOutputPattern)

        return patternToRecall, self.syncUpdateParamsStep, self.syncUpdateParamsDistance

    """Run the network using the input data until the output state doesn't change
    or a maximum number of iteration has been reached."""
    def recall_async(self, input_pattern, wantedOutputPattern):

        self.asyncUpdateParamsStep = []
        self.asyncUpdateParamsDistance = []

        iteration_count = 0

        result = input_pattern.copy()

        while True:

            self.traceAsyncUpdate(iteration_count, result, wantedOutputPattern)
            # get all the indexes betweeb 1 and the input pattern length
            update_list = range(size(input_pattern))

            # shuffle the numbers in order to make not deterministic update
            shuffle(update_list)

            # Update the pattern in the given order
            changed, result = self.update_all_neurons(update_list, result)

            iteration_count += 1

            # If the pattern was changed or not changed but we calc until the max_iterations then we return the pattern
            #if not changed or iteration_count == self.numOfSteps:
            if (iteration_count == self.numOfSteps) or (changed == False):
                self.traceAsyncUpdate(iteration_count, result, wantedOutputPattern)
                return result, self.asyncUpdateParamsStep, self.asyncUpdateParamsDistance
                """Calculate the output of the given neuron"""

    def calculate_neuron_output(self, neuron, input_pattern):
        
        num_neurons = len(input_pattern)

        neuron_output = 0.0

        # update the neuron output by taking the vector of the specific neuron
        # from the weights matrix and mutliple with the input pattern
        for j in range(num_neurons):
            neuron_output += self.NetorkWeights[neuron][j] * input_pattern[j]
        return 1.0 if neuron_output > 0.0 else -1.0

    """Iterate over every neuron and update it's output"""
    def update_all_neurons(self, update_list, input_pattern):
        result = input_pattern.copy()

        changed = False

        # Update every neuron by the update_list order(random order) 
        for neuron in update_list:
            neuron_output = self.calculate_neuron_output(neuron, result)

            # If the neuron output was changed update the neuron
            if neuron_output != result[neuron]:
                result[neuron] = neuron_output
                changed = True
                return changed, result

        return changed, result

    def traceSyncUpdate(self, step, currentVecrtor, WantedVector):
        self.syncUpdateParamsStep += [step]
        self.syncUpdateParamsDistance += [spatial.distance.euclidean(currentVecrtor,WantedVector)]
    def traceAsyncUpdate(self, step, currentVecrtor, WantedVector):
        self.asyncUpdateParamsStep += [step]
        self.asyncUpdateParamsDistance += [spatial.distance.euclidean(currentVecrtor,WantedVector)]

    def hopfield_energy(W, patterns):
        return array([-0.5*dot(dot(p.T,W),p) for p in patterns])
