//#include "stdafx.h"
#include "net.h"


/*
class constructors and methods for Net.cpp/.h
*/
Net::Net(const std::vector<unsigned> &topology) {
	unsigned numLayers = topology.size();

	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1]; //inline or num outputs == 0 if last index or (:) topology at next index


		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) { //add bias neuron with <=
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));

			std::cout << "made a neuron" << std::endl;
		}
		//
		m_layers.back().back().setOutputVal(1.0);
	}

}

void Net::feedForward(const std::vector<double> &inputVals) {
	assert(inputVals.size() == m_layers[0].size());
	//set input
	for (unsigned k = 0; k < inputVals.size(); k++) {
		m_layers[0][k].setOutputVal(inputVals[k]); 
	}

	//feedforward
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		Layer &prevLayer = m_layers[layerNum - 1]; //get previous layer
		for (unsigned neuron = 0; neuron < m_layers[layerNum].size()-1; neuron++) {
			m_layers[layerNum][neuron].feedForward(prevLayer);
		}
	}
}


void Net::backProp(const std::vector<double> &targetVals) {
	// calculate overall net error 

	Layer &outputLayer = m_layers.back(); 
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) { //ignore bias neuron. 
		double delta = targetVals[n] - outputLayer[n].getOutputVal(); 
		m_error += delta * delta;
		m_error = sqrt(m_error);
	}

	m_error /= outputLayer.size() - 1;

	//preint out some information
	m_recentAverageError = (m_recentAverageError * m_recentAverageSoothingFactor)
		/ (m_recentAverageSoothingFactor + 1.0);
	
	//calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//calculate gradients on hidden layers

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) { //go through all the hidden layers -2 (for the input and output layers)
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum +1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}

	}

	//update connection weights

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);
		}
	}

}

void Net::getResults(std::vector<double> &resultVals) const {

	resultVals.clear(); 
	for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal);
	}
}
