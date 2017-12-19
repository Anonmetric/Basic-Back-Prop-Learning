#include "Neuron.h"

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {

	for (unsigned k = 0; k < numOutputs; k++)
	{
		m_outputWeights.push_back(Connections()); 
		m_outputWeights[k].weight = randomWeight(); 
	}
	m_myIndex = myIndex;
}

double Neuron::randomWeight() {
	return rand() / double(RAND_MAX);
}


void Neuron::feedForward(Layer &prevLayer) {

	double sum = 0.0; 
	for (unsigned pn = 0; pn < prevLayer.size(); pn++) {
		sum += prevLayer[pn].getOutputVal() *
			prevLayer[pn].m_outputWeights[m_myIndex].weight;
	}
	m_outputVal = Neuron::transferFunction(sum);
} 


double Neuron::transferFunction(double x) {
	return tanh(x); 
}

double Neuron::transferFunctionDervivative(double x){
	return 1 - x*x;
}


void Neuron::calcOutputGradients(double targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDervivative(m_outputVal);
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDervivative(m_outputVal);

}

void Neuron::updateInputWeights(Layer &prevLayer) {

	for (unsigned n = 0; n < prevLayer.size; n++) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			eta
			* neuron.getOutputVal()
			* m_gradient
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}


double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}