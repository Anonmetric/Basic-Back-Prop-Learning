#pragma once

#include "stdafx.h"
#include <vector>
#include <cstdlib>
#include <cmath>
//#include "Net.h"




struct Connections {
	double weight;
	double deltaWeight;
};


class Neuron {
	typedef std::vector<Neuron> Layer;

public:
	//constructor
	Neuron(unsigned numOutputs, unsigned myIndex);

	//getters setters
	void setOutputVal(double val) { m_outputVal = val;  };
	double getOutputVal() { return m_outputVal; };

	//function
	void feedForward(Layer &prevousLayer); 

	//update functions

	void calcOutputGradients(double targetVal);
	void calculateHiddenGradients(const Layer &nextLayer);

	void updateInputWeights(Layer &prevLayer);


private:
	
	std::vector<Connections> m_outputWeights;
	static double randomWeight();
	static double transferFunction(double x);
	static double transferFunctionDervivative(double x);

	double sumDOW(const Layer &nextLayer) const;

	static double eta; 
	static double alpha; 
	double m_outputVal;
	double m_gradient;
	unsigned m_myIndex;

};

double Neuron::eta = 0.15;
double Neuron::alpha = .5; 


