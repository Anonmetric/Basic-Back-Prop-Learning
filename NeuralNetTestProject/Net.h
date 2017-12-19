#pragma once

#include "stdafx.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include "Neuron.h"

typedef std::vector<Neuron> Layer;


class Net {
	

public: 
	
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVales);
	void getResults(std::vector<double> &resultVals) const;
private:
	std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSoothingFactor;

};