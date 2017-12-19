// NeuralNetTestProject.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Net.h"
#include <vector>
#include <iostream>

using namespace std;

int main()
{
	//e.g. 
	

	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net myNet(topology);

	/*
	std::vector<double> inputVals; 
	std::vector<double> targetVals;
	std::vector<double> resultVals;


	myNet.feedForward(inputVals);
	myNet.backProp(targetVals);
	myNet.getResults(resultVals); */

	system("pause");

    return 0;
}

