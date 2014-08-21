#ifndef __ACTS_H__
#define __ACTS_H__


#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2\core\core.hpp>
#include "Feature-Points.h"

using namespace std;
using namespace cv;





class Intrinsc_Parameters
{
public:
	Intrinsc_Parameters(){};
	~Intrinsc_Parameters(){};
	double fx, fy, cx, cy;
private:

};



void ReadACTS(ifstream &fin, Intrinsc_Parameters &params, Feature_Points &points, int &featurecnt, int &startframe, int &endframe, int &step);
#endif