#ifndef __ACTS_H__
#define __ACTS_H__


#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2\core\core.hpp>

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


class Feature_Point
{
public:
	Feature_Point(){};
	~Feature_Point(){};
	void add(Point3d point)
	{
		points.push_back(point);
	}
	Point3d get_point(int frame)
	{
		return points[frame];
	}

private:
	vector<Point3d> points;
};
void ReadACTS(ifstream &fin, Intrinsc_Parameters &params, vector<Feature_Point> &points, int&featurecnt);
#endif