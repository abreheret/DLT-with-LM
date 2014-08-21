#ifndef __FEATURE_POINTS_H__
#define __FEATURE_POINTS_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2\core\core.hpp>
#include <map>

using namespace cv;
using namespace std;
class Feature_Point
{
public:
	Feature_Point(){};
	~Feature_Point(){};
	void add(Point2d point, int frame)
	{
		points.insert(pair<int, Point2d>(frame, point));
//		points.push_back(point);
//		frames.push_back(frame);
	}
	Point2d get_point(int frame)
	{
		return points[frame];
	}

	//bool hasframe(int frame)
	//{
	//	return binary_search(frames.begin(), frames.end(), frame);
	//}

	//void calibrate_points(double focal, Point2d pp)
	//{
	//	calibrated_points.resize(points.size());
	//	calibrated_points.clear();
	//	for (auto i : points)
	//	{
	//		Point2d pd = Point2d((i.x - pp.x) / focal, (i.y - pp.y) / focal);
	//		calibrated_points.push_back(pd);
	//	}
	//}

private:
	map<int, Point2d> points;
//	vector<Point2d> points;
//	vector<int> frames;
//	vector<Point2d> calibrated_points;
};

class Feature_Points
{
public:
	Feature_Points(){};
	~Feature_Points(){};
	void add(Feature_Point fp)
	{
		points.push_back(fp);
	}
	void resize(int pointsnum,int images)
	{
		_npoints = pointsnum;
		_images = images;
		points.resize(pointsnum);
		index.resize(images);
		for (auto &i : index)
			i.resize(pointsnum);
	}
	void clear()
	{
		points.clear();
		for (int i = 0; i < _images;i++)
			for (int j = 0; j < _npoints; j++)
				index[i][j] = 0;
	}
	__inline void addindex(int i, int j)
	{
		index[i][j] = 1;
	}
	__inline bool getindex(int frame, int point)
	{
		if (index[frame][point])
			return true;
		else
			return false;
	}
	Feature_Point operator[] (int index)
	{
		return points[index];
	}
	vector<Feature_Point>::iterator begin()
	{
		return points.begin();
	}
	vector<Feature_Point>::iterator end()
	{
		return points.end();
	}

	vector<Feature_Point> points;
	int _images;
	int _npoints;
	vector<vector<char>> index;
};






#endif