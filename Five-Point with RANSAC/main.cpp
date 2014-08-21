#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <fstream>
#include "5pts.h"
#include "5pt-second.h"
#include "..\Common\ACTS.h"
#include <time.h>
#include <vector>
#include <levmar.h>
using namespace std;
using namespace cv;

//ifstream fin("pts.txt");
//ifstream ein("EE.txt");
ifstream actin("test.act");
ofstream actout("out.act");
ofstream fout("fout.txt");
Intrinsc_Parameters ip;
Feature_Points fp;
int npoints;



int main()
{
	int endframe;
	int step;
	int startframe;
	ReadACTS(actin, ip, fp, npoints, startframe, step, endframe);
	Mat Q1(npoints, 2, CV_64F), Q2(npoints, 2, CV_64F);
	for (int i = 0; i < npoints; i++)
	{
		Q1.at<double>(i, 0) = fp[i].get_point(0).x;
		Q1.at<double>(i, 1) = fp[i].get_point(0).y;
		Q2.at<double>(i, 0) = fp[i].get_point(1).x;
		Q2.at<double>(i, 1) = fp[i].get_point(1).y;
	}
	Mat EM;//essential matrix
	EM = findEssentialMartix_RANSAC(Q1, Q2, ip.fx, Point2d(ip.cx, ip.cy), 0.1, 0.5, 1000);
	Mat R, t, _mask;//Rotation matrix and translation matrix
	recoverPose(EM, Q1, Q2, R, t, ip.fx, Point2d(ip.cx, ip.cy),_mask);
	//calibrate correspondences with focal and principal point
	Mat Q1h = calibrate_points_homography(Q1, ip.fx, Point2d(ip.cx, ip.cy));
	Mat Q2h = calibrate_points_homography(Q2, ip.fx, Point2d(ip.cx, ip.cy));
	Mat error = reprojected_error(R, t, Q1h, Q2h);
	fout << "----R and t reconstruction error---\n\n";
	fout << error<<endl;
	refine_Rt(R, t, Q1h, Q2h);
	fout << "----------------------------------------------" << endl;
	error = reprojected_error(R, t, Q1h, Q2h);
	fout << error << endl;
	fout << "----------------------------------------------" << endl;



	Mat P1 = Mat::eye(3,4,CV_64F), P2(3, 4, CV_64F);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			P2.at<double>(i, j) = R.at<double>(i, j);
	}
	for (int i = 0; i < 3; i++)
		P2.at<double>(i, 3) = t.at<double>(i, 0);
	

	Mat X;
	//calibrate correspondences with focal and principal point
	Mat Q1c = calibrate_points(Q1, ip.fx, Point2d(ip.cx, ip.cy));
	Mat Q2c = calibrate_points(Q2, ip.fx, Point2d(ip.cx, ip.cy));
//triangulate 3D point with opencv implementation
	triangulatePoints(P1, P2, Q1.t(), Q2.t(), X);
	Mat X2;
	Q1c = Q1c.t();
	Q2c = Q2c.t();

//triangulate 3D point with my implementation
	X2 = triangulation(P1, P2, Q1c, Q2c);
	error=reproject3D_error(P1, P2, Q1h, Q2h, X2);
	fout << "---trigulation error---\n\n";
	fout << error << endl;
	fout << "----------------------------------------------------------------" << endl;
//try to compare my result and opencv's result
	refine_triangulation(P1, P2, Q1c, Q2c, X2);
	error = reproject3D_error(P1, P2, Q1h, Q2h, X2);
	fout << error << endl;
	fout << "----------------------------------------------------------------" << endl;

	WriteACTS(actin, actout, X2);

	getchar();

}















