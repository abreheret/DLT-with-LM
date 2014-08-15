#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <fstream>
#include "5pts.h"
#include "5pt-second.h"
#include "ACTS.h"
#include <time.h>
#include <vector>
#include <levmar.h>
using namespace std;
using namespace cv;

ifstream fin("pts.txt");
ifstream ein("EE.txt");
ifstream actin("test.act");

Intrinsc_Parameters ip;
vector<Feature_Point> fp;
int npoints;

void refine_triangulation(Mat P1, Mat P2, Mat Q1, Mat Q2, Mat &X);
void refine_Rt(Mat &R, Mat &t, Mat X1, Mat X2);

int main()
{
	
	ReadACTS(actin, ip, fp, npoints);
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
	refine_Rt(R, t, Q1h, Q2h);



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
	
//try to compare my result and opencv's result
	refine_triangulation(P1, P2, Q1c, Q2c, X2);
	
	getchar();

}

__inline void levmar_datagetP(Mat P1, Mat P2, double data[])
{
	for (int i = 0; i < 12; i++)
		data[i] = P1.at<double>(i);
	for (int i = 0; i < 12; i++)
		data[i + 15] = P2.at<double>(i);
}

void func(double *p, double *hx, int m, int n, void *adata)
{
	hx[0] = 0;
	hx[1] = 0;
	hx[2] = 0;
	double* data = (double*)adata;
	for (int i = 0; i < 2; i++)
	{
		hx[0] += data[i * 15 + 0] * p[0] - data[i * 15 + 12] + p[1] * data[i * 15 + 1] + p[2] * data[i * 15 + 2] + p[3] * data[i * 15 + 3];
		hx[1] += data[i * 15 + 4] * p[0] - data[i * 15 + 13] + p[1] * data[i * 15 + 5] + p[2] * data[i * 15 + 6] + p[3] * data[i * 15 + 6];
		hx[2] += data[i * 15 + 8] * p[0] - data[i * 15 + 14] + p[1] * data[i * 15 + 9] + p[2] * data[i * 15 + 10] + p[3] * data[i * 15 + 11];
	}

}
void refine_triangulation(Mat P1,Mat P2,Mat Q1,Mat Q2,Mat &X)
{
	int npoints = Q1.size().width;
	X = X.t();
	double data[30];
	levmar_datagetP(P1, P2, data);
	for (int i = 0; i < npoints; i++)
	{
		data[12] = Q1.at<double>(0, i);
		data[13] = Q1.at<double>(1, i);
		data[14] = 1;
		data[12 + 15] = Q2.at<double>(0, i);
		data[13 + 15] = Q2.at<double>(1, i);
		data[14 + 15] = 1;
// comments are used for testing the optimal result
//		double hx[3];
//		func(&((double*)(X.data))[i * 4], hx, 3, 3, data);
//		for (int j = 0; j < 3; j++)
//			cout << hx[j] << " ";
//		cout << endl; 
		dlevmar_dif(func, &((double*)(X.data))[i * 4], NULL, 3, 3, 1000, NULL, NULL, NULL, NULL, data);
//		func(&((double*)(X.data))[i * 4], hx, 3, 3, data);
//		for (int j = 0; j < 3; j++)
//			cout << hx[j] << " ";
//		cout << endl;

	}
}

void funcRt(double *p, double *hx, int m, int n, void *adata)
{
	static auto j = 0;
	j++;
	vector<Mat>* matrix = (vector<Mat>*)adata;
	Mat R = (*matrix)[0];
	Mat X1 = (*matrix)[1];

	Mat X2 = (*matrix)[2];
	double *data = (double*)adata;
	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	double t1, t2, t3;
	double k1, k2, k3, theta;
	t1 = p[0]; t2 = p[1]; t3 = p[2];
	k1 = p[3]; k2 = p[4]; k3 = p[5]; theta = p[6];
	r11 = R.at<double>(0); r12 = R.at<double>(1); r13 = R.at<double>(2); r21 = R.at<double>(3); r22 = R.at<double>(4); r23 = R.at<double>(5); r31 = R.at<double>(6); r32 = R.at<double>(7); r33 = R.at<double>(8);
	Mat T(3, 3, CV_64F);
	T.at<double>(0) = r21*(t2*(k1*sin(theta) - k2*k3*(cos(theta) - 1)) - t3*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1)) + r31*(t3*(k1*sin(theta) + k2*k3*(cos(theta) - 1)) + t2*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r11*(t2*(k2*sin(theta) + k1*k3*(cos(theta) - 1)) + t3*(k3*sin(theta) - k1*k2*(cos(theta) - 1)));
	T.at<double>(1) = r22*(t2*(k1*sin(theta) - k2*k3*(cos(theta) - 1)) - t3*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1)) + r32*(t3*(k1*sin(theta) + k2*k3*(cos(theta) - 1)) + t2*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r12*(t2*(k2*sin(theta) + k1*k3*(cos(theta) - 1)) + t3*(k3*sin(theta) - k1*k2*(cos(theta) - 1)));
	T.at<double>(2) = r23*(t2*(k1*sin(theta) - k2*k3*(cos(theta) - 1)) - t3*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1)) + r33*(t3*(k1*sin(theta) + k2*k3*(cos(theta) - 1)) + t2*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r13*(t2*(k2*sin(theta) + k1*k3*(cos(theta) - 1)) + t3*(k3*sin(theta) - k1*k2*(cos(theta) - 1)));
	T.at<double>(3) = r11*(t1*(k2*sin(theta) + k1*k3*(cos(theta) - 1)) + t3*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1)) + r31*(t3*(k2*sin(theta) - k1*k3*(cos(theta) - 1)) - t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r21*(t1*(k1*sin(theta) - k2*k3*(cos(theta) - 1)) + t3*(k3*sin(theta) + k1*k2*(cos(theta) - 1)));
	T.at<double>(4) = r12*(t1*(k2*sin(theta) + k1*k3*(cos(theta) - 1)) + t3*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1)) + r32*(t3*(k2*sin(theta) - k1*k3*(cos(theta) - 1)) - t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r22*(t1*(k1*sin(theta) - k2*k3*(cos(theta) - 1)) + t3*(k3*sin(theta) + k1*k2*(cos(theta) - 1)));
	T.at<double>(5) = r13*(t1*(k2*sin(theta) + k1*k3*(cos(theta) - 1)) + t3*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1)) + r33*(t3*(k2*sin(theta) - k1*k3*(cos(theta) - 1)) - t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r23*(t1*(k1*sin(theta) - k2*k3*(cos(theta) - 1)) + t3*(k3*sin(theta) + k1*k2*(cos(theta) - 1)));
	T.at<double>(6) = r11*(t1*(k3*sin(theta) - k1*k2*(cos(theta) - 1)) - t2*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1)) + r21*(t2*(k3*sin(theta) + k1*k2*(cos(theta) - 1)) + t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1)) - r31*(t1*(k1*sin(theta) + k2*k3*(cos(theta) - 1)) + t2*(k2*sin(theta) - k1*k3*(cos(theta) - 1)));
	T.at<double>(7) = r12*(t1*(k3*sin(theta) - k1*k2*(cos(theta) - 1)) - t2*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1)) + r22*(t2*(k3*sin(theta) + k1*k2*(cos(theta) - 1)) + t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1)) - r32*(t1*(k1*sin(theta) + k2*k3*(cos(theta) - 1)) + t2*(k2*sin(theta) - k1*k3*(cos(theta) - 1)));
	T.at<double>(8) = r13*(t1*(k3*sin(theta) - k1*k2*(cos(theta) - 1)) - t2*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1)) + r23*(t2*(k3*sin(theta) + k1*k2*(cos(theta) - 1)) + t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1)) - r33*(t1*(k1*sin(theta) + k2*k3*(cos(theta) - 1)) + t2*(k2*sin(theta) - k1*k3*(cos(theta) - 1)));
	Mat result;
	for (int i = 0; i < npoints; i++)
	{
		result = X2.row(i)*T*X1.row(i).t() + X1.row(i)*T.t()*X2.row(i).t();
		hx[i] = result.at<double>(0);
	}
	if (j == 24)
	{
		for (int i = 0; i < npoints; i++)
			cout << hx[i] << " ";
	}
}
void refine_Rt(Mat &R, Mat &t,Mat X1, Mat X2)
{

	vector<Mat> matrix(3);
	matrix[0] = R;
	matrix[1] = X1;
	matrix[2] = X2;
	double p[7];
	p[0] = t.at<double>(0);
	p[1] = t.at<double>(1);
	p[2] = t.at<double>(2);
	p[3] = 1;
	p[4] = 1;
	p[5] = 1;
	p[6] = 0;
	cout<<dlevmar_dif(funcRt, p, NULL, 7, npoints, 1000, NULL, NULL, NULL, NULL, &matrix);
	t.at<double>(0) = p[0];
	t.at<double>(1) = p[1];
	t.at<double>(2) = p[2];

	Mat dR(3, 3, CV_64F);
	double k1, k2, k3, theta;
	k1 = p[3];
	k2 = p[4];
	k3 = p[5];
	theta = p[6];
	dR.at<double>(0) = (cos(theta) - 1)*k2*k2 + (cos(theta) - 1)*k3*k3 + 1;
	dR.at<double>(1) = -k3*sin(theta) - k1*k2*(cos(theta) - 1);
	dR.at<double>(2) = k2*sin(theta) - k1*k3*(cos(theta) - 1);
	dR.at<double>(3) = k3*sin(theta) - k1*k2*(cos(theta) - 1);
	dR.at<double>(4) = (cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k3*k3 + 1;
	dR.at<double>(5) = -k1*sin(theta) - k2*k3*(cos(theta) - 1);
	dR.at<double>(6) = -k2*sin(theta) - k1*k3*(cos(theta) - 1);
	dR.at<double>(7) = k1*sin(theta) - k2*k3*(cos(theta) - 1);
	dR.at<double>(8) = (cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1;
	R = dR*R;
	
}













