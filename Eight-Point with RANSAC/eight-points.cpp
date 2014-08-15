#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <time.h> 
#include <levmar.h>
#include "ACTS.h"

using namespace cv;
using namespace std;

ifstream xin("pts1.txt");
ifstream yin("pts2.txt");
ifstream fin("test.act");
int points;

vector<double> X(points * 3);
vector<double> Y(points * 3);
Intrinsc_Parameters ip;
vector<Feature_Point> fp;

void read_points();
Mat eight_point(vector<int> nums);
int est_inliers(Mat E, double thresh);
int RANSAC(double thresh, double inlierratio, int iternum, Mat &E);
void func(double *p, double *hx, int m, int n, void *adata);
void jacf(double *p, double *j, int m, int n, void *adata);

//#define Jacobian

int main()
{

	ReadACTS(fin, ip, fp,points);

//	read_points();
	vector<int> nums(8);
	for (int i = 1; i < 9; i++)
		nums[i-1] = i;
	Mat E =eight_point(nums);
	cout << E << endl;
	RANSAC(0.01, 0.5, 1000, E);
	cout << E << endl;
	getchar();

}


//void read_points()
//{
//	for (int i = 0; i < points; ++i)
//	{
//		xin >> X[i * 3];
//		xin >> X[i * 3 + 1];
//		X[i * 3 + 2] = 1;
//		yin >> Y[i * 3];
//		yin >> Y[i * 3 + 1];
//		Y[i * 3 + 2] = 1;
//	}
//}
//
//Mat eight_point(vector<int> nums)
//{
//	vector<double> A(72);
//	for (int i = 0; i < 8; i++)
//	{
//		double x, y, u, v;
//		x = X[3 * nums[i]];
//		y = X[3 * nums[i] + 1];
//		u = Y[3 * nums[i]];
//		v = Y[3 * nums[i] + 1];
//		A[9 * i + 0] = u*x;
//		A[9 * i + 1] = u*y;
//		A[9 * i + 2] = u;
//		A[9 * i + 3] = v*x;
//		A[9 * i + 4] = v*y;
//		A[9 * i + 5] = v;
//		A[9 * i + 6] = x;
//		A[9 * i + 7] = y;
//		A[9 * i + 8] = 1;
//	}
//	Mat e(A);
//	e = e.reshape(1, 8);
//	SVD svd;
//	Mat E;
//	svd.solveZ(e, E);
//	return E.reshape(1, 3);
//}
//
//int est_inliers(Mat E,double thresh)
//{
//	int inliers = 0;
//	double* e = (double*)E.data;
//	for (int i = 0; i < points; i++)
//	{
//		double x, y, u, v;
//		x = X[3 * i];
//		y = X[3 * i+1];
//		u = Y[3 * i];
//		v = Y[3 * i + 1];
//		double r = u*x*e[0] + u*y*e[1] + u*e[2] + v*x*e[3] + v*y*e[4] + v*e[5] + x*e[6] + y*e[7] + e[8];
//		if (abs(r) < thresh)
//			inliers++;
//	}
//	return inliers;
//
//}

Mat eight_point(vector<int> nums)
{
	vector<double> A(72);
	for (int i = 0; i < 8; i++)
	{
		double x, y, u, v;
		x = fp[nums[i]].get_point(0).x;
		y = fp[nums[i]].get_point(0).y;
		u = fp[nums[i]].get_point(1).x;
		v = fp[nums[i]].get_point(1).y;
		A[9 * i + 0] = u*x;
		A[9 * i + 1] = u*y;
		A[9 * i + 2] = u;
		A[9 * i + 3] = v*x;
		A[9 * i + 4] = v*y;
		A[9 * i + 5] = v;
		A[9 * i + 6] = x;
		A[9 * i + 7] = y;
		A[9 * i + 8] = 1;
	}
	Mat e(A);
	e = e.reshape(1, 8);
	SVD svd;
	Mat E;
	svd.solveZ(e, E);
	return E.reshape(1, 3);
}

int RANSAC(double thresh,double inlierratio,int iternum,Mat &E)
{
	vector<int> sample(8);
	vector<int> maxsample(8);
	vector<int> inliers;
	inliers.clear();
	int maxinlier=0;
	srand((unsigned)time(NULL));
	for (int i = 0; i < iternum; i++)
	{
		for (int i = 0; i < 8; i++)
		{
			int tmp = rand() % points;
			int resample = 0;
			for (int j = 0; j<i; j++)
			{
				if (sample[j] == tmp)
				{
					resample = 1;
					break;
				}
			}
			if (resample != 0)
			{
				i--; continue;
			}
			else
			{
				sample[i] = tmp;
			}
		}
		E = eight_point(sample);
		int inliercnt = est_inliers(E, thresh);
		if (inliercnt>maxinlier)
		{
			for (int i = 0; i < 8; i++)
				maxsample[i] = sample[i];
			maxinlier = inliercnt;
		}
	}
	cout << maxinlier<<endl;
	for (int i = 0; i<8; i++)
		cout << maxsample[i] << " ";
	cout << endl;
	if ((maxinlier + 0.0) / points>inlierratio)
	{
		//least square 
		E = eight_point(maxsample);
		double* e = (double*)E.data;
		for (int i = 0; i < points; i++)
		{
			double x, y, u, v;
			x = fp[i].get_point(0).x;
			y = fp[i].get_point(0).y;
			u = fp[i].get_point(1).x;
			v = fp[i].get_point(1).y;
			double r = u*x*e[0] + u*y*e[1] + u*e[2] + v*x*e[3] + v*y*e[4] + v*e[5] + x*e[6] + y*e[7] + e[8];
			if (abs(r) < thresh)
				inliers.push_back(i);
		}
#ifdef Jacobian
		cout << dlevmar_der(func,jacf, e, NULL, 9, maxinlier, 2000, NULL, NULL, NULL, NULL, (void*)&inliers[0]) << endl;
#else
		cout << dlevmar_dif(func, e, NULL, 8, maxinlier, 2000, NULL, NULL, NULL, NULL, (void*)&inliers[0]) << endl;
#endif
		//cout << norm(E, NORM_L2)<<endl;
		//Mat U, V, D;
		//SVD svd;
		//svd.compute(E, V, U, D,SVD::FULL_UV);
		//cout << U << endl;
		//cout << V << endl;
		//cout << D << endl;
		//V.at<double>(0, 0) = 1;
		//V.at<double>(1, 1) = 1;
		//V.at<double>(2, 2) = 0;
		//E = U*V*D;
		//cout << E << endl;
	}
	
	else
	{
		return -1;
	}
}


int est_inliers(Mat E, double thresh)
{
	int inliers = 0;
	double* e = (double*)E.data;
	for (int i = 0; i < points; i++)
	{
		double x, y, u, v;
		x = fp[i].get_point(0).x;
		y = fp[i].get_point(0).y;
		u = fp[i].get_point(1).x;
		v = fp[i].get_point(1).y;
		double r = u*x*e[0] + u*y*e[1] + u*e[2] + v*x*e[3] + v*y*e[4] + v*e[5] + x*e[6] + y*e[7] + e[8];
		if (abs(r) < thresh)
			inliers++;
	}
	return inliers;

}

void func(double *p, double *hx, int m, int n, void *adata)
{
	int* inliers = (int*)adata;
	for (int i = 0; i < n; i++)
	{
		double x, y, u, v;
		x = fp[inliers[i]].get_point(0).x;
		y = fp[inliers[i]].get_point(0).y;
		u = fp[inliers[i]].get_point(1).x;
		v = fp[inliers[i]].get_point(1).y;
		double r = u*x*p[0] + u*y*p[1] + u*p[2] + v*x*p[3] + v*y*p[4] + v*p[5] + x*p[6] + y*p[7] + p[8];
		hx[i] = r;
	}
}


void jacf(double *p, double *j, int m, int n, void *adata)
{
	int* inliers = (int*)adata;
	for (int i = 0; i < n; i++)
	{
		double x, y, u, v;
		x = fp[inliers[i]].get_point(0).x;
		y = fp[inliers[i]].get_point(0).y;
		u = fp[inliers[i]].get_point(1).x;
		v = fp[inliers[i]].get_point(1).y;
		j[i * 9] = x*u;
		j[i * 9 + 1] = u*y;
		j[i * 9 + 2] = u;
		j[i *9 + 3] = v*x;
		j[i * 9 + 4] = v*y;
		j[i * 9 + 5] = v;
		j[i * 9 + 6] = x;
		j[i * 9 + 7] = y;
		j[i * 9 + 8] = 1;
	}
}
