#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <levmar.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
#define points 63


ifstream xin("pts1.txt");
ifstream yin("pts2.txt");
ofstream fout("H.txt");

vector<double> X(points * 3);
vector<double> Y(points * 3);
double singlecost(double*h, double*x, double*y);
double cost(double* h, vector<double>& x, vector<double>&y);
void DLT(Mat A, Mat& H);
void func(double *p, double *hx, int m, int n, void *adata);
void est_hx(double *h, double *hx, double *y);
int main()
{
	Mat img1, img2;
	img1 = imread("img1.jpg");
	img2 = imread("img2.jpg");

	for (int i = 0; i < points; ++i)
	{
		xin >> X[i * 3];
		xin >> X[i * 3 + 1];
		X[i * 3 + 2] = 1;
		yin >> Y[i * 3];
		yin >> Y[i * 3 + 1];
		Y[i * 3 + 2] = 1;
	}
	Mat A(points * 2, 9, CV_64F);
	Mat H(3, 3, CV_64F);
	DLT(A, H);
	Mat img3;
	H = H.reshape(1, 3);
	warpPerspective(img1, img3, H, img1.size());
	imshow("img1", img2);
	imshow("img2", img3);
	waitKey(0);
//	double hx[126];
//	func((double*)H.data, hx, 0, 126, NULL);
	cout<<dlevmar_dif(func, (double*)H.data, NULL, 8, 63*2, 2000, NULL, NULL, NULL, NULL, NULL);
	warpPerspective(img1, img3, H, img1.size());
	imshow("img3", img3);

	waitKey(0);

}

void func(double *p, double *hx, int m, int n, void *adata)
{
	for (int i = 0; i < n; i++)
	{
		double x, y, u, v;
		x = X[i/2 * 3 + 0];
		y = X[i/2 * 3 + 1];
		u = Y[i/2 * 3 + 0];
		v = Y[i/2* 3 + 1];
		if (i % 2)
			hx[i] = -p[0] * x - p[1] * y - p[2] + p[6] * u*x + p[7] * u*y +  u;
		else
			hx[i] = -p[3] * x - p[4] * y - p[5] + p[6] * v*x + p[7] * v*y +  v;
	}

}


void est_hx(double *h, double *hx, double *y)
{
	double x1 = h[0] * y[0] + h[1] * y[1] + h[2] * y[2];
	double x2 = h[3] * y[0] + h[4] * y[1] + h[5] * y[2];
	double x3 = h[6] * y[0] + h[7] * y[1] + h[8] * y[2];
	x1 /= x3;
	x2 /= x3;
	x3 = 1;
	hx[0] = x1;
	hx[1] = x2;
	hx[2] = x3;
}


double cost(double* h, vector<double>& x, vector<double>&y)
{
	double cost = 0;
	for (int i = 0; i < points; i++)
	{
		cost += singlecost(h, &x[i * 3], &y[i * 3]);
	}
	return cost;

}

double singlecost(double*h, double*x, double*y)
{
	double y1 = h[0] * y[0] + h[1] * y[1] + h[2] * y[2];
	double y2 = h[3] * y[0] + h[4] * y[1] + h[5] * y[2];
	double y3 = h[6] * y[0] + h[7] * y[1] + h[8] * y[2];
	y1 /= y3;
	y2 /= y3;
	y3 = 1;
	double x1 = x[0];
	double x2 = x[1];
	double x3 = x[2];
	return pow(x1 - y1, 2) + pow(x2 - y2, 2) + pow(x3 - y3, 2);
}


void DLT(Mat A,Mat& H)
{
	for (int i = 0; i < points; i++)
	{
		double x, y, u, v;
		x = X[i * 3 + 0];
		y = X[i * 3 + 1];
		u = Y[i * 3 + 0];
		v = Y[i * 3 + 1];
		((double*)(A.data))[i * 18 + 0] = -x;
		((double*)(A.data))[i * 18 + 1] = -y;
		((double*)(A.data))[i * 18 + 2] = -1;
		((double*)(A.data))[i * 18 + 3] = 0;
		((double*)(A.data))[i * 18 + 4] = 0;
		((double*)(A.data))[i * 18 + 5] = 0;
		((double*)(A.data))[i * 18 + 6] = u*x;
		((double*)(A.data))[i * 18 + 7] = u*y;
		((double*)(A.data))[i * 18 + 8] = u;
		((double*)(A.data))[i * 18 + 9] = 0;
		((double*)(A.data))[i * 18 + 10] = 0;
		((double*)(A.data))[i * 18 + 11] = 0;
		((double*)(A.data))[i * 18 + 12] = -x;
		((double*)(A.data))[i * 18 + 13] = -y;
		((double*)(A.data))[i * 18 + 14] = -1;
		((double*)(A.data))[i * 18 + 15] = v*x;
		((double*)(A.data))[i * 18 + 16] = v*y;
		((double*)(A.data))[i * 18 + 17] = v;
	}
	SVD svd;
	svd.solveZ(A, H);
	for (int i = 0; i < 8; i++)
		((double*)H.data)[i] /= ((double*)H.data)[8];
	((double*)H.data)[8] = 1;
}