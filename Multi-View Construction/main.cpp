#include "../Common/ACTS.h"
#include "keyframes.h"
#include <opencv2\core\core.hpp>
#include <iostream>
#include <fstream>
#include "../Five-Point with RANSAC/5pt-second.h"
#include <sba.h>
#include <cvsba.h>

using namespace std;
using namespace cv;


ifstream actin("all.act");
ofstream fout("fout-multi.txt");
int npoints;
int startframe;
int endframe;
int step;

void proj(int j, int i, double *aj, double *bi, double *xij, void *adata);
void refineXfromC(Mat X, Mat P1, Mat P2, vector<int> index, Mat Qh1, Mat Qh2);

int main()
{
	Intrinsc_Parameters ip;
	Feature_Points fp;
	vector<int> keyframes;
	vector<Point3d> currentpoints3D;
	vector<vector<Point2d>> currentpoints2D;
	vector<int> currentpointsindex;
	vector<int> newaddedpoints;
	vector<Mat> IntrinscMatrix, distCoeffs;
	Mat Intrinc = Mat::eye(3, 3, CV_64F);

	Mat zero = Mat(5, 1, CV_64FC1, cv::Scalar::all(0));
	distCoeffs.push_back(zero);
	ReadACTS(actin, ip, fp, npoints, startframe, endframe, step);
	Intrinc.at<double>(0) = ip.fx;
	Intrinc.at<double>(4) = ip.fy;
	Intrinc.at<double>(2) = ip.cx;
	Intrinc.at<double>(5) = ip.cy;
	IntrinscMatrix.push_back(Intrinc);
	startframe = 0; endframe = 200;
	getkeyframes(fp, startframe, 100, endframe, keyframes, npoints);
	cout << "keyframes:" << endl;
	for (auto i : keyframes)
		cout << i << " ";
	cout << endl;
	vector<int> matchpoints;

	vector<Mat> Rs(keyframes.size());
	vector<Mat> ts(keyframes.size());

	Rs.clear();
	ts.clear();

	Rs.push_back(Mat::eye(3, 3, CV_64F));
	ts.push_back(Mat::zeros(3, 1, CV_64F));
	for (int frame = 1; frame < keyframes.size(); frame++)
	{
		cout << "frames:" << frame + 1 << '\t';
		newaddedpoints.clear();
		for (int i = 0; i < npoints; i++)
		{
			if (fp.getindex(keyframes[frame-1], i) && fp.getindex(keyframes[frame], i))
			{
				matchpoints.push_back(i);
			}
		}
		int matchcnt = matchpoints.size();
		Mat Q1(matchcnt, 2, CV_64F), Q2(matchcnt, 2, CV_64F);
		for (int i = 0; i < matchcnt; i++)
		{
			if (!binary_search(currentpointsindex.begin(), currentpointsindex.end(), matchpoints[i]))
			{
				currentpointsindex.push_back(matchpoints[i]);
				newaddedpoints.push_back(i);
			}
			Q1.at<double>(i, 0) = fp[matchpoints[i]].get_point(keyframes[0]).x;
			Q1.at<double>(i, 1) = fp[matchpoints[i]].get_point(keyframes[0]).y;
			Q2.at<double>(i, 0) = fp[matchpoints[i]].get_point(keyframes[1]).x;
			Q2.at<double>(i, 1) = fp[matchpoints[i]].get_point(keyframes[1]).y;
		}

		cout << "points:" << currentpointsindex.size() << endl;

		Mat EM;//essential matrix
		cout << "computing EM"<<'\t';
		EM = findEssentialMartix_RANSAC(Q1, Q2, ip.fx, Point2d(ip.cx, ip.cy), 0.1, 0.5, 100);
		
		Mat R, t, _mask;//Rotation matrix and translation matrix
		cout << "computing R and t" << '\t';
		recoverPose(EM, Q1, Q2, R, t, ip.fx, Point2d(ip.cx, ip.cy), _mask);

		//optimize the P=(R|t) using the X
		Mat Q1h = calibrate_points_homography(Q1, ip.fx, Point2d(ip.cx, ip.cy));
		Mat Q2h = calibrate_points_homography(Q2, ip.fx, Point2d(ip.cx, ip.cy));
		cout << "refine the Rt"<<'\t';
		refine_Rt(R, t, Q1h, Q2h);

		//construct the camera matrix
		Mat P1 (3,4,CV_64F), P2(3, 4, CV_64F);
		Mat R1 = Rs[Rs.size() - 1]; Mat t1 = ts[ts.size() - 1];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
				P1.at<double>(i, j) = R1.at<double>(i, j);
		}
		for (int i = 0; i < 3; i++)
			P1.at<double>(i, 3) = t1.at<double>(i, 0);
		Mat R2 = R1*R;
		Rs.push_back(R2);
		Mat t2 = t1 + t;
		ts.push_back(t2);

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
				P2.at<double>(i, j) = R2.at<double>(i, j);
		}
		for (int i = 0; i < 3; i++)
			P2.at<double>(i, 3) = t2.at<double>(i, 0);

		Mat X;
		//calibrate correspondences with focal and principal point
		Mat Q1c = calibrate_points(Q1, ip.fx, Point2d(ip.cx, ip.cy));
		Mat Q2c = calibrate_points(Q2, ip.fx, Point2d(ip.cx, ip.cy));
		Q1c = Q1c.t();
		Q2c = Q2c.t();

		//triangulate 3D point with my implementation
		X = triangulation(P1, P2, Q1c, Q2c);

		//optimize the X using the P above
		cout << "refine X";
		refineXfromC(X, P1,P2, newaddedpoints, Q1h,Q2h);


		for (int i = 0; i < newaddedpoints.size(); i++)
		{
			currentpoints3D.push_back(Point3d(X.at<double>(0, newaddedpoints[i]), X.at<double>(1, newaddedpoints[i]), X.at<double>(2, newaddedpoints[i])));
		}
		currentpoints2D.resize(frame + 1);
		for (int img = 0; img < frame; img++)
		{
			for (int i = 0; i < newaddedpoints.size(); i++)
			{
				int nowimg = currentpointsindex[currentpointsindex.size() - newaddedpoints.size() + i];
				if (fp.getindex(keyframes[img], nowimg))
				{
					currentpoints2D[img].push_back(Point2d(fp[nowimg].get_point(keyframes[img])));
				}
				else
				{
					currentpoints2D[img].push_back(Point2d(0, 0));
				}
			}
		}
		for (int i = 0; i < currentpointsindex.size(); i++)
		{
			if (fp.getindex(keyframes[frame], currentpointsindex[i]))
			{
				currentpoints2D[frame].push_back(Point2d(fp[currentpointsindex[i]].get_point(keyframes[frame])));
			}
			else
			{
				currentpoints2D[frame].push_back(Point2d(0, 0));
			}
		}
		vector<vector<int>> visiability;
		visiability.resize(frame+1);
	
		for (int i = 0; i <=frame; i++)
		{
			visiability[i].resize(currentpointsindex.size());
			for (int j = 0; j < currentpointsindex.size(); j++)
			{
				if (fp.getindex(keyframes[i], currentpointsindex[j]))
					visiability[i][j] = 1;
				else
					visiability[i][j] = 0;
			}
		}


		IntrinscMatrix.push_back(Intrinc);

		distCoeffs.push_back(zero);
		int ncon = currentpointsindex.size() - newaddedpoints.size();
		int mcon = frame;
		cvsba::Sba sba;
		cout << "sba!";
		sba.run(currentpoints3D, currentpoints2D, visiability, IntrinscMatrix, Rs, ts, distCoeffs,mcon,ncon);
		std::cout << "Initial error=" << sba.getInitialReprjError() << ". Final error=" << sba.getFinalReprjError() << std::endl;
	}
	////construct vmask for bundle adjustment
	////vector<char> vmask(2 * npoints);
	////for (int i = 0; i < npoints; i++)
	////{
	////	vmask[2 * i] = fp.index[keyframes[0]][i];
	////	vmask[2 * i + 1] = fp.index[keyframes[1]][i];
	////}
	//vector<char> vmask(2 * matchcnt);
	//for (int i = 0; i < 2 * matchcnt; i++)
	//{
	//	vmask[i] = 1;
	//}
	////define some parameter for bundle adjustment
	//int n = npoints, m = 2; int cnp = 6, pnp = 3, mnp = 2;
	////construct p for bundle adjustment
	//vector<double> p(m * cnp + n * pnp);
	//for (int i = 0; i < m; i++)
	//{
	//	p[i*cnp + 0] = 1;
	//	p[i*cnp + 1] = 1;
	//	p[i*cnp + 2] = 0;
	//	p[i*cnp + 3] = 0;
	//	p[i*cnp + 4] = 0;
	//	p[i*cnp + 5] = 0;
	//}
	//p[1 * 6 + 3] = t.at<double>(0);
	//p[1 * 6 + 4] = t.at<double>(1);
	//p[1 * 6 + 5] = t.at<double>(2);
	//
	//for (int i = 0; i < matchcnt; i++)
	//{
	//		p[m*cnp + i*pnp + 0] = X.at<double>(0, i);
	//		p[m*cnp + i*pnp + 1] = X.at<double>(1, i);
	//		p[m*cnp + i*pnp + 2] = X.at<double>(2, i);

	//}
	////constuct x for bundle adjustment
	//vector<double> x;
	////for (int i = 0; i < npoints; i++)
	////{
	////	for (int j = 0; j < endframe - startframe + 1; j++)
	////	{
	////		if (fp.getindex(j, i))
	////		{
	////			x.push_back(fp[i].get_point(j).x);
	////			x.push_back(fp[i].get_point(j).y);
	////		}
	////	}
	////}
	//for (int i = 0; i < matchcnt; i++)
	//{
	//	x.push_back(Q1c.at<double>(0, i));
	//	x.push_back(Q1c.at<double>(1, i));
	//	x.push_back(Q2c.at<double>(0, i));
	//	x.push_back(Q2c.at<double>(1, i));
	//}
	//vector<Mat> adata;
	//adata.push_back(R);
	//sba_motstr_levmar(matchcnt, 0, 2, 1, &vmask[0], &p[0], cnp, pnp, &x[0], NULL, mnp, proj, NULL, &adata[0], 1000, NULL, NULL, NULL);
	//
	getchar();
}


//void proj(int j, int i, double *aj, double *bi, double *xij, void *adata)
//{
//	vector<Mat> * data = (vector<Mat>*) adata;
//	Mat R = (*data)[j - 1];
//	double k1 = aj[0], k2 = aj[1], theta = aj[2], t1 = aj[3], t2 = aj[4], t3 = aj[5];
//	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
//	r11 = R.at<double>(0); r12 = R.at<double>(1); r13 = R.at<double>(2); r21 = R.at<double>(3); r22 = R.at<double>(4); r23 = R.at<double>(5); r31 = R.at<double>(6); r32 = R.at<double>(7); r33 = R.at<double>(8);
//	Mat T(3, 3, CV_64F);
//	T.at<double>(0) = r21*(t2*(k1*sin(theta) - k2 * 1 * (cos(theta) - 1)) - t3*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1) * 1 * 1 + 1)) + r31*(t3*(k1*sin(theta) + k2 * 1 * (cos(theta) - 1)) + t2*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r11*(t2*(k2*sin(theta) + k1 * 1 * (cos(theta) - 1)) + t3*(1 * sin(theta) - k1*k2*(cos(theta) - 1)));
//	T.at<double>(1) = r22*(t2*(k1*sin(theta) - k2 * 1 * (cos(theta) - 1)) - t3*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1) * 1 * 1 + 1)) + r32*(t3*(k1*sin(theta) + k2 * 1 * (cos(theta) - 1)) + t2*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r12*(t2*(k2*sin(theta) + k1 * 1 * (cos(theta) - 1)) + t3*(1 * sin(theta) - k1*k2*(cos(theta) - 1)));
//	T.at<double>(2) = r23*(t2*(k1*sin(theta) - k2 * 1 * (cos(theta) - 1)) - t3*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1) * 1 * 1 + 1)) + r33*(t3*(k1*sin(theta) + k2 * 1 * (cos(theta) - 1)) + t2*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r13*(t2*(k2*sin(theta) + k1 * 1 * (cos(theta) - 1)) + t3*(1 * sin(theta) - k1*k2*(cos(theta) - 1)));
//	T.at<double>(3) = r11*(t1*(k2*sin(theta) + k1 * 1 * (cos(theta) - 1)) + t3*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1) * 1 * 1 + 1)) + r31*(t3*(k2*sin(theta) - k1 * 1 * (cos(theta) - 1)) - t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r21*(t1*(k1*sin(theta) - k2 * 1 * (cos(theta) - 1)) + t3*(1 * sin(theta) + k1*k2*(cos(theta) - 1)));
//	T.at<double>(4) = r12*(t1*(k2*sin(theta) + k1 * 1 * (cos(theta) - 1)) + t3*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1) * 1 * 1 + 1)) + r32*(t3*(k2*sin(theta) - k1 * 1 * (cos(theta) - 1)) - t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r22*(t1*(k1*sin(theta) - k2 * 1 * (cos(theta) - 1)) + t3*(1 * sin(theta) + k1*k2*(cos(theta) - 1)));
//	T.at<double>(5) = r13*(t1*(k2*sin(theta) + k1 * 1 * (cos(theta) - 1)) + t3*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1) * 1 * 1 + 1)) + r33*(t3*(k2*sin(theta) - k1 * 1 * (cos(theta) - 1)) - t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1)*k2*k2 + 1)) - r23*(t1*(k1*sin(theta) - k2 * 1 * (cos(theta) - 1)) + t3*(1 * sin(theta) + k1*k2*(cos(theta) - 1)));
//	T.at<double>(6) = r11*(t1*(1 * sin(theta) - k1*k2*(cos(theta) - 1)) - t2*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1) * 1 * 1 + 1)) + r21*(t2*(1 * sin(theta) + k1*k2*(cos(theta) - 1)) + t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1) * 1 * 1 + 1)) - r31*(t1*(k1*sin(theta) + k2 * 1 * (cos(theta) - 1)) + t2*(k2*sin(theta) - k1 * 1 * (cos(theta) - 1)));
//	T.at<double>(7) = r12*(t1*(1 * sin(theta) - k1*k2*(cos(theta) - 1)) - t2*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1) * 1 * 1 + 1)) + r22*(t2*(1 * sin(theta) + k1*k2*(cos(theta) - 1)) + t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1) * 1 * 1 + 1)) - r32*(t1*(k1*sin(theta) + k2 * 1 * (cos(theta) - 1)) + t2*(k2*sin(theta) - k1 * 1 * (cos(theta) - 1)));
//	T.at<double>(8) = r13*(t1*(1 * sin(theta) - k1*k2*(cos(theta) - 1)) - t2*((cos(theta) - 1)*k2*k2 + (cos(theta) - 1) * 1 * 1 + 1)) + r23*(t2*(1 * sin(theta) + k1*k2*(cos(theta) - 1)) + t1*((cos(theta) - 1)*k1*k1 + (cos(theta) - 1) * 1 * 1 + 1)) - r33*(t1*(k1*sin(theta) + k2 * 1 * (cos(theta) - 1)) + t2*(k2*sin(theta) - k1 * 1 * (cos(theta) - 1)));
//	
//	Mat result = T.row(0)*bi[0] + T.row(1)*bi[1] + T.row(2)*bi[2];
//	result.at<double>(0) /= result.at<double>(2);
//	result.at<double>(1) /= result.at<double>(2);
//	xij[0] = result.at<double>(0);
//	xij[1] = result.at<double>(1);
//}


void funcXfromC(double *p, double *hx, int m, int n, void *adata)
{
	Vector<Mat> * matrix = (Vector<Mat>*)adata;
	Mat P1 = (*matrix)[0];
	Mat P2 = (*matrix)[1];
	Mat index = (*matrix)[2];
	Mat Qh1 = (*matrix)[3];
	Mat Qh2 = (*matrix)[4];
	int npoints = index.size().height;
	Mat X(4, 1, CV_64F);
	for (int i = 0; i < npoints; i++)
	{
		X.at<double>(0) = p[3 * i];
		X.at<double>(1) = p[3 * i + 1];
		X.at<double>(2) = p[3 * i + 2];
		X.at<double>(3) = 1;
		for (int j = 0; j < 3; j++)
		{
			Mat result = P1.row(j)*X;
			hx[3 * i + j] = result.at<double>(0) - Qh1.at<double>(index.at<int>(i), j);
			result = P2.row(j)*X;
			hx[3 * i + j] += result.at<double>(0) - Qh2.at<double>(index.at<int>(i), j);
		}
	}
}

void refineXfromC(Mat X, Mat P1,Mat P2, vector<int> index, Mat Qh1,Mat Qh2)
{
	int npoints = index.size();
	vector<double> p(npoints * 3);
	for (int i = 0; i < npoints; i++)
	{
		for (int j = 0; j < 3; j++)
			p[i * 3 + j] = X.at<double>(j, index[i]);
	}
	Vector<Mat> matrix;
	matrix.push_back(P1);
	matrix.push_back(P2);
	matrix.push_back(Mat(index));
	matrix.push_back(Qh1);
	matrix.push_back(Qh2);
	dlevmar_dif(funcXfromC, &p[0], NULL, 3 * npoints, 3 * npoints, 1000, NULL, NULL, NULL, NULL, &matrix);
	for (int i = 0; i < npoints; i++)
	{
		for (int j = 0; j < 3; j++)
			X.at<double>(j, index[i]) = p[i * 3 + j];
	}
}








