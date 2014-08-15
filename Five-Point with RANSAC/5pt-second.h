#ifndef __5PT_SECOND_H__
#define __5PT_SECOND_H__

#include <opencv2\core\core.hpp>
#include <time.h>
#include <opencv2\calib3d\calib3d.hpp>


using namespace cv;



int runKernel(vector<double>, vector<double>, OutputArray);
Mat calibrate_points_homography(Mat Q, double focal, Point2d pp);
Mat findEssentialMartix_RANSAC(Mat &, Mat &, double , Point2d , double , double , int );
void getCoeffMat(double *, double *);
__inline void getsamplepoints(Mat &, Mat &, vector<int>&, vector<double> &, vector<double>&);
int est_inliers(Mat, Mat &, Mat &, double);
int recoverPose(InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
	OutputArray _t, double focal, Point2d pp, InputOutputArray _mask);
void decomposeEssentialMat(InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t);
Mat triangulation(Mat &P1, Mat&P2, Mat&points1, Mat &points2);
Mat calibrate_points(Mat Q, double focal, Point2d pp);
#endif
