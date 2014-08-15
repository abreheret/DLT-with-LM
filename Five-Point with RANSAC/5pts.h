#ifndef __5pts_H__
#define __5pts_H__

#include <iostream>
#include <opencv2\core\core.hpp>

using namespace std;
using namespace cv;


Mat compute_nullspace_basis(vector<Point2d> &lps, vector<Point2d> &rps, int n);
Mat compute_constraints(Mat &basis);
Mat compute_Gbasis(Mat &constraints);
Mat compute_action_matrix(Mat &Gbasis);
Mat compute_Ematrices_Gb(Mat &At, Mat &basis);

#endif