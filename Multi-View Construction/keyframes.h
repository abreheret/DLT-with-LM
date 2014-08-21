#ifndef __KEYFRAMES_H__
#define __KEYFRAMES_H__
#include "..\Common\ACTS.h"
#include "..\Common\Feature-Points.h"






void getkeyframes(Feature_Points & fp, int start, int mincorrespondences, int framecnt, vector<int> &keyframes, int npoints);



#endif