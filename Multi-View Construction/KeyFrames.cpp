#include "keyframes.h"



void getkeyframes(Feature_Points &fp, int start, int mincorrespondences, int framecnt, vector<int> &keyframes,int npoints)
{
	int min = start;
	keyframes.clear();
	keyframes.push_back(min);
	for (int i = min+1; i < framecnt; i++)
	{
		int cnt = 0;
		for (int j = 0; j < npoints; j++)
		{
			if (fp.getindex(min,j) && fp.getindex(i,j))
			{
				cnt++;
			}
		}
		if (cnt < mincorrespondences)
		{
			min = i - 1;
			keyframes.push_back(min);
		}
	}
}	