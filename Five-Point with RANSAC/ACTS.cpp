#include "ACTS.h"



void ReadACTS(ifstream &fin, Intrinsc_Parameters &params, vector<Feature_Point> &points, int &featurecnt)
{
	string s;
	while (getline(fin, s))
	{
		if (s.find("intrinsic parameter",0)!=s.npos)
		{
			fin >> params.fx;
			fin >> params.fy;
			fin >> params.cx;
			fin >> params.cy;
			break;
		}
	}
	while (getline(fin, s))
	{
		if (s.find("Feature Tracks")!=s.npos)
		{
			featurecnt;
			fin >> featurecnt;
			points.resize(featurecnt);
			points.clear();
			getline(fin, s);
			for (int i = 0; i < featurecnt; i++)
			{
				Feature_Point fp;
				int cnt;
				fin >> cnt;
				getline(fin, s);
				for (int j = 0; j < cnt; j++)
				{
					int frame; double x; double y;
					fin >> frame >> x >> y;
					fp.add(Point3d(x, y, frame + 0.1));
				}
				points.push_back(fp);
			}
			break;

		}
	}
}