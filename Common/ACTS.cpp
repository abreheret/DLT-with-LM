#include "ACTS.h"
#include "Feature-Points.h"



void ReadACTS(ifstream &fin, Intrinsc_Parameters &params, Feature_Points &points, int &featurecnt, int &startframe, int &step, int &endframe)
{
	string s;
	for (int i = 0; i < 3; i++)
		getline(fin, s);
	//fin.readsome(&s[0], 6);
	//fin >> startframe;
	//fin.readsome(&s[0], 5);
	//fin >> step;
	//fin.readsome(&s[0], 4);
	//fin >> endframe;
	startframe = 0;
	endframe = 200;
	int frames = endframe - startframe + 1;
	while (getline(fin, s))
	{
		if (s.find("intrinsic parameter", 0) != s.npos)
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
		if (s.find("Feature Tracks") != s.npos)
		{

			fin >> featurecnt;
			points.resize(featurecnt,frames);
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
					fp.add(Point2d(x, y), frame);
					points.addindex(frame, i);
				}
				points.add(fp);
			}
			break;

		}
	}
}


void WriteACTS(ifstream &fin, ofstream& fout, Mat X)
{
	string s;
	fin.seekg(0);
	while (getline(fin, s))
	{
		if (s.find("Feature Tracks") == s.npos)
		{
			fout << s << endl;
		}
		else
		{
			fout << s << endl;
			break;
		}
	}
	int npoints;
	int a, b, c;
	double x, y, z, w;
	fin >> npoints;
	fout << npoints << endl;
	getline(fin, s);
	fout << s << endl;
	for (int i = 0; i < npoints; i++)
	{
		fin >> a >> b >> c;
		fout << a << '\t' << b << '\t' << c << '\t';
		x = X.at<double>(0, i);
		y = X.at<double>(1, i);
		z = X.at<double>(2, i);
		w = X.at<double>(3, i);
		x /= w; y /= w; z /= w;
		fout << x << '\t' << y << '\t' << z << endl;
		getline(fin, s);
		getline(fin, s);
		fout << s << endl << endl << endl;

	}
	while (getline(fin, s))
	{
		fout << s << endl;
	}

}


void calibrate_points(vector<Feature_Point>&points, Intrinsc_Parameters &params, int npoints)
{
	double focal = params.fx;
	double cx = params.cx;
	double cy = params.cy;

}