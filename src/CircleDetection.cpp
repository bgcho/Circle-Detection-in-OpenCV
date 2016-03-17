//============================================================================
// Name        : CircleDetection.cpp
// Author      : Bill Byung Gu Cho
// Version     : 1.0
// Copyright   : Free For All
// Description : Circle detection using hough transform in C++, Ansi-style
//============================================================================


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace cv;
using namespace std;



// CircleGroup class
class CircleGroup{
private:
	Mat circles;
	int no_circles;

public:
	void sortCircles(Mat, int, int);
	Mat getCircles(){return this->circles;}
	Mat getCircleCenters();
	Mat getCircleRadii();
	int getNoCircles(){ return no_circles; }


	CircleGroup(Mat circles_in){
		this->no_circles = circles_in.rows;
		this->sortCircles(circles_in, this->no_circles, 0);
	}

	CircleGroup(Mat circles_in, int no_limit, int opt){

		this->no_circles = min(no_limit, circles_in.rows);
		this->sortCircles(circles_in, this->no_circles, opt);
	}

	~CircleGroup(){}
};

Mat permGenerator(int n, int k);
Mat shuffleRows(const Mat &matrix);
Mat removeOverlapCircles(Mat circles);
Mat getAffineMat(Mat src_pts, Mat dst_pts);
Mat optimalAffineMat(CircleGroup* src, CircleGroup* dst, float th, unsigned long int max_numLoop);




int main(int argc, char** argv)
{
	// Parameters for circle detection
	int filt_size = 3;
	float filt_std = 2;
	float dp = 1;			// Inverse ratio of resolution
	float edge_th = 100;	// Upper threshold for canny edge detector
	float cen_th = 12;		// Center detection threshold
	float min_rad = 1;		// minimum detectable radius
	float max_rad = 30;		// maximum detectable radius


	// Parameters for registration
	float th = 1;			// objective function threshold
	unsigned long int max_numLoop = 50000;		// Number of maximum loops
	int opt = 0;			// circle picking option
							// opt=0: pick the larger circles
							// opt=1: pick the circles far from the mean center

	Mat src, src_gray, dst, dst_gray;

	// Read the image
	string arg1(argv[1]);
	string src_file = "../images/pair" + arg1 + "/figure_A.bmp" ;
	string dst_file = "../images/pair" + arg1 + "/figure_B.bmp" ;

	src = imread(src_file, 1);
	dst = imread(dst_file, 1);

	if( !src.data || !dst.data ){ return -1; }

	// Convert it to gray
	cvtColor( src, src_gray, CV_BGR2GRAY );
	cvtColor( dst, dst_gray, CV_BGR2GRAY );

	// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray, src_gray, Size(filt_size,filt_size), filt_std, filt_std );
	GaussianBlur( dst_gray, dst_gray, Size(filt_size,filt_size), filt_std, filt_std );

	// Apply the Hough Transform to find the circles
	vector<Vec3f> src_circles_vec;
	vector<Vec3f> dst_circles_vec;
	HoughCircles( src_gray, src_circles_vec, CV_HOUGH_GRADIENT, dp, src_gray.rows/32, edge_th, cen_th, min_rad, max_rad );
	HoughCircles( dst_gray, dst_circles_vec, CV_HOUGH_GRADIENT, dp, dst_gray.rows/32, edge_th, cen_th, min_rad, max_rad );


	// Convert STL vectors to CV Mat objects
	Mat src_circles_Mat(src_circles_vec.size(), 3, CV_32FC1);
	Mat dst_circles_Mat(dst_circles_vec.size(), 3, CV_32FC1);

	for (size_t i = 0 ; i < src_circles_vec.size() ; ++i ){
		for(int j = 0; j < 3 ; ++j){
			src_circles_Mat.at<float>(i,j) = src_circles_vec.at(i)[j];
		}
	}

	for (size_t i = 0 ; i < dst_circles_vec.size() ; ++i ){
		for(int j = 0; j < 3 ; ++j){
			dst_circles_Mat.at<float>(i,j) = dst_circles_vec.at(i)[j];
		}
	}

	// Remove the overlapping circles
	Mat src_circles;
	Mat dst_circles;

	src_circles =  removeOverlapCircles(src_circles_Mat);
	dst_circles =  removeOverlapCircles(dst_circles_Mat);

	CircleGroup src_ft(src_circles, 40, opt);
	CircleGroup dst_ft(dst_circles, 10, opt);

	Mat H = optimalAffineMat(&src_ft, &dst_ft, th, max_numLoop);
	H.pop_back(1);


	Mat warped_img = Mat::zeros( dst_gray.rows, dst_gray.cols, dst_gray.type());

	// warp the source image
	warpAffine(src, warped_img, H, Size(warped_img.cols, warped_img.rows));

	// Draw the circles detected
	int src_r, dst_r;
	for( int i = 0; i < src_circles.rows ; ++i ){
		Point src_center(cvRound(src_circles.at<float>(i,0)), cvRound(src_circles.at<float>(i,1)));
		src_r= cvRound(src_circles.at<float>(i,2));
		circle( src, src_center, 3, Scalar(0,255,0), -1, 8, 0 );
		circle( src, src_center, src_r, Scalar(0,0,255), 2, 8, 0 );
	}

	for( int i = 0; i < dst_circles.rows; ++i ){
        Point dst_center(cvRound(dst_circles.at<float>(i,0)), cvRound(dst_circles.at<float>(i,1)));
        dst_r = cvRound(dst_circles.at<float>(i,2));
        circle( dst, dst_center, 3, Scalar(0,255,0), -1, 8, 0 );
        circle( dst, dst_center, dst_r, Scalar(0,0,255), 2, 8, 0 );
	}

	// Show the results
	namedWindow( "Image A", CV_WINDOW_AUTOSIZE );
	imshow( "Image A", src );

	namedWindow( "Image B", CV_WINDOW_AUTOSIZE );
	imshow( "Image B", dst );

	namedWindow( "Warped Image A", CV_WINDOW_AUTOSIZE );
	imshow( "Warped Image A", warped_img);

	waitKey(0);

	char dosave;
	struct stat st = {0};
	string save_dir = "../result_images/pair";
	save_dir = save_dir + arg1;
	char *save_dir1 = new char[save_dir.size() + 1];
	memcpy(save_dir1, save_dir.c_str(), save_dir.size() + 1);
	cout << "Save the images ? (y/n): ";
	cin >> dosave;
	if (dosave == 'y'){
		if (stat(save_dir1, &st)==-1){
			mkdir(save_dir1, 0700);
		}

		string src_save_name;
		string dst_save_name;
		string warp_save_name;
		src_save_name = save_dir + "/figure_A_circle.bmp";
		dst_save_name = save_dir + "/figure_B_circle.bmp";
		warp_save_name = save_dir + "/figure_A_warped.bmp";

		imwrite(src_save_name, src);
		imwrite(dst_save_name, dst);
		imwrite(warp_save_name, warped_img);
	}


	return 0;


}

// Sort the circles from further to closer to the mean center of the circles
void CircleGroup::sortCircles(Mat circles_in, int no_limit, int opt)
{
	Mat sorted_circles(circles_in.rows, circles_in.cols, circles_in.type());
	Mat radii = circles_in.col(2);
	Mat sort_indx;
	switch (opt){
	case 0:
		// Get the big circles
		sortIdx(radii, sort_indx, CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING);
		for (int j = 0 ; j < sort_indx.rows ; ++j){
			sorted_circles.at<float>(j,0) = circles_in.at<float>(sort_indx.at<int>(j,0),0);
			sorted_circles.at<float>(j,1) = circles_in.at<float>(sort_indx.at<int>(j,0),1);
			sorted_circles.at<float>(j,2) = circles_in.at<float>(sort_indx.at<int>(j,0),2);
		}
		break;
	case 1:
		// Get the circles far from the mean center position
		Mat row_mean;
		reduce(circles_in, row_mean, 0, CV_REDUCE_AVG);
		Mat mean_x = repeat(row_mean.col(0),circles_in.rows, 1);
		Mat mean_y = repeat(row_mean.col(1),circles_in.rows, 1);
		Mat center_x = circles_in.col(0);
		Mat center_y = circles_in.col(1);
		Mat diff_x = center_x - mean_x;
		Mat diff_y = center_y - mean_y;
		Mat distance;
		Mat sort_indx;
		sqrt(diff_x.mul(diff_x)+diff_y.mul(diff_y), distance);
		sortIdx(distance, sort_indx, CV_SORT_EVERY_COLUMN+CV_SORT_DESCENDING);
		for (int i = 0 ; i < sort_indx.rows ; ++i){
			sorted_circles.at<float>(i,0) = circles_in.at<float>(sort_indx.at<int>(i,0),0);
			sorted_circles.at<float>(i,1) = circles_in.at<float>(sort_indx.at<int>(i,0),1);
			sorted_circles.at<float>(i,2) = circles_in.at<float>(sort_indx.at<int>(i,0),2);
		}
		break;
	}
	Mat sorted_circles_cut = sorted_circles.rowRange(0,no_limit);

	this->circles = sorted_circles_cut;
}


Mat CircleGroup::getCircleCenters(void){
	Mat center = this->circles.colRange(0,2);
	return center;
}

Mat CircleGroup::getCircleRadii(void){
	Mat radii = this->circles.colRange(2,3);
	return radii;
}

Mat permGenerator(int n, int k)
{
	vector< vector<int> > permutations;
	vector<int> d(n);
    vector<int> d_short(k);
    for (std::size_t i = 0; i != d.size(); ++i) { d[i] = i;}
    do
    {
        for (int i = 0; i < k; i++){ d_short[i] = d[i]; }
        permutations.push_back(d_short);
        std::reverse(d.begin()+k,d.end());
    } while (next_permutation(d.begin(),d.end()));

    Mat perm_Mat(permutations.size(), k, CV_32SC1);
    for (size_t i = 0 ; i < permutations.size() ; ++i ){
    	for(int j = 0; j < k ; ++j){
    		perm_Mat.at<int>(i,j) = permutations[i][j];
    	}
    }
    return perm_Mat;
}

Mat shuffleRows(const Mat &matrix)
{
  std::vector <int> seeds;
  for (int cont = 0; cont < matrix.rows; cont++)
    seeds.push_back(cont);

  randShuffle(seeds);

  Mat output;
  for (int cont = 0; cont < matrix.rows; cont++)
    output.push_back(matrix.row(seeds[cont]));

  return output;
}

Mat removeOverlapCircles(Mat circles){
	Mat circles_removed;
	Mat circle_centers = circles.colRange(0,2);
	Mat diff_centers_sqr, diff_sqr;
	Mat circle_radii = circles.col(2);
	Mat r_sum;
	Mat c_diff;
	Mat tmp_mask, mask;
	Mat tmp_circles;
	double minVal, maxVal;
	int minIndx, maxIndx;


	for (int i = 0 ; i < circles.rows ; ++i){
		pow(circle_centers-repeat(circle_centers.row(i),circles.rows,1), 2, diff_centers_sqr);
		reduce(diff_centers_sqr, diff_sqr, 1, CV_REDUCE_SUM);
		sqrt(diff_sqr, c_diff);
		r_sum = abs(circle_radii + repeat(circle_radii.row(i),circles.rows,1));
		tmp_mask = c_diff < r_sum;
		mask = repeat(tmp_mask, 1, 3);
		circles.copyTo(tmp_circles, mask);
		minMaxIdx(tmp_circles.col(2), &minVal, &maxVal, &minIndx, &maxIndx);
		tmp_circles.release();

		if (maxIndx==i){
			circles_removed.push_back(circles.row(i));
		}
	}
	return circles_removed;
}

// get the affine transform matrix given two pairs of corresponding points.
Mat getAffineMat(Mat x1, Mat x2)
{
	float data_A[4][4] = {{x1.at<float>(0,0), -x1.at<float>(0,1), 1.0, 0.0} ,
						  {x1.at<float>(0,1), x1.at<float>(0,0),  0.0, 1.0} ,
						  {x1.at<float>(1,0), -x1.at<float>(1,1), 1.0, 0.0} ,
						  {x1.at<float>(1,1), x1.at<float>(1,0),  0.0, 1.0}};

	Mat A(4,4,CV_32FC1, &data_A);
	float data_b[4] = {x2.at<float>(0,0), x2.at<float>(0,1),
				       x2.at<float>(1,0), x2.at<float>(1,1)};
	Mat b(1,4,CV_32FC1, &data_b);
	Mat x = A.inv()*b.t();
	Mat H = (Mat_<float>(3,3) << x.at<float>(0,0), -x.at<float>(1,0), x.at<float>(2,0),
									 x.at<float>(1,0),  x.at<float>(0,0), x.at<float>(3,0),
									 0.0, 0.0, 1.0);
	return H;
}


Mat optimalAffineMat(CircleGroup* src, CircleGroup* dst, float th, unsigned long int max_numLoop)
{
	unsigned long int numLoop = 1;
	unsigned long int max_count = 0;

	// get the circle centers and radii
	Mat x1 = src->getCircleCenters();
	Mat x2 = dst->getCircleCenters();
	Mat r1 = src->getCircleRadii();
	Mat r2 = dst->getCircleRadii();

	// create homogeneous coordinate matrix of the center points
	Mat x1_hom = x1.t();
	Mat make_hom1 = Mat::ones(1,x1.rows, CV_32FC1);
	x1_hom.push_back(make_hom1);

	Mat x2_hom = x2.t();
	Mat make_hom2 = Mat::ones(1,x2.rows, CV_32FC1);
	x2_hom.push_back(make_hom2);

	// generate permutations and shuffle randomly
	Mat perPts1 = permGenerator(x1.rows, 2);
	Mat perPts2 = permGenerator(x2.rows, 2);
	Mat ranPts1 = shuffleRows(perPts1);
	Mat ranPts2 = shuffleRows(perPts2);

	Mat sub_x1(2, 2, CV_32FC1);
	Mat sub_x2(2, 2, CV_32FC1);
	Mat x1_warp_hom, x1_warp, x1_warp_copy, r1_copy;
	Mat diff_center, diff_center2, distance1, distance2, distance;
	Mat affMat, affMat_final;
	unsigned long int count = 0;
	int x11_indx, x12_indx, x21_indx, x22_indx;
	Mat min_distance;
	float min_distance_num;

	for (int i = 0 ; i < ranPts1.rows ; ++i){
		for (int j = 0 ; j <ranPts2.rows ; ++j){
			count = 0;
			x11_indx = ranPts1.at<int>(i,0);
			x12_indx = ranPts1.at<int>(i,1);
			x21_indx = ranPts2.at<int>(j,0);
			x22_indx = ranPts2.at<int>(j,1);
			x1.row(x11_indx).copyTo(sub_x1.row(0));
			x1.row(x12_indx).copyTo(sub_x1.row(1));
			x2.row(x21_indx).copyTo(sub_x2.row(0));
			x2.row(x22_indx).copyTo(sub_x2.row(1));

			// get the affine transform matrix for each case
			affMat = getAffineMat(sub_x1,  sub_x2);

			// warp x1
			x1_warp_hom = affMat*x1_hom;
			x1_warp_hom.pop_back(1);
			x1_warp = x1_warp_hom.t();

			for (int k = 0 ; k < x1_warp.rows ; ++k){
				x1_warp_copy = repeat(x1_warp.row(k),x2.rows,1);
				pow(x2-x1_warp_copy, 2, diff_center);
				reduce(diff_center, diff_center2, 1, CV_REDUCE_SUM);
				sqrt(diff_center2, distance1);
				r1_copy =repeat(r1.row(k),r2.rows,1);
				distance2 = r2 - r1_copy;
				distance = distance1 + distance2;
				reduce(distance, min_distance, 0, CV_REDUCE_MIN);
				min_distance_num = min_distance.at<float>(0,0);
				if (min_distance_num < th){ count++; }

			}

			if (max_count < count){
				max_count = count;
				cout << "max count = " << count << " / " << min(x1.rows, x2.rows) << endl;
				affMat_final = affMat;
			}
			numLoop++;

			if (max_count >= 0.9*min(x1.rows,x2.rows) || numLoop>max_numLoop){ break; }
		}
		if (max_count >= 0.9*min(x1.rows,x2.rows) || numLoop>max_numLoop){ break; }
	}

	return affMat_final;
}


