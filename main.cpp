/*
 * @Description: 
 * @Version: 
 * @Author: William
 * @Date: 2021-06-27 10:59:27
 * @LastEditors: William
 * @LastEditTime: 2021-07-25 17:36:50
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

void imageRotate(const cv::Mat &src, cv::Mat &dst, const double degree, const double scale);


/*图像旋转（以图像中心为旋转中心）*/
void affine_trans_rotate(cv::Mat& src, cv::Mat& dst, double Angle){

	/* 
	double angle = Angle*CV_PI / 180.0;
	//构造输出图像
	int dst_rows = round(fabs(src.rows * cos(angle)) + fabs(src.cols * sin(angle)));//图像高度
	int dst_cols = round(fabs(src.cols * cos(angle)) + fabs(src.rows * sin(angle)));//图像宽度
 
	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	} 
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 
	cv::Mat T1 = (cv::Mat_<double>(3,3) << 1.0,0.0,0.0 , 0.0,-1.0,0.0, -0.5*src.cols , 0.5*src.rows , 1.0); // 将原图像坐标映射到数学笛卡尔坐标
	cv::Mat T2 = (cv::Mat_<double>(3,3) << cos(angle),-sin(angle),0.0 , sin(angle), cos(angle),0.0, 0.0,0.0,1.0); //数学笛卡尔坐标下顺时针旋转的变换矩阵
	double t3[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 }, { 0.5*dst.cols, 0.5*dst.rows ,1.0} }; // 将数学笛卡尔坐标映射到旋转后的图像坐标
	cv::Mat T3 = cv::Mat(3.0,3.0,CV_64FC1,t3);
	cv::Mat T = T1*T2*T3;
	cv::Mat T_inv = T.inv(); // 求逆矩阵
 
	for (double i = 0.0; i < dst.rows; i++){
		for (double j = 0.0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1.0);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高
		//	std::cout << v << std::endl;
 
			// 双线性插值
			// 判断是否越界
			if (int(Angle) % 90 == 0) {
				if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
				if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; //必须要加上，否则会出现边界问题
			}
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top ; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}

 */

	cv::Size dst_sz(src.cols , src.rows );

	cv::Point2f center(src.cols / 2.0, src.rows  / 2.0);

	cv::Mat rot_mat = cv::getRotationMatrix2D(center,Angle,1.0);

	cv::Scalar borderColor = Scalar(0);

	cv::warpAffine(src, dst, rot_mat, dst_sz, INTER_LINEAR, BORDER_CONSTANT, borderColor);



}
 
/*平移变换*（以图像左顶点为原点）/
/****************************************
tx: 水平平移距离 正数向右移动 负数向左移动
ty: 垂直平移距离 正数向下移动 负数向上移动
*****************************************/
void affine_trans_translation(cv::Mat& src, cv::Mat& dst, double tx, double ty){
	//构造输出图像
	int dst_rows = src.rows;//图像高度
	int dst_cols = src.cols;//图像宽度
 
	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 
	cv::Mat T = (cv::Mat_<double>(3, 3) << 1,0,0 , 0,1,0 , tx,ty,1); //平移变换矩阵
	cv::Mat T_inv = T.inv(); // 求逆矩阵
 
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高
 
			/*双线性插值*/
			// 判断是否越界
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}
 
 
/*尺度变换*（以图像左顶点为原点）/
/***************
cx: 水平缩放尺度
cy: 垂直缩放尺度
***************/
void affine_trans_scale(cv::Mat& src, cv::Mat& dst, double cx, double cy){
	//构造输出图像
	int dst_rows = round(cy*src.rows);//图像高度
	int dst_cols = round(cx*src.cols);//图像宽度
 
	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 
	cv::Mat T = (cv::Mat_<double>(3, 3) <<cx,0,0, 0,cy,0 ,0,0,1 ); //尺度变换矩阵
	cv::Mat T_inv = T.inv(); // 求逆矩阵
 
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高
 
			/*双线性插值*/
			// 判断是否越界
			if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
			if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; 
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}
 
/*偏移变换*（以图像中心为偏移中心）/
/***************************************
sx: 水平偏移尺度 正数向左偏移 负数向右偏移
sy: 垂直偏移尺度 正数向上偏移 负数向下偏移
****************************************/
void affine_trans_deviation(cv::Mat& src, cv::Mat& dst, double sx, double sy){
	//构造输出图像
	int dst_rows = fabs(sy)*src.cols + src.rows;//图像高度
	int dst_cols = fabs(sx)*src.rows + src.cols;//图像宽度
 
	if (src.channels() == 1) {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_rows, dst_cols, CV_8UC3); //RGB图初始
	}
 
	cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, -0.5*src.cols, 0.5*src.rows, 1); // 将原图像坐标映射到数学笛卡尔坐标
	cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1,sy,0, sx,1,0, 0,0,1); //数学笛卡尔坐标偏移变换矩阵
	double t3[3][3] = { { 1, 0, 0 }, { 0, -1, 0 }, { 0.5*dst.cols, 0.5*dst.rows, 1 } }; // 将数学笛卡尔坐标映射到旋转后的图像坐标
	cv::Mat T3 = cv::Mat(3, 3, CV_64FC1, t3);
	cv::Mat T = T1*T2*T3;
	cv::Mat T_inv = T.inv(); // 求逆矩阵
 
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高
 
			/*双线性插值*/
			// 判断是否越界
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}
 
/*组合变换*/
/*变换顺序：缩放->旋转—>偏移*/
void affine_trans_comb(cv::Mat& src, cv::Mat& dst, double cx, double cy, double Angle, double sx, double sy){
	double angle = Angle*CV_PI / 180;
	//构造输出图像
	int dst_s_rows = round(cy*src.rows);//尺度变换后图像高度
	int dst_s_cols = round(cx*src.cols);//尺度变换后图像宽度
 
	int dst_sr_rows = round(fabs(dst_s_rows * cos(angle)) + fabs(dst_s_cols * sin(angle)));//再经过旋转后图像高度
	int dst_sr_cols = round(fabs(dst_s_cols * cos(angle)) + fabs(dst_s_rows * sin(angle)));//再经过旋转后图像宽度
 
	int dst_srd_rows = fabs(sy)*dst_sr_cols + dst_sr_rows;//最后经过偏移后图像高度
    int dst_srd_cols = fabs(sx)*dst_sr_rows + dst_sr_cols;//最后经过偏移后图像宽度
 
 
	if (src.channels() == 1) {
 
		dst = cv::Mat::zeros(dst_srd_rows, dst_srd_cols, CV_8UC1); //灰度图初始
	}
	else {
		dst = cv::Mat::zeros(dst_srd_rows, dst_srd_cols, CV_8UC3); //RGB图初始
	}
 
	cv::Mat T1 = (cv::Mat_<double>(3, 3) << cx, 0, 0, 0, cy, 0, 0, 0, 1); //尺度变换矩阵
 
	cv::Mat T21 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, -0.5*dst_s_cols, 0.5*dst_s_rows, 1); // 将尺度变换后的图像坐标映射到数学笛卡尔坐标
	cv::Mat T22 = (cv::Mat_<double>(3, 3) << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1); //数学笛卡尔坐标下顺时针旋转的变换矩阵
	cv::Mat T2 = T21*T22;// 这里不需要转回图像坐标了，因为下面的偏移变换是在笛卡尔坐标下进行的
 
	cv::Mat T32 = (cv::Mat_<double>(3, 3) << 1, sy, 0, sx, 1, 0, 0, 0, 1); //数学笛卡尔坐标偏移变换矩阵
	double t33[3][3] = { { 1, 0, 0 }, { 0, -1, 0 }, { 0.5*dst.cols, 0.5*dst.rows, 1 } }; // 将数学笛卡尔坐标映射到偏移后的图像坐标
	cv::Mat T33 = cv::Mat(3, 3, CV_64FC1, t33);
	cv::Mat T3 = T32*T33;
 
	cv::Mat T = T1*T2*T3; //矩阵相乘按照组合变换的顺序
	cv::Mat T_inv = T.inv(); // 求逆矩阵
 
	for (int i = 0; i < dst.rows; i++){
		for (int j = 0; j < dst.cols; j++){
			cv::Mat dst_coordinate = (cv::Mat_<double>(1, 3) << j, i, 1);
			cv::Mat src_coordinate = dst_coordinate * T_inv;
			double v = src_coordinate.at<double>(0, 0); // 原图像的横坐标，列，宽
			double w = src_coordinate.at<double>(0, 1); // 原图像的纵坐标，行，高
 
			/*双线性插值*/
			// 判断是否越界
			if (int(Angle) % 90 == 0) {
				if (v < 0) v = 0; if (v > src.cols - 1) v = src.cols - 1;
				if (w < 0) w = 0; if (w > src.rows - 1) w = src.rows - 1; //必须要加上，否则会出现边界问题
			}
 
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1){
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top; //pw为坐标 行 的小数部分(坐标偏差)
				double pv = v - left; //pv为坐标 列 的小数部分(坐标偏差)
				if (src.channels() == 1){
					//灰度图像
					dst.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw*(1 - pv)*src.at<uchar>(bottom, left) + pw*pv*src.at<uchar>(bottom, right);
				}
				else{
					//彩色图像
					dst.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw*pv*src.at<cv::Vec3b>(bottom, right)[0];
					dst.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw*pv*src.at<cv::Vec3b>(bottom, right)[1];
					dst.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw*(1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw*pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
}


int main()
{
    Mat origin_tmp;
    Mat afterExpand;
    Mat after_scale;
    Mat after_Rotate;
    Mat after_affine;

    Mat Background(480,640,CV_8UC1,Scalar(0));

    int rotation = 0;
    int Delta_x = 0;
    int Delta_Y = 0;
    int scale = 100;

    const int threshold = 5;

    //Background.resize(480,640);

    origin_tmp = imread("templates/apple-4.png",1);


	

	//cv::Mat scale_mat = cv::getRotationMatrix2D(center,angle,1.0);

    for(int i=0; i < 5000; i++)
    {
		
        scale += (rand() % 5 - 1);
        if(scale <50) scale = 50;
        else if (scale >200) scale = 200;
        affine_trans_scale(origin_tmp,after_scale,scale/100.0, scale/100.0);
		

        rotation += (rand() % 7 - 2);
        if(rotation < -90) rotation = -90;
        else if(rotation > 90) rotation = 90;
        affine_trans_rotate(after_scale,after_Rotate,rotation);
        //imageRotate(origin_tmp,after_Rotate,rotation,scale/100.0);
        //origin_tmp = after_Rotate;

        copyMakeBorder(after_Rotate,afterExpand,(480-after_Rotate.rows)/2,(480-after_Rotate.rows)/2,(640-after_Rotate.cols)/2,(640-after_Rotate.cols)/2,BORDER_CONSTANT,0);
        imshow("change", origin_tmp);
        imshow("after",afterExpand);
		cout << "Rotation Angle: "<< rotation << "; Scale factor: " << scale << endl;
        waitKey(100);

    }
	



   // affine_trans_rotate(origin_tmp,after_affine,45);

    /*使用拓展边界方法进行平移*/
   // copyMakeBorder(after_affine,afterExpand,(480-origin_tmp.rows)/2,(480-origin_tmp.rows)/2,(640-origin_tmp.cols)/2,(640-origin_tmp.cols)/2,BORDER_CONSTANT,0);
    imshow("origin", origin_tmp );

	//imshow("Rotated Image", after_Rotate);

	//imshow("Scaled Image", after_scale);
    //imshow("after",afterExpand);

    

    waitKey(0);

    
    
    //imageRotate(frame, after_affine, 45, 1);

   
    //imshow("rotated",after_affine);

    //waitKey(0);
    return 0;
}

void imageRotate(const cv::Mat &src, cv::Mat &dst, const double degree, const double scale)
{
    double angle = degree;
    int length = 0;
    if (scale <= 1)
        length = sqrt(src.cols*src.cols + src.rows*src.rows);
    else 
        length = sqrt(src.cols*src.cols + src.rows*src.rows) * scale;

    cv::Mat tmp_img = cv::Mat(length,length,src.type());

    int ROI_X = length / 2 - src.cols / 2;
    int ROI_Y = length / 2 - src.rows / 2;
    cv::Rect ROI_SRC(ROI_X, ROI_Y, src.cols, src.rows);
    cv::Mat tmp_ROI(tmp_img, ROI_SRC);
    src.copyTo(tmp_ROI);
    cv::Point2f rotate_center(length / 2, length / 2);
    cv::Mat affineMat = cv::getRotationMatrix2D(rotate_center, angle, scale);
    cv::warpAffine(tmp_img, tmp_img, affineMat, cv::Size(length, length), cv::INTER_CUBIC, 0, cv::Scalar(255, 255, 255));
    dst = tmp_img;
}