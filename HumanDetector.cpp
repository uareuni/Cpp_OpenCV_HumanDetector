#include <opencv.hpp>
#include <highgui.h>
#include <windows.h>
#include <iostream>


using namespace std;
using namespace cv;

int detect_hog_inria(Mat src);
int detect_hogcascades(Mat src);
int diffthre(Mat src, Mat thre);


int main()
{
   bool stop = false;
   int delay = 50;
   int n = 7;
   int element_shape = MORPH_RECT;
   Mat element = getStructuringElement(element_shape, Size(n, n));
   Mat src;
   char msg[50];

   Mat src1 = imread("1_18.jpg");
   Mat src2 = imread("2_34.jpg");
   Mat src3 = imread("3_30.jpg");
   Mat src4 = imread("4_23.jpg");
   Mat src5 = imread("5_21.jpg");

   src = src3.clone();
   Mat srcimg[5] = { src1, src2, src3, src4, src5 };

   Mat f = srcimg[0];

   Mat back = Mat::zeros(f.size(), CV_32FC3); // f가져와도 될까?

   Mat floatimg;

   for (int i = 0; i<5; i++)
   {
      srcimg[i].convertTo(floatimg, CV_32FC3);
      accumulateWeighted(floatimg, back, 0.001);
   }

   String accwin = "acc";
   namedWindow(accwin, WINDOW_NORMAL);
   imshow(accwin, back);
   waitKey();
   back.convertTo(back, CV_8UC3, 255);
   imshow(accwin, back);

   imwrite("back.jpg", back);
   waitKey();
   Mat diff;
   absdiff(src, back, diff);
   String diffwin = "diff";
   namedWindow(diffwin, WINDOW_NORMAL);
   imshow(diffwin, diff);

   Mat thre;
   cvtColor(diff, diff, CV_BGR2GRAY);
   blur(diff, diff, Size(3, 3));

   threshold(diff, thre, 70, 255, THRESH_BINARY);
   String threwin = "thre";
   namedWindow(threwin, WINDOW_NORMAL);
   imshow(threwin, thre);

   Mat thretemp = thre.clone();

   erode(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);

   String morph = "morph";
   namedWindow(morph, WINDOW_NORMAL);
   imshow(morph, thre);

   Mat contour, threedge;
   vector<vector<Point>> contours;
   vector<Vec4i> hierarchy;

   Canny(thre, threedge, 60, 255, 3);
   threedge.copyTo(contour);
   findContours(contour, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
   
   vector<Rect> rect(contours.size());
   vector<Mat> roi(contours.size());
   String roiWindow = "roi";
   namedWindow(roiWindow, WINDOW_AUTOSIZE);

   int cnt_inria = 0;
   int cnt_cascade = 0;
   int cnt_total = 0;

   for (int i = 0; i < contours.size(); i++){
      rect[i] = boundingRect(Mat(contours[i]));
      rectangle(thre, rect[i], Scalar(255), 1);
      Mat temp(src, rect[i]);
      Mat temproi(thretemp, rect[i]);
      roi[i] = temp.clone();

      resize(roi[i], roi[i], Size((roi[i].cols), (roi[i].rows)));

      imshow(roiWindow, roi[i]);
      if (roi[i].cols > 48 && roi[i].rows > 96){
         cnt_inria = detect_hog_inria(roi[i]);
         cnt_cascade = detect_hogcascades(roi[i]);
         if (cnt_inria == -1 || cnt_cascade == -1)
            printf("Error: HOG descriptor Fail");
         else if (cnt_inria == 0 && cnt_cascade == 0){
            cnt_total += diffthre(roi[i], temproi);
         }
         else if (cnt_inria >= cnt_cascade)
            cnt_total += cnt_inria;
         else cnt_total += cnt_cascade;
      }
      waitKey();
   }


   resize(src, src, Size(640, 480));
   sprintf(msg, "The number of people: %d", cnt_total);
   putText(src, (string)(msg), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);
   imshow("Result", src);


   waitKey();

}

int diffthre(Mat src, Mat thre){
   int n = 5;
   int element_shape = MORPH_RECT;
   Mat element = getStructuringElement(element_shape, Size(n, n));
   erode(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   dilate(thre, thre, element);
   imshow("thretemp", thre);

   Mat contour, threedge;
   vector<vector<Point>> contours;
   vector<Vec4i> hierarchy;

   Canny(thre, threedge, 60, 255, 3);
   threedge.copyTo(contour);
   findContours(contour, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

   return contours.size();
}

int detect_hog_inria(Mat src){
   HOGDescriptor hog;
   hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

   double hit_thr = 0;
   double gr_thr = 2;
   int cnt_inria = 0;

   Mat frame;
   Mat roi;

   frame = src.clone();
   // detect
   vector<Rect> found;
   hog.detectMultiScale(frame, found, hit_thr, Size(8, 8), Size(32, 32), 1.05, gr_thr);

   cnt_inria = (int)found.size();
   // draw results (bounding boxes)
   if (cnt_inria == 0)
      return 0;
   else
      for (int i = 0; i < (int)found.size(); i++){
         if (roi.cols > 70 || roi.rows > 170){
            cnt_inria--;
            if (found[i].x + found[i].width >= frame.cols)
               found[i].width = frame.cols - found[i].x - 1;
            else if (found[i].x < 0)
               found[i].x = 0;
            if (found[i].y + found[i].height >= frame.rows)
               found[i].height = frame.rows - found[i].y - 1;
            else if (found[i].y < 0)
               found[i].y = 0;
            roi = frame(found[i]).clone();
         }
         else
            rectangle(frame, found[i], Scalar(0, 255, 0), 2);

      }

   // display
   imshow("inria", frame);
   waitKey();
   return cnt_inria;
}


int detect_hogcascades(Mat src)
{
   // detector (48x96 template)
   string cascadeName = "hogcascade_pedestrians.xml";
   CascadeClassifier detector;
   if (!detector.load(cascadeName))
   {
      cerr << "ERROR: Could not load classifier cascade" << endl;
      return -1;
   }

   // parameters
   int gr_thr = 6;
   double scale_step = 1.1;
   Size min_obj_sz(48, 96);
   Size max_obj_sz(100, 200);

   // run
   Mat frame;
   Mat roi;

   // input image
   frame = src.clone();

   // detect
   vector<Rect> found;
   detector.detectMultiScale(frame, found, scale_step, gr_thr, 0, min_obj_sz, max_obj_sz);

   int cnt_cc = (int)found.size();

   if (cnt_cc == 0){
      return 0;
   }
   else
      // draw results (bounding boxes)
      for (int i = 0; i < (int)found.size(); i++){
         
         rectangle(frame, found[i], Scalar(0, 255, 0), 2);

      }

   // display
   imshow("cascade", frame);
   return cnt_cc;

}