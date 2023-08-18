#include<iostream>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;
using namespace std;



// ====================================== draw line, circle, or something =======================================
// #define w 400
// using namespace cv;
// void MyEllipse( Mat img, double angle );
// void MyFilledCircle( Mat img, Point center );
// void MyPolygon( Mat img );
// void MyLine( Mat img, Point start, Point end );
// int main( void ){
//   char atom_window[] = "Drawing 1: Atom";
//   char rook_window[] = "Drawing 2: Rook";
//   Mat atom_image = Mat::zeros( w, w, CV_8UC3 );
//   Mat rook_image = Mat::zeros( w, w, CV_8UC3 );
//   MyEllipse( atom_image, 90 );
//   MyEllipse( atom_image, 0 );
//   MyEllipse( atom_image, 45 );
//   MyEllipse( atom_image, -45 );
//   MyFilledCircle( atom_image, Point( w/2, w/2) );
//   MyPolygon( rook_image );
//   rectangle( rook_image,
//          Point( 0, 7*w/8 ),
//          Point( w, w),
//          Scalar( 0, 255, 255 ),
//          FILLED,
//          LINE_8 );
//   MyLine( rook_image, Point( 0, 15*w/16 ), Point( w, 15*w/16 ) );
//   MyLine( rook_image, Point( w/4, 7*w/8 ), Point( w/4, w ) );
//   MyLine( rook_image, Point( w/2, 7*w/8 ), Point( w/2, w ) );
//   MyLine( rook_image, Point( 3*w/4, 7*w/8 ), Point( 3*w/4, w ) );
//   imwrite( "atom_image.jpg", atom_image );
//   moveWindow( atom_window, 0, 200 );
//   imwrite( "rook_image.jpg", rook_image );
// //   moveWindow( rook_window, w, 200 );
// //   waitKey( 0 );
//   return(0);
// }
// void MyEllipse( Mat img, double angle )
// {
//   int thickness = 2;
//   int lineType = 8;
//   ellipse( img,
//        Point( w/2, w/2 ),
//        Size( w/8, w/16 ),
//        angle,
//        0,
//        360,
//        Scalar( 0, 0, 255 ),
//        thickness,
//        lineType );
// }
// void MyFilledCircle( Mat img, Point center )
// {
//   circle( img,
//       center,
//       w/24,
//       Scalar( 255, 255, 255 ),
//       FILLED,
//       LINE_8 );
// }
// void MyPolygon( Mat img )
// {
//   int lineType = LINE_8;
//   Point rook_points[1][20];
//   rook_points[0][0]  = Point(    w/4,   7*w/8 );
//   rook_points[0][1]  = Point(  3*w/4,   7*w/8 );
//   rook_points[0][2]  = Point(  3*w/4,  13*w/16 );
//   rook_points[0][3]  = Point( 11*w/16, 13*w/16 );
//   rook_points[0][4]  = Point( 19*w/32,  3*w/8 );
//   rook_points[0][5]  = Point(  3*w/4,   3*w/8 );
//   rook_points[0][6]  = Point(  3*w/4,     w/8 );
//   rook_points[0][7]  = Point( 26*w/40,    w/8 );
//   rook_points[0][8]  = Point( 26*w/40,    w/4 );
//   rook_points[0][9]  = Point( 22*w/40,    w/4 );
//   rook_points[0][10] = Point( 22*w/40,    w/8 );
//   rook_points[0][11] = Point( 18*w/40,    w/8 );
//   rook_points[0][12] = Point( 18*w/40,    w/4 );
//   rook_points[0][13] = Point( 14*w/40,    w/4 );
//   rook_points[0][14] = Point( 14*w/40,    w/8 );
//   rook_points[0][15] = Point(    w/4,     w/8 );
//   rook_points[0][16] = Point(    w/4,   3*w/8 );
//   rook_points[0][17] = Point( 13*w/32,  3*w/8 );
//   rook_points[0][18] = Point(  5*w/16, 13*w/16 );
//   rook_points[0][19] = Point(    w/4,  13*w/16 );
//   const Point* ppt[1] = { rook_points[0] };
//   int npt[] = { 20 };
//   fillPoly( img,
//         ppt,
//         npt,
//         1,
//         Scalar( 255, 255, 255 ),
//         lineType );
// }
// void MyLine( Mat img, Point start, Point end )
// {
//   int thickness = 2;
//   int lineType = LINE_8;
//   line( img,
//     start,
//     end,
//     Scalar( 0, 0, 0 ),
//     thickness,
//     lineType );
// }



//============================================= erode and dilate ===============================================
// int main(){
//     Mat src = imread("flower.jpg", IMREAD_COLOR);
//     Mat dst_1 = src.clone();
//     Mat dst_2 = src.clone();
//     Mat element = getStructuringElement( 0,
//                        Size( 3, 3 ),
//                        Point( 0, 0 ) );
//     dilate(src, dst_1, element);
//     erode(src, dst_2, element);
//     imwrite("erode.jpg", dst_2);
//     imwrite("dilate.jpg", dst_1);
//     cout<< element <<endl;
//     return 0;
// }



// =============================================== gaussianblur ===========================================
// int main(){
//     Mat src = imread("view.jpg", IMREAD_COLOR);
//     Mat dst = src.clone();
//     GaussianBlur(src, dst, Size(5,5), 0, 0);
//     imwrite("show_2.jpg", dst);
//     return 0;
// }



//============================================= read and write yaml or xml =======================================
// static void help(char** av)
// {
//     cout << endl
//         << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
//         << "usage: "                                                                      << endl
//         <<  av[0] << " outputfile.yml.gz"                                                 << endl
//         << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
//         << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
//         << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
//         << "For example: - create a class and have it serialized"                         << endl
//         << "             - use it to read and write matrices."                            << endl;
// }
// class MyData
// {
// public:
//     MyData() : A(0), X(0), id()
//     {}
//     explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
//     {}
//     void write(FileStorage& fs) const                        //Write serialization for this class
//     {
//         fs << "{" << "A" << A << "X" << X << "id" << id << "}";
//     }
//     void read(const FileNode& node)                          //Read serialization for this class
//     {
//         A = (int)node["A"];
//         X = (double)node["X"];
//         id = (string)node["id"];
//     }
// public:   // Data Members
//     int A;
//     double X;
//     string id;
// };
// //These write and read functions must be defined for the serialization in FileStorage to work
// static void write(FileStorage& fs, const std::string&, const MyData& x)
// {
//     x.write(fs);
// }
// static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
//     if(node.empty())
//         x = default_value;
//     else
//         x.read(node);
// }
// // This function will print our custom class to the console
// static ostream& operator<<(ostream& out, const MyData& m)
// {
//     out << "{ id = " << m.id << ", ";
//     out << "X = " << m.X << ", ";
//     out << "A = " << m.A << "}";
//     return out;
// }
// int main(int ac, char** av)
// {
//     if (ac != 2)
//     {
//         help(av);
//         return 1;
//     }
//     string filename = av[1];
//     { //write
//         Mat R = Mat_<uchar>::eye(3, 3),
//             T = Mat_<double>::zeros(3, 1);
//         MyData m(1);
//         FileStorage fs(filename, FileStorage::WRITE);
//         // or:
//         // FileStorage fs;
//         // fs.open(filename, FileStorage::WRITE);
//         fs << "iterationNr" << 100;
//         fs << "strings" << "[";                              // text - string sequence
//         fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
//         fs << "]";                                           // close sequence
//         fs << "Mapping";                              // text - mapping
//         fs << "{" << "One" << 1;
//         fs <<        "Two" << 2 << "}";
//         fs << "R" << R;                                      // cv::Mat
//         fs << "T" << T;
//         fs << "MyData" << m;                                // your own data structures
//         fs.release();                                       // explicit close
//         cout << "Write Done." << endl;
//     }
//     {//read
//         cout << endl << "Reading: " << endl;
//         FileStorage fs;
//         fs.open(filename, FileStorage::READ);
//         int itNr;
//         //fs["iterationNr"] >> itNr;
//         itNr = (int) fs["iterationNr"];
//         cout << itNr;
//         if (!fs.isOpened())
//         {
//             cerr << "Failed to open " << filename << endl;
//             help(av);
//             return 1;
//         }
//         FileNode n = fs["strings"];                         // Read string sequence - Get node
//         if (n.type() != FileNode::SEQ)
//         {
//             cerr << "strings is not a sequence! FAIL" << endl;
//             return 1;
//         }
//         FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
//         for (; it != it_end; ++it)
//             cout << (string)*it << endl;
//         n = fs["Mapping"];                                // Read mappings from a sequence
//         cout << "Two  " << (int)(n["Two"]) << "; ";
//         cout << "One  " << (int)(n["One"]) << endl << endl;
//         MyData m;
//         Mat R, T;
//         fs["R"] >> R;                                      // Read cv::Mat
//         fs["T"] >> T;
//         fs["MyData"] >> m;                                 // Read your own structure_
//         cout << endl
//             << "R = " << R << endl;
//         cout << "T = " << T << endl << endl;
//         cout << "MyData = " << endl << m << endl << endl;
//         //Show default behavior for non existing nodes
//         cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
//         fs["NonExisting"] >> m;
//         cout << endl << "NonExisting = " << endl << m << endl;
//     }
//     cout << endl
//         << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;
//     return 0;
// }



//============================================ mix two photos =============================================
// //we'reNOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
// using std::cin;
// using std::cout;
// using std::endl;
// int main( void )
// {
//    double alpha = 0.5; double beta; double input;
//    Mat src1, src2, dst;
//    cout << " Simple Linear Blender " << endl;
//    cout << "-----------------------" << endl;
//    cout << "* Enter alpha [0.0-1.0]: ";
//    cin >> input;
//    //We use the alpha provided by the user if it is between 0 and 1
//    if( input >= 0 && input <= 1 )
//      { alpha = input; }
//    src1 = imread( "cat12.jpg" );
//    src2 = imread( "cat100.jpg" );
//    if( src1.empty() ) { cout << "Error loading src1" << endl; return EXIT_FAILURE; }
//    if( src2.empty() ) { cout << "Error loading src2" << endl; return EXIT_FAILURE; }
//    beta = ( 1.0 - alpha );
//    addWeighted( src1, alpha, src2, beta, 0.0, dst);
//    imwrite( "mix.jpg", dst );
//    return 0;
// }



//============= erode, dilate, open, close, ... morphologyEx(src, dst, select 0~6 here, element) ================
// int main(){
//     Mat src = imread("open.png", IMREAD_COLOR);
//     Mat dst = src.clone();
//     Mat element = getStructuringElement( 0,
//                        Size( 5, 5 ),
//                        Point( 0, 0 ) );
//     morphologyEx(src, dst, 0, element);
//     imwrite("morphologyEx.png", dst);
//     cout<< element <<endl;
//     return 0;
// }



//================================== find horizontal and vertical lines =====================================
// int main(){
//     Mat gray, bw;
//     Mat src = imread("music.jpg", IMREAD_COLOR);

//     if(src.channels() == 3){
//         cvtColor(src, gray, COLOR_BGR2GRAY);
//     }else{
//         gray = src;
//     }
    
//     adaptiveThreshold(~gray, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
//     Mat horizontal = bw.clone();
//     Mat vertical = bw.clone();

//     int horizontal_size = horizontal.cols / 30;
//     Mat element_1 = getStructuringElement(0, Size(horizontal_size, 1));
//     erode(horizontal, horizontal, element_1, Point(-1, -1));
//     dilate(horizontal, horizontal, element_1, Point(-1, -1));
    
//     int vertical_size = vertical.rows / 30;
//     Mat element_2 = getStructuringElement(0, Size(1, vertical_size));
//     erode(vertical, vertical, element_2, Point(-1, -1));
//     dilate(vertical,vertical, element_2, Point(-1, -1));
    
//     imwrite("horizontal.jpg", horizontal);
//     imwrite("vertical.jpg", vertical);
//     cout<< element_1 <<endl;
//     cout<< element_2 <<endl;
//     return 0;
// }



//======================================== image pyramids =======================================================
// int main(){
//     Mat src = imread("cat100.jpg");
//     cout << src.cols << "---" << src.rows <<endl;
//     Mat dst_up;
//     Mat dst_down;
//     pyrUp(src, dst_up, Size(src.cols * 2, src.rows * 2));
//     pyrDown(src, dst_down, Size(src.cols / 2, src.rows / 2));
//     imwrite("up.png", dst_up);
//     imwrite("down.png", dst_down);
// }



//================================================== threshold ===============================================
// int main(){
//     /* 0: Binary
//      1: Binary Inverted
//      2: Threshold Truncated
//      3: Threshold to Zero
//      4: Threshold to Zero Inverted
//     */
//     Mat dst, src_gray;
//     Mat src = imread("view.jpg");
//     if(src.channels() == 3){
//         cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     }else{
//         src_gray = src;
//     }
//     threshold(src_gray, dst, 100.0, 255.0, 0);
//     imwrite("threshold.jpg", dst);
// }



//================================= Thresholding Operations using inRang e================================

// 待定...

//============================================== Canny Edge Detector ======================================
// Mat src, src_gray;
// Mat dst, detected_edges;

// int lowThreshold = 0;
// const int max_lowThreshold = 100;
// const int ratio_ = 3;
// const int kernel_size = 3;
// const char* window_name = "Edge Map";

// static void CannyThreshold(int, void*){
//     blur(src_gray, detected_edges, Size(3,3));
//     Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio_, kernel_size);
//     dst = Scalar::all(0);
//     src.copyTo(dst, detected_edges);
//     imshow(window_name, dst);
// }

// int main(int argc, char** argv){
//     CommandLineParser parser( argc, argv, "{@input | fruits.jpg | input image}" );
//     src = imread("/home/zzq/code/demo/2.jpg");
//     resize(src, src, Size(), 0.2, 0.2);
//     dst.create(src.size(), src.type());
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     namedWindow(window_name, WINDOW_AUTOSIZE);
//     createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
//     CannyThreshold(0,0);
//     waitKey(0);

//     return 0;
// }

//=================================== Hough Line Transform =============================

// int main(){
//     Mat dst, cdst, cdstp;

//     const string name = "2.jpg";
//     Mat src = imread(name, IMREAD_GRAYSCALE);
//     resize(src, src, Size(512,512));
//     Canny(src, dst, 50, 200, 3);
//     //dst是Canny处理过的图片
//     cvtColor(dst, cdst, COLOR_GRAY2BGR);
//     cdstp = cdst.clone();

//     vector<Vec2f> lines;
//     HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0);

//     for(size_t i = 0; i < lines.size(); i++){
//         float rho = lines[i][0], theta = lines[i][1];
//         Point pt1, pt2;
//         double a = cos(theta), b = sin(theta);
//         double x0 = a*rho, y0 = b*rho;
//         pt1.x = cvRound(x0 + 1000*(-b));
//         pt1.y = cvRound(y0 + 1000*(a));
//         pt2.x = cvRound(x0 - 1000*(-b));
//         pt2.y = cvRound(y0 - 1000*(a));
//         line(cdst, pt1, pt2, Scalar(0,255,0), 1);
//     }

//     vector<Vec4i> linesP;
//     HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10);
//     for(size_t i = 0; i < linesP.size(); i++){
//         Vec4i L = linesP[i];
//         line(cdstp, Point(L[0], L[1]), Point(L[2], L[3]), Scalar(0,255,0), 1);
//     }

//     imshow("Source", dst);
//     imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
//     imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstp);
//     waitKey(0);

// }

//=================================== Hough Cricle Transform ==============================

// int main(int argc, char** argv){

//  // Loads an image
//  Mat src = imread("cat100.jpg");
//  // Check if image is loaded fine
 
//  Mat gray;
//  cvtColor(src, gray, COLOR_BGR2GRAY);
//  medianBlur(gray, gray, 3);

//  vector<Vec3f> circles;
//  HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
//  gray.rows/16, // change this value to detect circles with different distances to each other
//  100, 30, 1, 30); //主要影响因子是倒数第三个参数
//  // (min_radius & max_radius) to detect larger circles
 
//  for( size_t i = 0; i < circles.size(); i++ ){
//  Vec3i c = circles[i];
//  Point center = Point(c[0], c[1]);
//  // circle center
//  circle( src, center, 1, Scalar(0,100,100), 3, LINE_AA);
//  // circle outline
//  int radius = c[2];
//  circle( src, center, radius, Scalar(0,0,255), 3, LINE_AA);
//  }
//  imshow("detected circles", src);
//  waitKey();

// }

//============================ Remapping  image =============================

// void update_map(int &ind, Mat &map_x, Mat &map_y);
// const string filename = "up.png";
// int main(int argc, const char** argv){
//     Mat src = imread(filename, IMREAD_COLOR);
//     Mat dst(src.size(), src.type());
//     Mat map_x(src.size(), CV_32FC1);
//     Mat map_y(src.size(), CV_32FC1);

//     const char* remap_window = "Remap demo";
//     namedWindow(remap_window, WINDOW_AUTOSIZE);

//     int ind = 0;
//     for(;;){
//         update_map(ind, map_x, map_y);
//         remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
//         imshow(remap_window, dst);
//         char c = (char)waitKey(1000);
//         if(c==27) break;
//     }
// }

// void update_map(int &ind, Mat &map_x, Mat &map_y){
//     for(int i = 0; i < map_x.rows; i++){
//         for(int j = 0; j < map_y.cols; j++){
//             switch(ind){
//                 case 0:
//                 if(j > map_x.cols*0.25 && j < map_x.cols*0.75 && i > map_x.rows*0.25 && i < map_x.rows*0.75){
//                     map_x.at<float>(i,j) = 2*(j - map_x.cols*0.25f) + 0.5f;
//                     map_y.at<float>(i,j) = 2*(i - map_x.rows*0.25f) + 0.5f;
//                 }else{
//                     map_x.at<float>(i,j) = 0;
//                     map_y.at<float>(i,j) = 0;
//                 }break;

//                 case 1:
//                 map_x.at<float>(i,j) = (float)j;
//                 map_y.at<float>(i,j) = (float)(map_x.rows - i);
//                 break;

//                 case 2:
//                 map_x.at<float>(i, j) = (float)(map_x.cols - j);
//                 map_y.at<float>(i, j) = (float)i;
//                 break;

//                 case 3:
//                 map_x.at<float>(i, j) = (float)(map_x.cols - j);
//                 map_y.at<float>(i, j) = (float)(map_x.rows - i);
//                 break;

//                 default:
//                 break;
//             }
//         }
//     }
//     ind = (ind + 1) % 4;
// }

//============================== Affine Transformations =============================

// int main(int argc, char** argv){

//     const string filename = "view.jpg";
//     Mat src = imread(filename);

//     Point2f srcTri[3];
//     srcTri[0] = Point2f(0.f, 0.f);
//     srcTri[1] = Point2f(src.cols - 1.f, 0.f);
//     srcTri[2] = Point2f(0.f, src.rows - 1.f);

//     Point2f dstTri[3];
//     dstTri[0] = Point2f(0.f, src.rows * 0.33f);
//     dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
//     dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);

//     Mat warp_mat = getAffineTransform(srcTri, dstTri);
//     Mat warp_dst = Mat::zeros(src.size(), src.type());
//     warpAffine(src, warp_dst, warp_mat, warp_dst.size());

//     Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
//     double angle = -90.0;
//     double scale = 0.6;
//     Mat rot_mat = getRotationMatrix2D(center, angle, scale);
//     Mat warp_rotate_dst;
//     warpAffine(src, warp_rotate_dst, rot_mat, warp_dst.size());

//     imshow("Source image", src);
//     imshow("Warp_1", warp_dst);
//     imshow("Warp_2", warp_rotate_dst);
//     waitKey();
//     return 0;
// }

//=========================== Histogram Equalization =============================

// int main(int argc, char** argv){
//     const string filename = "view.jpg";
//     Mat src = imread(filename);
//     cvtColor(src, src, COLOR_BGR2GRAY);
//     Mat dst;
//     equalizeHist(src, dst); //直方图均衡化
//     imshow("src", src);
//     imshow("dst", dst);
//     waitKey();
// }

//=========================================== Contours =========================================

// Mat src_gray;
// int thresh = 100;
// RNG rng(12345);

// void thresh_callback(int, void*);
// int main(int argc, char** argv){
    
//     Mat src = imread("sudoku.png");
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     blur(src_gray, src_gray, Size(3,3));

//     const char* source_window = "Source";
//     namedWindow(source_window);
//     imshow(source_window, src);

//     const int max_thresh = 255;
//     createTrackbar("Min", source_window, &thresh, max_thresh, thresh_callback);
//     thresh_callback(0, 0);
//     waitKey();
// }

// void thresh_callback(int, void*){
//     Mat canny_output;
//     Canny(src_gray, canny_output, thresh, thresh*2);
//     vector<vector<Point>> contours;
//     vector<Vec4i> hierarchy;
//     findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

//     Mat draw = Mat::zeros(canny_output.size(), CV_8UC3);
//     for(size_t i = 0; i < contours.size(); i++){
//         Scalar color = Scalar(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
//         drawContours(draw, contours, (int)i, color, 2, 8,hierarchy);
//     }
//     imshow("Contours", draw);
// }

//=============================== Bounding boxes and circles ===================================

// Mat gray;
// int thresh = 100;
// RNG rng(12345);

// void thresh_callback(int, void*);
// int main(int argc, char** argv){
    
//     Mat src = imread("sudoku.png");
//     cvtColor(src, gray, COLOR_BGR2GRAY);
//     blur(gray, gray, Size(3,3));

//     const char* source_window = "Source";
//     namedWindow(source_window);
//     imshow(source_window, src);
//     const int max_thresh = 255;
//     createTrackbar("Canny thresh", source_window, &thresh, max_thresh, thresh_callback);
//     thresh_callback(0, 0);
//     waitKey();
// }

// void thresh_callback(int, void*){

//     Mat canny_output;
//     Canny(gray, canny_output, thresh, thresh*2);
//     vector<vector<Point>> contours;
//     findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

//     vector<vector<Point>> contours_ploy(contours.size());
//     vector<Rect> boundRect(contours.size());
//     vector<Point2f> centers(contours.size());
//     vector<float> radius(contours.size());

//     for(size_t i = 0; i < contours.size(); i++){
//         approxPolyDP(contours[i], contours_ploy[i], 3, true);
//         boundRect[i] = boundingRect(contours_ploy[i]);
//         minEnclosingCircle(contours_ploy[i], centers[i], radius[i]);
//     }

//     Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
//     for(size_t i=0; i < contours.size(); i++){
//         Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
//         drawContours(drawing, contours, (int)i, color, 2);
//         rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
//         circle(drawing, centers[i], (int)radius[i], color, 2);
//     }
//     imshow(" Contours", drawing);

// }

//================================ Bounding rotate boxes and ellipses for contours ======================

/*暂时感觉用处不大*/

//========================================= Image Moments ================================ //未理解，需再看

// Mat src_gray;
// int thresh = 100;
// RNG rng(12345);

// void thresh_callback(int, void*);

// int main(int argc, char** argv){
//     Mat src = imread("");
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     blur(src_gray, src_gray, Size(3,3));

//     const char* source_window = "Source_window";
//     namedWindow(source_window);
//     imshow(source_window, src_gray);

//     const int max_thresh = 255;
//     createTrackbar("thresh", source_window, &thresh, max_thresh, thresh_callback);
//     thresh_callback(0, 0);
//     waitKey();
// }

// void thresh_callback(int, void*){
//     Mat canny_output;
//     Canny(src_gray, canny_output, thresh, thresh*2, 3);
//     vector<vector<Points>> contours;
//     findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

//     vector<Moments> mu(contours.size());
//     for(size_t i=0; i < contours.size(); i++){
//         mu[i] = moments(contours[i]);
//     }
//     vector<Point2f> mc(contours.size());
//     for(size_t i=0; i < contours.size(); i++){
//         mc[i] = Point2f( static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
//         static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)) );
//         cout << "mc[" << i << "]=" << mc[i] << endl;
//     }
    
//     Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
//     for( size_t i = 0; i< contours.size(); i++ ){
//         Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
//         drawContours( drawing, contours, (int)i, color, 2 );
//         circle( drawing, mc[i], 4, color, -1 );
//  }
//     imshow( "Contours", drawing );
//     cout << "\t Info: Area and Contour Length \n";
//     for( size_t i = 0; i < contours.size(); i++ ){
//         cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
//         << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength( contours[i], true ) << endl;
//     }
// }

//======================================= Out of focus deblur filter ========================================
/*目前效果不好，没有实现相应功能*/
// void help();
// void calcPSF(Mat& outputImg, Size filterSize, int R);
// void fftshift(const Mat& inputImg, Mat& outputImg);
// void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
// void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
// const String keys =
// "{help h usage ? | | print this message }"
// "{image |original.JPG | input image name }"
// "{R |53 | radius }"
// "{SNR |5200 | signal to noise ratio}"
// ;
// int main(int argc, char *argv[])
// {
//  help();
//  CommandLineParser parser(argc, argv, keys);
//  if (parser.has("help"))
//  {
//  parser.printMessage();
//  return 0;
//  }
//  int R = 10;
//  int snr = 40;

//  if (!parser.check())
//  {
//  parser.printErrors();
//  return 0;
//  }
//  Mat imgIn;
//  imgIn = imread("original.jpg", IMREAD_GRAYSCALE);
//  if (imgIn.empty()) //check whether the image is loaded or not
//  {
//  cout << "ERROR : Image cannot be loaded..!!" << endl;
//  return -1;
//  }
//  Mat imgOut;
//  // it needs to process even image only
//  Rect roi = Rect(0, 0, imgIn.cols & -2, imgIn.rows & -2);
//  //Hw calculation (start)
//  Mat Hw, h;
//  calcPSF(h, roi.size(), R);
//  calcWnrFilter(h, Hw, 1.0 / double(snr));
//  //Hw calculation (stop)
//  // filtering (start)
//  filter2DFreq(imgIn(roi), imgOut, Hw);
//  // filtering (stop)
//  imgOut.convertTo(imgOut, CV_8U);
//  normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
//  imwrite("result.jpg", imgOut);
//  return 0;
// }
// void help()
// {
//  cout << "2018-07-12" << endl;
//  cout << "DeBlur_v8" << endl;
//  cout << "You will learn how to recover an out-of-focus image by Wiener filter" << endl;
// }
// void calcPSF(Mat& outputImg, Size filterSize, int R)
// {
//  Mat h(filterSize, CV_32F, Scalar(0));
//  Point point(filterSize.width / 2, filterSize.height / 2);
//  circle(h, point, R, 255, -1, 8);
//  Scalar summa = sum(h);
//  outputImg = h / summa[0];
// }
// void fftshift(const Mat& inputImg, Mat& outputImg)
// {
//  outputImg = inputImg.clone();
//  int cx = outputImg.cols / 2;
//  int cy = outputImg.rows / 2;
//  Mat q0(outputImg, Rect(0, 0, cx, cy));
//  Mat q1(outputImg, Rect(cx, 0, cx, cy));
//  Mat q2(outputImg, Rect(0, cy, cx, cy));
//  Mat q3(outputImg, Rect(cx, cy, cx, cy));
//  Mat tmp;
//  q0.copyTo(tmp);
//  q3.copyTo(q0);
//  tmp.copyTo(q3);
//  q1.copyTo(tmp);
//  q2.copyTo(q1);
//  tmp.copyTo(q2);
// }
// void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
// {
//  Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
//  Mat complexI;
//  merge(planes, 2, complexI);
//  dft(complexI, complexI, DFT_SCALE);
//  Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
//  Mat complexH;
//  merge(planesH, 2, complexH);
//  Mat complexIH;
//  mulSpectrums(complexI, complexH, complexIH, 0);
//  idft(complexIH, complexIH);
//  split(complexIH, planes);
//  outputImg = planes[0];
// }
// void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
// {
//  Mat h_PSF_shifted;
//  fftshift(input_h_PSF, h_PSF_shifted);
//  Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
//  Mat complexI;
//  merge(planes, 2, complexI);
//  dft(complexI, complexI);
//  split(complexI, planes);
//  Mat denom;
//  pow(abs(planes[0]), 2, denom);
//  denom += nsr;
//  divide(planes[0], denom, output_G);
// }

//=================================== Motion Deblur Filter =======================================

//================= Anisotropic images segmentation by a gridient structure tensor ===============

//============================ Periodic Noise Removing Filter ====================================

//=================================== Harris corner detector =====================================
// Mat src, src_gray;
// int thresh = 200;
// int max_thresh = 255;

// const char* source_window = "Source image";
// const char* corners_window = "Corners detect";

// void cornerHarris_demo(int, void*);
// int main(int argc, char** argv){
//     src = imread("square.jpg");
//     cvtColor(src, src_gray, COLOR_BGR2GRAY);
//     namedWindow(source_window);
//     createTrackbar("Threshold", source_window, &thresh, max_thresh, cornerHarris_demo);
//     imshow(source_window, src);
//     cornerHarris_demo(0, 0);
//     waitKey();
//     return 0;
// }

// void cornerHarris_demo(int, void*){
//     int blockSize = 2;
//     int kSize = 3;
//     double k = 0.05;

//     Mat dst = Mat::zeros(src.size(), CV_32FC1);
//     cornerHarris(src_gray, dst, blockSize, kSize, k);

//     Mat dst_norm, dst_norm_scaled;
//     normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//     convertScaleAbs(dst_norm, dst_norm_scaled);
//     for(int i=0; i < dst_norm.rows; i++){
//         for(int j=0; j < dst_norm.cols; j++){
//             if((int)dst_norm.at<float>(i,j) > thresh)
//                 circle(dst_norm_scaled, Point(j,i), 5, Scalar(0), 2, 8, 0);
//         }
//     }
//     namedWindow(corners_window);
//     imshow(corners_window, dst_norm_scaled);
// }

//================================= Shi Tomasi coner detector ============================

//================================= Creating your own corner detector ====================

//================================= Detecting corners location in subpixels ==============
Mat src, src_gray;

int maxCorners = 10;
int maxTrackbar = 25;

RNG rng(12345);
const char* source_window = "Image";

void subpixels_detector(int, void*);

int main(int argc, char** argv){

    src = imread("square.jpg");
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    namedWindow(source_window);
    createTrackbar("Max corners:", source_window, &maxCorners, maxTrackbar, subpixels_detector);
    imshow(source_window, src);
    subpixels_detector(0, 0);
    waitKey();
    return 0;
}

void subpixels_detector(int, void*){
    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    int gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    Mat copy = src.clone();
    goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(),
    blockSize, useHarrisDetector, k);

    cout << "Number of corners detected: " << corners.size() << endl;
    int radius = 4;
    for(size_t i=0; i < corners.size(); i++)
        circle(copy, corners[i], radius, Scalar(rng.uniform(0,256), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED );

    namedWindow(source_window);
    imshow(source_window, copy);

    Size winSize  = Size(5, 5);
    Size zeroZone = Size(-1, -1);
    TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 40, 0.01);

    cornerSubPix(src_gray, corners, winSize, zeroZone, criteria);
    for(size_t i=0; i < corners.size(); i++)
        cout << " -- Refined Corner [" << i << "] (" << corners[i].x << "," << corners[i].y << ")" << endl;
}


