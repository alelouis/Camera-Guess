
#include "Header.h"

using namespace cv;
using namespace std;


int main()
{
    
    FFN *DIGITnetwork = new FFN();
    DIGITnetwork->initFFN(44, 20, 10);
    
    vector<vector<float>> Xapp = readMatFromFile("data_set/old/Xapp.txt");
    vector<vector<float>> Ta = readMatFromFile("data_set/old/TA.txt");
    vector<vector<float>> Xtest = readMatFromFile("data_set/old/Xtest.txt");
    vector<vector<float>> Tt = readMatFromFile("data_set/old/TT.txt");
    
    cout << "Training Neural Network... Please Wait..." << endl;
    DIGITnetwork->train_by_iteration(Xapp,Ta,1000);
    
    VideoCapture cap(1);
    if (!cap.isOpened()) {
        
        printf("Flux vidéo introuvable !\n");
        return 1;
        
    }
    namedWindow( "Source", CV_WINDOW_AUTOSIZE );
    while(1){
        Mat frame;
        
        Mat frame_processed;
        //frame = imread("/Users/Alexis/Desktop/img_test.png");
        cap >> frame;
        flip(frame, frame, -1);
        cvtColor(frame, frame_processed, CV_BGR2GRAY);
        Mat threshold_output;
        Mat binary;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        threshold( frame_processed, threshold_output, 150, 255, THRESH_BINARY_INV );
        binary = threshold_output.clone();
        dilate(binary, binary, Mat(), Point(-1, -1), 3, 1, 1);
        
        Canny(frame_processed, frame_processed, 100, 200, 3);
        findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        vector<Rect> boundRect(contours.size());
        
        
        
        for( int i = 0; i < contours.size(); i++ )
        {
            boundRect[i] = boundingRect( Mat(contours[i]) );
        }
        
        Mat drawing = binary.clone();
        CvPoint* sq_tl = new CvPoint();
        CvPoint* sq_br = new CvPoint();
        cvtColor(drawing, drawing, CV_GRAY2BGR);
        Scalar color = Scalar(255,0,0);
        vector<deque<deque<int>>> digit_mats;
        for( int i = 0; i< boundRect.size(); i++ )
        {
            int side_x = boundRect[i].br().x-boundRect[i].tl().x;
            int side_y = boundRect[i].br().y-boundRect[i].tl().y;
            int max_size = max(side_x,side_y);
            int min_size = min(side_x,side_y);
            float delta = (max_size-min_size)/2.0;
            sq_tl->x = boundRect[i].tl().x-delta-max_size/8.0;
            sq_tl->y = boundRect[i].tl().y-max_size/8.0;
            sq_br->x = boundRect[i].br().x+delta+max_size/8.0;
            sq_br->y = boundRect[i].br().y+max_size/8.0;
            if(sq_br->x<480 && sq_br->y<480 && max_size<300 && min_size>10 &&
               sq_tl->x > 0 && sq_tl->y > 0){
                Rect* current_box = new Rect(*sq_tl,*sq_br);
                rectangle(drawing, *sq_tl, *sq_br, color, 2, 8, 0);
                Mat current_resized = binary(*current_box).clone();
                resize(current_resized,current_resized,Size(32,32));
                deque<deque<int>> current_vector_mat;
                current_vector_mat.resize(current_resized.rows);
                for (int row = 0; row < current_resized.rows; ++row) {
                    for(int col = 0; col< current_resized.cols; ++col){
                        current_vector_mat[row].push_back(current_resized.at<bool>(row,col));
                    }
                }
                digit_mats.push_back(current_vector_mat);
            }
        }
        
        
        for(int dig=0;dig<digit_mats.size();dig++){
            deque<deque<int>> data8 = compress_mat(digit_mats[dig]);
            vector<float> draw_variables = pca(normalize(extract_variables(data8)));
            DIGITnetwork->sim(draw_variables);
            vector<float> current_output = DIGITnetwork->get_ffn_outputs();
            int guess = (int)distance(current_output.begin(), max_element(current_output.begin(), current_output.end()));
            putText(drawing, to_string(guess), cvPoint(boundRect[dig].tl().x,boundRect[dig].tl().y),
                    FONT_HERSHEY_PLAIN, 1, cvScalar(255,255,255), 1, CV_AA);
        }
        imshow( "Source", drawing );
    }
    return 0;
}


