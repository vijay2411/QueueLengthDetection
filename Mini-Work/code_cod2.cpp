#include <bits/stdc++.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include </home/nvidia/users/mini/CudaFunc.h>
using namespace cv;
using namespace std;

//usr/local/include/opencv4/opencv2/
//int ix=30,fx=200 ;
int caly ;
int calx ;
int X,Y ;
int sx,sy,dx,dy ;
// MUST SET BELOW PARAMETERS FIRST
double BLOB_RATIO=0.3;
int XFIT=5 ;
int YFIT=7 ;
int DYNAMIC_COUNT=50 ;
int VARIANCE_FROM_AVG =25 ; //variance from avg pixel
int VAR =125 ; //variance from avg pixel

vector<Point2f> points;  
vector<Point2f> points2;
bool proj_set=false;
bool size_set=false;

Mat staticAVG,dynAVG ;
Mat staticAVG_dif,dynAVG_dif ;
double white_area_variable=0.0;

void on_mouse2( int e, int x, int y, int d, void *ptr ){

	if (e == EVENT_LBUTTONDOWN )

	{

		if(points2.size() < 3 )

		{


			points2.push_back(Point2f(float(x),float(y)));

			cout << x << " "<< y <<endl;

		}

		else if(points2.size()==3)

		{

			points2.push_back(Point2f(float(x),float(y)));

			cout << x << " "<< y <<endl;

			cout << " Done, Thanks! " <<endl;

			size_set=true;

			// Deactivate callback

			setMouseCallback("Display window2", NULL, NULL);

		}


	}

}

void on_mouse( int e, int x, int y, int d, void *ptr ){

	if (e == EVENT_LBUTTONDOWN )

	{

		if(points.size() < 3 )

		{


			points.push_back(Point2f(float(x),float(y)));

			cout << x << " "<< y <<endl;

		}

		else if(points.size()==3)

		{

			points.push_back(Point2f(float(x),float(y)));

			cout << x << " "<< y <<endl;

			cout << " Done, Thanks! " <<endl;

			proj_set=true;

			// Deactivate callback

			setMouseCallback("Display window", NULL, NULL);

		}


	}

}

void size_points(Mat src2final){

   

	imshow( "Display window2", src2final );

	setMouseCallback("Display window2",on_mouse2, NULL );

	while(1)

	{

		int key=cvWaitKey(0);

		if(key==27) break;

	}

	return;

}

void proj_points(Mat ss){

	imshow( "Display window", ss );

	setMouseCallback("Display window",on_mouse, NULL );

	while(1)

	{

		int key=cvWaitKey(0);

		if(key==27) break;

	}

	return;

}

Mat proj(Mat input_Frame,bool colour,Mat ss){

		vector<Point2f> left_image;                 // Stores 4 points(x,y) of the logo image. Here the four points are 4 corners of image.

		vector<Point2f> right_image;        // stores 4 points that the user clicks(mouse left click) in the main image.


		if(!proj_set){

			proj_points(ss);

		}


		left_image.push_back(points[0]);

		left_image.push_back(points[1]);

		left_image.push_back(points[2]);

		left_image.push_back(points[3]);


//Dont Change////////////////////////////////

		right_image.push_back(Point2f(float(472),float(52)));

		right_image.push_back(Point2f(float(472),float(830)));

		right_image.push_back(Point2f(float(800),float(830)));

		right_image.push_back(Point2f(float(800),float(52)));

////////////////////////////////////////////


		Mat H = findHomography(left_image,right_image,0 );

		Mat logoWarped;

		warpPerspective(input_Frame,logoWarped,H,Size(1280,875));

		Mat gray,src2final;

		if(colour) cvtColor(logoWarped,gray,CV_BGR2GRAY); //IN COLOUR

		else gray=logoWarped.clone(); //IN GRAY_SCALE

		threshold(gray,gray,0,255,CV_THRESH_BINARY);

		logoWarped.copyTo(src2final,gray);


		if(!size_set){

			size_points(src2final);

		}


		float p1x=points2[0].x;

		float p1y=points2[0].y;

		float p2x=points2[1].x;

		float p2y=points2[1].y;

		float p3x=points2[2].x;

		float p3y=points2[2].y;

		float p4x=points2[3].x;

		float p4y=points2[3].y;


		float a1x=(p1x+p2x)/2;

		float a1y=(p1y+p4y)/2;

		float a2x=(p3x+p4x)/2;

		float a2y=(p2y+p3y)/2;


		Mat retImage = src2final(Rect(a1x,a1y,a2x-a1x,a2y-a1y));

		//img[900:56, 389:831]

   return retImage;

   }

Mat proj(Mat input_Frame,bool colour){

		// We need 4 corresponding 2D points(x,y) to calculate homography.

		vector<Point2f> left_image;                 // Stores 4 points(x,y) of the logo image. Here the four points are 4 corners of image.

		vector<Point2f> right_image;        // stores 4 points that the user clicks(mouse left click) in the main image.


		// Image containers for main and logo image

		Mat imageMain;

		Mat imageLogo=input_Frame;

		float x1 = 327;

		float y1 = 4;

		float x2 = 44;

		float y2 = 441;

		float x3 = 571;

		float y3 = 562;

		float x4 = 440;

		float y4 = 5;

		left_image.push_back(Point2f(x1,y1));

		left_image.push_back(Point2f(x2,y2));

		left_image.push_back(Point2f(x3,y3));

		left_image.push_back(Point2f(x4,y4));


		right_image.push_back(Point2f(float(472),float(52)));

		right_image.push_back(Point2f(float(472),float(830)));

		right_image.push_back(Point2f(float(800),float(830)));

		right_image.push_back(Point2f(float(800),float(52)));



		Mat H = findHomography(         left_image,right_image,0 );

		Mat logoWarped;

		// Warp the logo image to change its perspective

		warpPerspective(imageLogo,logoWarped,H,Size(1280,875));

		//showFinal(imageMain,logoWarped);


		Mat gray,gray_inv,src1final,src2final;

		if(colour) cvtColor(logoWarped,gray,CV_BGR2GRAY); //IN COLOUR

		else gray=logoWarped.clone(); //IN GRAY_SCALE

		threshold(gray,gray,0,255,CV_THRESH_BINARY);

		//adaptiveThreshold(gray,gray,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,5,4);

		bitwise_not ( gray, gray_inv );

				//imageMain.copyTo(src1final,gray_inv);

		logoWarped.copyTo(src2final,gray);

		Mat finalImage = src2final;

		Mat retImage = finalImage(Rect(430,12,440,804));

		//img[900:56, 389:831]

   return retImage;

   }

void find_line(Mat img,int xfit, int yfit, bool up,double ylimTop, double tempBlobRatio){ //note that img has to be grayscale
	//ylimtop is the ratio from top which is ylimit as upper box ;
	int ylimTopPix = (1.0-ylimTop)*img.rows;
	calx = img.cols/xfit ;
	caly = img.rows/yfit ;
	int rows = img.rows ;
	int cols = img.cols ;
	bool done=false ;
	int bw[rows-ylimTopPix][cols];
	for(int i=ylimTopPix;i<rows;i++)
		for(int j=0;j<cols;j++)
			bw[i-ylimTopPix][j]=0;
	// Mat img2 = img.clone() ;

	if((int)img.at<uchar>(ylimTopPix,0)>150)
		bw[0][0]= 1 ;

	for(int i=1;i<cols;i++)
		if((int)img.at<uchar>(0,i)>150)
			bw[0][i]=bw[0][i-1]+1 ;
		else 
			bw[0][i]=bw[0][i-1] ;

	for(int j=ylimTopPix+1;j<rows;j++)
		if((int)img.at<uchar>(j,0)>150)
			bw[j-ylimTopPix][0]=bw[j-1-ylimTopPix][0]+1 ;
		else 
			bw[j-ylimTopPix][0]=bw[j-1-ylimTopPix][0] ;                

	if(up){
		int y=1+ylimTopPix ; int x=1 ;
		for(;y<rows;y++){
			for(x=1;x<cols;x++){
				if((int)(img.at<uchar>(y,x))>150)
					bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1]+1;
				else
					bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1];
				
				if(y>=caly+ylimTopPix&&x>=calx && ((double)(((int)bw[y-ylimTopPix][x]+(int)bw[y-caly-ylimTopPix][x-calx]-(int)bw[y-caly-ylimTopPix][x]-(int)bw[y-ylimTopPix][x-calx])*1.0/(calx*caly))>tempBlobRatio))
					done=true ;
				if(done)
					break ;                
			}
			if(done)
				break ;
		}
		if(!done){
			X=0;Y=img.rows;}
		else{
			X=x-calx ;Y=y-caly ;}
			// line(img2,Point(0,Y),Point(img2.cols,Y),Scalar(255),1) ;
			// line(img2,Point(X,0),Point(X,img2.rows),Scalar(255),1) ;
	}

	else{
		int y=1+ylimTopPix ; int x=1 ;
		for(;y<rows;y++)
			for(x=1;x<cols;x++)
				if((int)(img.at<uchar>(y,x))>150)
					bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1]+1;
				else
					bw[y-ylimTopPix][x]=(int)bw[y-ylimTopPix][x-1]+(int)bw[y-1-ylimTopPix][x]-(int)bw[y-1-ylimTopPix][x-1];

		for(y=rows-1;y>=caly+ylimTopPix;y--){
			for(x=calx;x<cols;x++){
				//cout<<"ratio:"<<(double)(((int)bw[y][x]+(int)bw[y-caly][x-calx]-(int)bw[y-caly][x]-(int)bw[y][x-calx])*1.0/(calx*caly))<<endl;
				if((double)(((int)bw[y-ylimTopPix][x]+(int)bw[y-caly-ylimTopPix][x-calx]-(int)bw[y-caly-ylimTopPix][x]-(int)bw[y-ylimTopPix][x-calx])*1.0/(calx*caly))>tempBlobRatio)
					done=true ;
				if(done)
					break ;
			}
		if(done)
			break ;
		}
		
		if(!done)
			{X=0;Y=ylimTopPix;}
		else
			{X=x-calx ;Y=y-caly ;}

			// line(img2,Point(0,Y),Point(img2.cols,Y),Scalar(255),1) ;
			// line(img2,Point(X,0),Point(X,img2.rows),Scalar(255),1) ;                      
	}
}

void find_line(Mat img,int xfit, int yfit, bool up){
	return find_line(img,xfit,yfit,up,1.00,BLOB_RATIO) ;
}

void find_line(Mat img){
	return find_line(img,XFIT,YFIT,true) ;
}

void find_line(Mat img,bool up){

		return find_line(img,XFIT,YFIT,up) ;

}

bool isRed_(Mat statMat,Mat dynMat){
	double ImgRatio = 0.4 ;
	find_line(statMat,XFIT*2,YFIT*2,false,ImgRatio,0.5) ;
	int sy=Y ;
	find_line(dynMat,XFIT*2,YFIT*2,false,ImgRatio,0.5) ;
	int dy=Y ;
	cout<<"dy:"<<dy<<" sy:"<<sy<<" rows:"<<statMat.rows ;
	int ch1=(1.00-ImgRatio)*statMat.rows;
	bool bs=sy>ch1 && sy<statMat.rows ;
	bool bd=dy>ch1 ;
	cout<<" bs:"<<bs<<" bd:"<<bd<<endl ;
	if(!(bs&&!bd))
		cout<<"**********************************GREEN********************************************" ;
		
	return (bs&&!bd) ;
}

bool isRed(Mat statMat,Mat dynMat){

			int rows = statMat.rows ;

			int cols = statMat.cols ;

			

	int ch1=(1.00-(double)(1.5/YFIT))*rows;

			int whitesInStatic=0 ;

			int whitesInDynamic=0 ;

			int total=(rows-ch1)*cols ; 

			for(int y=rows-1;y>=ch1-1;y--)

					for(int x=0;x<cols;x++){

							if((int)statMat.at<uchar>(y,x)>150)

									whitesInStatic++ ;

							if((int)dynMat.at<uchar>(y,x)>150)

									whitesInDynamic++ ;

					}

					

			double whitesInStaticRatio = whitesInStatic*1.0/total ;

			double whitesInDynamicRatio = whitesInDynamic*1.0/total ;

			cout<<"static_ratio:"<<whitesInStaticRatio<<"dynamic_ratio:"<<whitesInDynamicRatio<<endl;

			double ifVehicleInStaticRatio = 2*(1.0)/(XFIT*3) ;

			double ifVehicleInDynamicRatio = (1.0)/(XFIT*5) ;

			cout<<"static_ratio_reqd:"<<ifVehicleInStaticRatio<<"dynamic_ratio_reqd:"<<ifVehicleInDynamicRatio<<endl;

			bool Red = false ;

			if(whitesInStaticRatio  > ifVehicleInStaticRatio && whitesInDynamicRatio <ifVehicleInDynamicRatio)

					Red = true ;
			cout<<"Red:"<<Red<<endl;

			return Red ;

}

void white_area(Mat img){
	int rows = img.rows ;
	int cols = img.cols ;
	int countpos=0;
	int countall=0;
	for(int j=0;j<rows;j++){
		for(int i=0;i<cols;i++){
			if((int)img.at<uchar>(j,i)>150)
				countpos++;
			countall++;
		}
	}
	white_area_variable=(double)((countpos*1.0)/(countall*1.0));

}

int main(int argc, char *argv[]){
	auto startcode = chrono::high_resolution_clock::now();

	if(argc<2) cout<<"Video name missing!"<<endl ;
	else{
		ofstream outfile;
		outfile.open ("results.txt");
		string vid_name = argv[1] ;
		VideoCapture Video_cap(vid_name) ;
		if(!Video_cap.isOpened()){
			cout << "Error opening video stream or file" << endl;
			return -1;
		}
		
		double Vwidth = Video_cap.get(CAP_PROP_FRAME_WIDTH) ;
		double Vheight = Video_cap.get(CAP_PROP_FRAME_HEIGHT);
		double fps = round(Video_cap.get(CAP_PROP_FPS)) ;
		cout<<"Width: "<<Vwidth<<endl ;
		cout<<"Length: "<<Vheight<<endl ;
		cout<<"FPS: "<<fps<<endl ;

		Mat frame(Size(Vwidth,Vheight),CV_8UC3,Scalar(0)),sumframe(frame.size(),CV_64FC3,Scalar(0)) ;
		Mat avgframe,ErodeDilateMat,MedianBlurMat,difframe,difframe_2,difframe_3,
		difdisplayframe,difdisplayframe_2,difframe_result,temp2frame,blackwhiteframe,tempframe,
		projDYN,projSTAT,projCOL,staticINIT,SS ;

		int count=0 ;
		int group=DYNAMIC_COUNT ;
		deque<Mat> forAverage ;
		count=0 ;

		//skip

		Video_cap.read(frame);
		int height = frame.rows ;
		int width = frame.cols ;
		imwrite("intbkg.jpg",frame) ;
		cvtColor(frame, staticINIT, CV_BGR2GRAY);
		cout<<"count "<<endl;

	////////////////////for projection selection///////////////////////////////////
			if(argc==3){
				SS=staticINIT.clone();
				SS=proj(SS,true,SS);
			}
	/////////////////////////////////////////////////////////////////////////////////////////
		// while(true){
		//     imshow("ss",SS) ;
		//     if(waitKey(0)==27) break ;   
		// }                

		staticAVG = staticINIT.clone() ;

		////I changed the next line////
		if(argc==3){
			dynAVG=staticINIT.clone();
		}

		for(int i=0;i<(int)1*fps;i++){
			Video_cap.read(frame);
		}
		cvtColor(frame, blackwhiteframe, CV_BGR2GRAY);
		//
		int framefactor=1;
		int root=3;
		int square=root*root;
		
		int loop=0;
		Mat tempframes[square*framefactor];
		Mat tempframe_old;
		Mat tempframes_total[6];
		
		cout<<"BAW "<<blackwhiteframe.size()<<" "<<blackwhiteframe.channels()<<endl ;
		for(int i=0;i<6;i++){
			Mat local ;
			(blackwhiteframe).convertTo(local,CV_64FC1) ;
			tempframes_total[i]=local ;
		}
		//fill initially by capturing initial 6*square*framefactor frames(initial 6 secs)
		//fill initially by capturing initial 6*square*framefactor frames
		//fill initially by capturing initial 6*square*framefactor frames
		//fill initially by capturing initial 6*square*framefactor frames
		
		Mat sum_tempframes;//sum of the 6 average frames;
		//Mat avg_tempframes;//avg of the 6 average frames;
		for(int i=0;i<6;i++)
			if(i==0)
				sum_tempframes=tempframes_total[i]/6;
			else
				sum_tempframes+=tempframes_total[i]/6;
		
		cout<<"Sumtempframes "<<sum_tempframes.size()<<" "<<sum_tempframes.channels()<<" " ;
		sum_tempframes.convertTo(dynAVG,CV_8UC1,1.0/6.0);
		
		imwrite("firstimg.jpeg",dynAVG) ;
		char *converted_arr[framefactor];
		for(int i=0;i<framefactor;i++)
			converted_arr[i]=(char *)malloc(square*height*width) ;
		
		int loop_counter=0;
		bool isRED=false ;
		 //VideoWriter outdyn("outdyn.avi",CV_FOURCC('M','J','P','G'),10/9, Size(width,height),false);
		 //VideoWriter outstat("outstat.avi",CV_FOURCC('M','J','P','G'),10/9, Size(width,height),false);
		while(true){
			auto startloop = chrono::high_resolution_clock::now() ;
			int counter=0;
			
			while(counter<square*framefactor){
				bool success = Video_cap.read(frame) ;
				if(!success) {
						cout<<"Cannot read frame anymore"<<endl;
						break ;
				}
				cvtColor(frame, blackwhiteframe, CV_BGR2GRAY); //TO black and white	
				if(!isRED) {//Green
					//0.997 depends on fps ... change later
					staticAVG=0.997*staticAVG+0.003*blackwhiteframe;
					cout<<"------------------static_changes---------------------------"<<endl;
				}
				//I can reduce substantial time here ;
				tempframes[counter]=blackwhiteframe.clone();
				counter++;
			}
			
			cout<<"Loop going "<<endl ;
			
			for(int f=0;f<framefactor;f++)
				for(int r=0;r<square;r++)
					for(int y=0;y<height;y++)
						for(int x=0;x<width;x++)
							converted_arr[f][r*height*width+y*width+x]=((char)(tempframes[f*square+r].at<uchar>(y,x)));
						
			char sum_arr[height*width] ;
			auto st = chrono::high_resolution_clock::now() ;
			cout<<"Gpu function called . . ." ;
			AvgCalGpu(height,width,3,framefactor,converted_arr,sum_arr) ;	
			cout<<" . . . Gpu function has returned"<<endl ;
			cout<<"Time taken by gpu function "<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-st).count()/1000.0<<endl;
			char sum_array[height][width];
			for(int i=0;i<height;i++)
				for(int j=0;j<width;j++)
					sum_array[i][j]=sum_arr[(i*width)+j];
			
			
			Mat A(height,width,CV_8UC1,sum_array);
			std::memcpy(A.data,sum_array,height*width*sizeof(char));
			//imwrite("Avgphoto.jpeg",A) ;
			Mat Aframe ;
			A.convertTo(Aframe,CV_64FC1) ;
			Mat Bframe=Aframe.clone();

			sum_tempframes-=tempframes_total[loop_counter%6]/6;
			tempframes_total[loop_counter%6]=Bframe;
			sum_tempframes+=Bframe/6;
			//imwrite("sumframes.jpeg",sum_tempframes);
			sum_tempframes.convertTo(dynAVG,CV_8UC1);	
			//imwrite("avg.jpeg",dynAVG) ;
	
			///DYNAVG CODE OVER.... dynAVG is not correct
	
			threshold(cv::abs(blackwhiteframe-staticAVG),staticAVG_dif,VARIANCE_FROM_AVG,255,THRESH_BINARY) ;
			threshold(cv::abs(blackwhiteframe-dynAVG),dynAVG_dif,VARIANCE_FROM_AVG,255,THRESH_BINARY) ;
			//imwrite("dynavgdiff.jpeg",dynAVG_dif);

			// bool isRED = isRed(proj(staticAVG_dif,false,SS),proj(dynAVG_dif,false,SS));

			isRED = isRed_(staticAVG_dif,dynAVG_dif);
			if(argc==3){
				projDYN = proj(dynAVG_dif,0,SS) ;
				projSTAT = proj(staticAVG_dif,0,SS) ;
			}else{
				projDYN = proj(dynAVG_dif,0) ;
				projSTAT = proj(staticAVG_dif,0) ;
			}
			white_area(projSTAT);
			
			
			//Image_processing
			// erode(projDYN,projDYN,Mat(),Point(-1,-1),2,1,1) ;
			// dilate(projDYN,projDYN,Mat(),Point(-1,-1),2,1,1) ;
			// erode(projSTAT,projSTAT,Mat(),Point(-1,-1),2,1,1) ;
			// dilate(projSTAT,projSTAT,Mat(),Point(-1,-1),2,1,1) ;
			//medianBlur(difframe_result,MedianBlurMat,5) ;
			// imshow("Queue_Length_DYNAMIC",(projDYN)) ;// dx=X ;dy=Y ;


			// imshow("Queue_Length_STATIC",(staticAVG_dif)) ; //sx=X ;sy=Y ;
			// imshow("Queue_Length_Dynamic",(dynAVG_dif)) ; //sx=X ;sy=Y ;
			// cout<<"find_line:"<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-start).count()<<endl; 

					// line(projCOL,Point(0,sy),Point(projDYN.cols,sy),Scalar(255,255,255),1) ;
					// line(projCOL,Point(sx,0),Point(sx,projDYN.rows),Scalar(255,255,255),1) ;   
					
			// imshow("Projected",projCOL);
		   //cout<<"DX: "<<dx<<","<<"DY: "<<dy<<endl ;
			//cout<<"SX: "<<sx<<","<<"SY: "<<sy<<endl ;

			//imshow("Dynavg_dif",dynAVG_dif) ;
			//outdyn<<projDYN;
			//imshow("STATICAVG_dif",staticAVG_dif) ;
			if(argc==3)
				find_line(proj(staticAVG_dif,0,SS)) ;
			else
				find_line(proj(staticAVG_dif,0)) ;
			sy=Y ;
			if(argc==3)
				find_line(proj(dynAVG_dif,0,SS)) ;
			else
				find_line(proj(dynAVG_dif,0)) ;
			dy=Y ;
			//writing part
			string output_ans = to_string(loop)+":"+to_string(white_area_variable)+","+to_string(sy) ;
			outfile<<output_ans<<endl ;
			//cout<<output_ans<<endl ;
			//cout<<"________________________dy"<<dy<<"====================="<<sy<<endl ;
			//outstat<<projSTAT;
			//cout<<"imshow:"<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-start3).count()<<endl; 
			//imshow("DiffFrame",difframe_result) ;
			cout<<loop++<<endl;

			if(waitKey(1)==27) break ;
			cout<<"total:"<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now()-startloop).count()/1000.0<<endl; 
			loop_counter++;	
		}

		//imwrite("Size_det.jpg",frame) ;
		cout<<"I am out of the loop,Press Esc to exit!"<<endl ;
		cout<<"Total time to run code:"<<chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now()-startcode).count()<<endl; 

		if(waitKey(10000)==27) return 0 ;
    }
	return 0 ;
}
