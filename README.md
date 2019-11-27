# QueueLengthDetection
To compile and run:
nvcc -c CudaFunc.cu -o cfo.o
g++ -c code_cod2.cpp -o ccd.o `pkg-config --cflags --libs opencv`
g++ -o cod.o ccd.o cfo.o -L/usr/local/cuda/lib64 -lcudart `pkg-config --cflags --libs opencv`
./cod.o cod_vid.avi

In the last line cod_vid.avi is the video name. To run any other video than cod_vid.avi use command "./cod.o [video_name] 1" to run.
Recommended to check and verify on cod_vid.avi first.
Results will be available in results.txt as-> loop_number:percentage_white_area,y_coordinate
loop_number is "n" means that it is the result of the (9*n)th frame, i.e. nth second if the video is 9fps.
y_coordinate is from the top of the frame. Small y_cooordinate means large queues/ last vehicle at far end.
