//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <iostream>
//#include <conio.h>
//#include <ctime>
//
//#include "FindCloestCPU.h"
//
//using namespace std;
//
//int main()
//{
//	srand(time(NULL));
//	 numberof points
//	const int count = 10000;
//
//	 array of points
//	int *indexOfCloest = new int[count];
//	float3d *points = new float3d[count];
//
//	 create a list of random points
//	for (int i = 0; i < count; i++)
//	{
//		points[i].x = (float)(rand() % 10000 - 5000);
//		points[i].y = (float)(rand() % 10000 - 5000);
//		points[i].z = (float)(rand() % 10000 - 5000);
//	}
//
//	 track the fast time so far
//	long fastest = 1000000;
//
//	 run the algorithm 10 times
//	for (int q = 0; q < 10; q++)
//	{
//		long startTime = clock();
//
//		 Run the algorithm
//		FindCloestCPU(points, indexOfCloest, count);
//		long finishTime = clock();
//		cout << "Run " << q << " tooks " << (finishTime - startTime) << " millis " << endl;
//
//		 if that run faster update the fastest time
//		if ((finishTime - startTime) < fastest)
//		{
//			fastest = finishTime - startTime;
//		}
//	}
//
//	cout << "Fastest time: " << fastest << endl;
//
//	cout << "Final results: " << endl;
//	for (int i = 0; i < 10; i++)
//	{
//		cout << i << "." << indexOfCloest[i] << endl;
//	}
//
//	delete[] indexOfCloest;
//	delete[] points;
//
//	_getch();
//	return 0;
//}
///*
//Run 0 tooks 1016 millis
//Run 1 tooks 978 millis
//Run 2 tooks 997 millis
//Run 3 tooks 983 millis
//Run 4 tooks 985 millis
//Run 5 tooks 974 millis
//Run 6 tooks 985 millis
//Run 7 tooks 998 millis
//Run 8 tooks 1001 millis
//Run 9 tooks 991 millis
//Fastest time: 974
//Final results:
//0.7122
//1.268
//2.5194
//3.2730
//4.2386
//
//5.7373
//6.3969
//7.8220
//8.4509
//9.8517
//
//*/
