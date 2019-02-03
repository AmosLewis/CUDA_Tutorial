#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <conio.h>
#include <ctime>

#include "FindCloestGPU.h"

using namespace std;

int main()
{
	srand(time(NULL));
	// numberof points
	const int count = 10000;

	// array of points
	int *indexOfCloest = new int[count];
	float3d *points = new float3d[count];

	int *d_indexOfCloest = new int[count];
	float3d *d_points = new float3d[count];

	// create a list of random points
	for (int i = 0; i < count; i++)
	{
		points[i].x = (float)(rand() % 10000 - 5000);
		points[i].y = (float)(rand() % 10000 - 5000);
		points[i].z = (float)(rand() % 10000 - 5000);
	}

	// allocate GPU memory
	cudaMalloc(&d_points, sizeof(float3d)*count);
	cudaMalloc(&d_indexOfCloest, sizeof(int)*count);

	//copy from CPU -> GPU
	cudaMemcpy(d_points, points, sizeof(float3d) * count, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_indexOfCloest, indexOfCloest, sizeof(int)*count, cudaMemcpyHostToDevice);

	// track the fast time so far
	long fastest = 1000000;

	// run the algorithm 10 times
	for (int q = 0; q < 10; q++)
	{
		long startTime = clock();

		// Run the algorithm
		//FindCloestCPU(points, indexOfCloest, count);

		FindCloestGPU2<<<(count / 1024) + 1, 1024 >>>(d_points, d_indexOfCloest, count);
		cudaMemcpy(indexOfCloest, d_indexOfCloest, sizeof(int)*count, cudaMemcpyDeviceToHost);

		long finishTime = clock();
		cout << "Run " << q << " tooks " << (finishTime - startTime) << " millis " << endl;

		// if that run faster update the fastest time
		if ((finishTime - startTime) < fastest)
		{
			fastest = finishTime - startTime;
		}
	}

	cout << "Fastest time: " << fastest << endl;

	cout << "Final results: " << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << i << "." << indexOfCloest[i] << endl;
	}

	delete[] indexOfCloest;
	delete[] points;
	cudaFree(d_points);
	cudaFree(d_indexOfCloest);

	cudaDeviceReset();
	//_getch();
	return 0;
}

/*
Run 0 tooks 172 millis
Run 1 tooks 156 millis
Run 2 tooks 156 millis
Run 3 tooks 156 millis
Run 4 tooks 156 millis
Run 5 tooks 165 millis
Run 6 tooks 171 millis
Run 7 tooks 157 millis
Run 8 tooks 172 millis
Run 9 tooks 156 millis
Fastest time: 156
Final results:
0.6634
1.5760
2.4348
3.8022
4.3039
5.5750
6.3481
7.5505
8.2954
9.7554
*/

/*
set q 60-70
Run 60 tooks 94 millis
Run 61 tooks 109 millis
Run 62 tooks 94 millis
Run 63 tooks 78 millis
Run 64 tooks 78 millis
Run 65 tooks 78 millis
Run 66 tooks 78 millis
Run 67 tooks 78 millis
Run 68 tooks 78 millis
Run 69 tooks 78 millis
Fastest time: 78
Final results:
0.4742
1.8046
2.2075
3.2417
4.4452
5.1367
6.8928
7.7888
8.337
9.1859
*/

/*
set thread 1024
Run 0 tooks 63 millis
Run 1 tooks 62 millis
Run 2 tooks 63 millis
Run 3 tooks 46 millis
Run 4 tooks 47 millis
Run 5 tooks 62 millis
Run 6 tooks 62 millis
Run 7 tooks 47 millis
Run 8 tooks 63 millis
Run 9 tooks 47 millis
Fastest time: 46
Final results:
0.1388
1.6690
2.5918
3.1731
4.1459
5.6047
6.3757
7.4032
8.5540
9.9065
*/

/*
set compute_20 -> compute_20
Run 0 tooks 47 millis
Run 1 tooks 47 millis
Run 2 tooks 62 millis
Run 3 tooks 63 millis
Run 4 tooks 46 millis
Run 5 tooks 63 millis
Run 6 tooks 47 millis
Run 7 tooks 62 millis
Run 8 tooks 47 millis
Run 9 tooks 67 millis
Fastest time: 46
Final results:
0.4752
1.594
2.7661
3.9382
4.8911
5.3446
6.6047
7.3418
8.1654
9.4205

*/

/*
use FindCloestGPU2
which use 1024 float3 shared memory
Run 0 tooks 31 millis
Run 1 tooks 47 millis
Run 2 tooks 31 millis
Run 3 tooks 47 millis
Run 4 tooks 47 millis
Run 5 tooks 31 millis
Run 6 tooks 32 millis
Run 7 tooks 31 millis
Run 8 tooks 32 millis
Run 9 tooks 41 millis
Fastest time: 31
Final results:
0.2876
1.6022
2.5693
3.4147
4.6879
5.5743
6.9816
7.5201
8.6942
9.4043
*/
