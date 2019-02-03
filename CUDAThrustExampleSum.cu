
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\sort.h>
#include <thrust\reduce.h>

using namespace std;

int main()
{
	thrust::device_vector<int> dv(0);
	thrust::host_vector<int> hv(0);
	
	for (int i = 0; i < 5; i++)
	{
		hv.push_back(rand() % 101);
	}

	dv = hv;
	thrust::sort(dv.begin(), dv.end());
	for (int i = 0; i < 5; i++)
		cout << "Ite: " << i << " is " << dv[i] << endl;

	float sum = thrust::reduce(dv.begin(), dv.end());

	cout << "Average is " << sum / 5.0f << endl;

	return 0;
}
/*
Ite: 0 is 38
Ite: 1 is 41
Ite: 2 is 72
Ite: 3 is 80
Ite: 4 is 85
Average is 63.2

*/