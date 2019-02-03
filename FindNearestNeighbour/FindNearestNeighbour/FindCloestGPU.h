#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <limits>

#include "FindCloestCPU.h"

using namespace std;

__global__
void FindCloestGPU(float3d* points,int* indices, int count)
{
	if (count <= 1) return;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < count)
	{
		float3d thisPoint = points[idx];
		float smallestSoFar = 3.4e38f;
		for (int i = 0; i < count; i++)
		{
			if (i == idx) continue;
			float dist =
				(thisPoint.x - points[i].x) *
				(thisPoint.x - points[i].x);
			dist +=
				(thisPoint.y - points[i].y) *
				(thisPoint.y - points[i].y);
			dist +=
				(thisPoint.z - points[i].z) *
				(thisPoint.z - points[i].z);
			if (dist < smallestSoFar)
			{
				smallestSoFar = dist;
				indices[idx] = i;
			}

		}
	}
}