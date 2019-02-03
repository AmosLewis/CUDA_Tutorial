//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
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

__device__ const int blockSize = 1024;

__global__
void FindCloestGPU2(float3d* points, int* indices, int count)
{
	// each thread caculate a point
	__shared__ float3d sharedpoints[blockSize];

	if (count <= 1) return;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int indexOfClosest = -1;

	// iterate every point
	float3d thisPoint;
	if (idx < count) thisPoint = points[idx];

	float distanceToCloest = 3.40282e38f;

	// iterate every block, currentBlockOfPoints ~ blockIdx.x
	for (int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++)
	{
		// copy 1024byte host memory to shared memory
		int sharedpointsId = threadIdx.x + currentBlockOfPoints * blockSize;
		if (sharedpointsId < count)
		{
			sharedpoints[threadIdx.x] = points[sharedpointsId];
		}
		// enable all 1024 thread copy 1024 each points in global to sharedpoints
		__syncthreads();

		//calculate distance between each current point in current block with all the points
		if (idx < count)	
		{
			// get the first point.x address
			float *ptr = &sharedpoints[0].x;
			// iterate each point(thread) to calculate distance between current point	
			for (int i = 0; i < blockSize; i++)
			{
				float dist = 
					(thisPoint.x - ptr[0]) * 
					(thisPoint.x - ptr[0]) +
					(thisPoint.y - ptr[1]) *
					(thisPoint.y - ptr[1]) +
					(thisPoint.z - ptr[2]) *
					(thisPoint.z - ptr[2]);
				// skip x,y,z memory for this point, get the address of the next point.x
				ptr += 3;

				if ((dist < distanceToCloest) &&
					(i + currentBlockOfPoints * blockSize < count) && // not over the right side
					(i + currentBlockOfPoints * blockSize != idx)) // not compare with the current point
				{
					distanceToCloest = dist;
					indexOfClosest = i + currentBlockOfPoints * blockSize;
				}
			}
		}
		// the fast points in current block might have finished caculate the distance
		// ensure all of the 1024threads in this block are finished
		// we are read to copy another chunck of points
		__syncthreads();	
	}

	if (idx < count)
	{
		indices[idx] = indexOfClosest;
	}
}

