#include <iostream>
#include <cmath>
#include <limits>
using namespace std;
struct float3d
{
	float x, y, z;
};

void FindCloestCPU(float3d* points, int* indices, int count)
{
	if (count <= 1) return;
	for (int curPoint = 0; curPoint < count; curPoint++)
	{
		float disToCloest = numeric_limits<float>::max();
		for (int i = 0; i < count; i++)
		{
			if (i == curPoint) continue;
			float dist = (
				(points[curPoint].x - points[i].x) *
				(points[curPoint].x - points[i].x) +
				(points[curPoint].y - points[i].y) *
				(points[curPoint].y - points[i].y) +
				(points[curPoint].z - points[i].z) *
				(points[curPoint].z - points[i].z));
			if (dist < disToCloest)
			{
				disToCloest = dist;
				indices[curPoint] = i;
			}
		}
	}
	return;
}