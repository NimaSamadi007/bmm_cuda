#include "bmm.h"
#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z


#define TILEX 32
#define TILEY 32
// check TILEX and TILEY value for optimal TILE assignment 
// consider that this part is done in preprosecing and in
// compile time. so this assignment doesn't have any overhead.
// these DIV values are optimal values that produce minimum
// time for matrix multiplication
const int DIV = (TILEX == 4 && TILEY == 4) ? TILEX / 4 :
		(TILEX == 4 && TILEY == 8) ? TILEX / 2 :
		(TILEX == 4 && TILEY == 16) ? TILEX :
		(TILEX == 4 && TILEY == 32) ? TILEX :
		(TILEX == 8 && TILEY == 4) ? TILEX / 4 :
		(TILEX == 8 && TILEY == 8) ? TILEX / 2 :
		(TILEX == 8 && TILEY == 16) ? TILEX :
		(TILEX == 8 && TILEY == 32) ? TILEX :
		(TILEX == 16 && TILEY == 4) ? TILEX / 8 :
		(TILEX == 16 && TILEY == 8) ? TILEX / 4 :
		(TILEX == 16 && TILEY == 16) ? TILEX :
		(TILEX == 16 && TILEY == 32) ? TILEX :
		(TILEX == 32 && TILEY == 4) ? TILEX / 8 :
		(TILEX == 32 && TILEY == 8) ? TILEX / 8 :
		(TILEX == 32 && TILEY == 16) ? TILEX / 2 : TILEX / 4;

// with repsect to DIV, assign TILE size
const int T = (TILEX * TILEY) / DIV;

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
	// shared memory def:
	__shared__ float as[TILEY][T];
	__shared__ float bs[T][TILEX];
	
	// number of read for each of matrices
	int Ra = TILEY / DIV;
	int Rb = TILEX / DIV;
	
	//global index
	int i = ty + by * TILEY;
	int j = tx + bx * TILEX;
		
	float s = 0;
	for(int k = 0; k < n / T; k++){		
		// as read:
		for(int m = 0; m < Ra; m++)
			as[ty][Ra * tx + m] = ad[(i * n) + Ra * tx + k * T + m];		
		// bs read:
		for(int m = 0; m < Rb; m++)
			bs[Rb * ty + m][tx] = bd[(ty * Rb + k * T + m) * n + j];
		__syncthreads();
		// calculation
		for (int m = 0; m < T; m++)
			s += as[ty][m] * bs[m][tx];
		__syncthreads();
	}
	cd[i * n + j] = s;
}

