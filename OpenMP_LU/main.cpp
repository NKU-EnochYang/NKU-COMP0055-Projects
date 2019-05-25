#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include <omp.h>

using namespace std;

const int N = 2000;

float mat[N][N];
float test[N][N];

const int thread_count = 4;
long long head, tail, freq;        // timers

void init_mat(float test[][N])
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			test[i][j] = rand() / 100;
}

void reset_mat(float mat[][N], float test[][N])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			mat[i][j] = test[i][j];
}

void naive_lu(float mat[][N])
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
			mat[k][j] = mat[k][j] / mat[k][k];
		mat[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void omp_lu(float mat[][N])
{
    #pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
	    #pragma omp for schedule(static)
		for (int j = k + 1; j < N; j++)
		{
		    mat[k][j] = mat[k][j] / mat[k][k];
		}
		mat[k][k] = 1.0;
		#pragma omp for schedule(static)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void omp_lu_dynamic(float mat[][N])
{
    #pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
	    #pragma omp for schedule(dynamic, 24)
		for (int j = k + 1; j < N; j++)
        {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
		mat[k][k] = 1.0;
		#pragma omp for schedule(dynamic, 24)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void omp_lu_guided(float mat[][N])
{
    #pragma omp parallel num_threads(thread_count)
	for (int k = 0; k < N; k++)
	{
	    #pragma omp for schedule(guided, 24)
		for (int j = k + 1; j < N; j++)
        {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
		mat[k][k] = 1.0;
		#pragma omp for schedule(guided, 24)
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
				mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
			mat[i][k] = 0;
		}
	}
}

void omp_lu_sse_dynamic(float mat[][N])
{
    __m128 t1, t2, t3;
    #pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        float temp1[4] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
        t1 = _mm_loadu_ps(temp1);
        int j = k + 1;
        for (j; j < N - 3; j += 4)
        {
            t2 = _mm_loadu_ps(mat[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(mat[k] + j, t3);
        }
        #pragma omp for schedule(dynamic, 24)
        for (j; j < N; j++)
        {
            mat[k][j] = mat[k][j] / mat[k][k];
        }
        mat[k][k] = 1.0;
        #pragma omp for schedule(dynamic, 24)
        for (int i = k + 1; i < N; i++)
        {
            float temp2[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_loadu_ps(temp2);
            j = k + 1;
            for (j; j <= N - 3; j += 4)
            {
                t2 = _mm_loadu_ps(mat[i] + j);
                t3 = _mm_loadu_ps(mat[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(mat[i] + j, t2);
            }
            for (j; j < N; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}

void print_mat(float mat[][N])
{
    if (N > 16)
        return;
	cout << endl;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			cout << mat[i][j] << " ";
		cout << endl;
	}
	cout << endl;
}

int main()
{
    init_mat(test);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	naive_lu(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "naive LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	omp_lu(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "blocked OpenMP LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	omp_lu_dynamic(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "dynamic OpenMP LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	omp_lu_guided(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "guided OpenMP LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	omp_lu_sse_dynamic(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "sse and dynamic OpenMP LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	cout << endl;
    return 0;
}
