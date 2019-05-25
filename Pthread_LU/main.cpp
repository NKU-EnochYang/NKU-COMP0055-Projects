#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
#include <algorithm>
#include <pthread.h>
using namespace std;

const int N = 2000;

float mat[N][N];
float test[N][N];

typedef struct
{
	int	threadId;
} threadParm_t;

const int thread_num = 8;
//const int seg = N / thread_num;
long long head, tail, freq;        // timers
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_task = PTHREAD_MUTEX_INITIALIZER;
pthread_t threads[thread_num];
threadParm_t threadParm[thread_num];
pthread_barrier_t barrier1;
pthread_barrier_t barrier2;
const int task = 1;
const int seg = task * thread_num;


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

void *block_pthread(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    int id = p->threadId;
    int s = id * (N / thread_num);
    int e = (id + 1) * (N / thread_num);

	for (int k = 0; k < N; k++)
	{
//	    int s = k + 1 + id * ((N - k - 1) / thread_num);
//        int e = k + 1 + (id + 1) * ((N - k - 1) / thread_num);
        int block = (N - k - 1) / thread_num;
	    if (k + 1 >= s && k < e)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
        }
        for (int j = k + 1 + id * block; j < k + 1 + (id + 1) * block; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        pthread_barrier_wait(&barrier1);
        mat[k][k] = 1.0;
//        for (int i = k + 1; i < N; i++)
//        {
//            if (i >= s && i < e)
//            {
//                for (int j = k + 1; j < N; j++)
//                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
//                mat[i][k] = 0;
//            }
//        }
        for (int i = k + 1 + id * block; i < k + 1 + (id + 1) * block; i++)
        {
            for (int j = k + 1; j < N; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
		pthread_barrier_wait(&barrier2);
	}
//	pthread_mutex_lock(&mutex);
//	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//	cout << "Block pthread " << id << ": " << (tail - head) * 1000.0 / freq << "ms" << endl;
//	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}

void *recycle_pthread_lu(void *parm)
{
	threadParm_t *p = (threadParm_t *)parm;
	int id = p->threadId;
    int s = id * task;
	int e = (id + 1) * task;
	for (int k = 0; k < N; k++)
	{
	    if ((k + 1) % seg >= s && (k + 1) % seg < e)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
        }
        pthread_barrier_wait(&barrier1);
        for (int i = k + 1; i < N; i++)
        {
            if (i % seg >= s && i % seg < e)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
        }
		pthread_barrier_wait(&barrier2);
	}
//	pthread_mutex_lock(&mutex);
//	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//	cout << "Recycle pthread " << id << ": " << (tail - head) * 1000.0 / freq << "ms" << endl;
//	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}

void *recycle_pthread_sse_lu(void *parm)
{
	threadParm_t *p = (threadParm_t *)parm;
	int id = p->threadId;
	int s = id * task;
	int e = (id + 1) * task;
	__m128 t1, t2, t3;
	for (int k = 0; k < N; k++)
	{
		if ((k + 1) % seg >= s && (k + 1) % seg < e)
		{
			float temp1[4] = { mat[k][k], mat[k][k], mat[k][k], mat[k][k] };
			t1 = _mm_loadu_ps(temp1);
			int j = k + 1;
			for (j; j < N - 3; j += 4)
			{
				t2 = _mm_loadu_ps(mat[k] + j);
				t3 = _mm_div_ps(t2, t1);
				_mm_storeu_ps(mat[k] + j, t3);
			}
			for (j; j < N; j++)
				mat[k][j] = mat[k][j] / mat[k][k];
			mat[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1; i < N; i++)
		{
			if (i % seg >= s && i % seg < e)
			{
				float temp2[4] = { mat[i][k], mat[i][k], mat[i][k], mat[i][k] };
				t1 = _mm_loadu_ps(temp2);
				int j = k + 1;
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
		pthread_barrier_wait(&barrier2);
	}
	pthread_exit(NULL);
}

void *recycle_pthread_avx_lu(void *parm)
{
	threadParm_t *p = (threadParm_t *)parm;
	int id = p->threadId;
	int s = id * task;
	int e = (id + 1) * task;
	__m256 t1, t2, t3;
	for (int k = 0; k < N; k++)
	{
		if ((k + 1) % seg >= s && (k + 1) % seg < e)
		{
			float temp1[8] = { mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k] };
			t1 = _mm256_loadu_ps(temp1);
			int j = k + 1;
			for (j; j < N - 7; j += 8)
			{
				t2 = _mm256_loadu_ps(mat[k] + j);
				t3 = _mm256_div_ps(t2, t1);
				_mm256_storeu_ps(mat[k] + j, t3);
			}
			for (j; j < N; j++)
				mat[k][j] = mat[k][j] / mat[k][k];
			mat[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier1);
		for (int i = k + 1; i < N; i++)
		{
			if (i % seg >= s && i % seg < e)
			{
				float temp2[8] = { mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k] };
				t1 = _mm256_loadu_ps(temp2);
				int j = k + 1;
				for (j; j < N - 7; j += 8)
				{
					t2 = _mm256_loadu_ps(mat[i] + j);
					t3 = _mm256_loadu_ps(mat[k] + j);
					t3 = _mm256_mul_ps(t1, t3);
					t2 = _mm256_sub_ps(t2, t3);
					_mm256_storeu_ps(mat[i] + j, t2);
				}
				for (j; j < N; j++)
					mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
				mat[i][k] = 0;
			}
		}
		pthread_barrier_wait(&barrier2);
	}
	//	pthread_mutex_lock(&mutex);
	//	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	//	cout << "recycle avx thread " << id << ": " << (tail - head) * 1000.0 / freq << "ms" << endl;
	//	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
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
    pthread_barrier_init(&barrier1, NULL, thread_num);
    pthread_barrier_init(&barrier2, NULL, thread_num);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
	init_mat(test);
//	cout << N << ", ";

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	naive_lu(mat);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//	cout << (tail - head) * 1000.0 / freq << ", ";
	cout << "naive LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (int i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, block_pthread, (void *)&threadParm[i]);
	}
	for (int i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//    cout << (tail - head) * 1000.0 / freq << ", ";
    cout << "block pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (long i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, recycle_pthread_lu, (void *)&threadParm[i]);
	}
	for (long i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//	cout << (tail - head) * 1000.0 / freq << ", ";
	cout << "recycle pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (long i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, recycle_pthread_sse_lu, (void *)&threadParm[i]);
	}
	for (long i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//	cout << (tail - head) * 1000.0 / freq << ", ";
	cout << "recycle sse pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	reset_mat(mat, test);
	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
	for (long i = 0; i < thread_num; i++)
	{
		threadParm[i].threadId = i;
		pthread_create(&threads[i], NULL, recycle_pthread_avx_lu, (void *)&threadParm[i]);
	}
	for (long i = 0; i < thread_num; i++)
		pthread_join(threads[i], NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
//	cout << (tail - head) * 1000.0 / freq << endl;
	cout << "recycle avx pthread LU: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	print_mat(mat);

	pthread_mutex_destroy(&mutex);
	pthread_barrier_destroy(&barrier1);
	pthread_barrier_destroy(&barrier2);
	return 0;
}
