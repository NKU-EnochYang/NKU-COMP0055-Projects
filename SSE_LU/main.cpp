#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <pmmintrin.h>
using namespace std;

const int N = 2000;

float mat[N][N];
float test[N][N];

void init_mat(float mat[][N])
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = rand() / 100;
}

void reset_mat(float test[][N], float mat[][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            test[i][j] = mat[i][j];
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

void sse_lu(float mat[][N])
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_loadu_ps(temp);
            int j = k + 1;
            for (j; j < N - 3; j += 4)
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
void sse_lu_aligned(float mat[][N])
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp[4] __attribute__((aligned(16))) = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_load_ps(temp);
            int j = k + 1;
            for (j; j % 4 != 0; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            for (j; j < N - 3; j += 4)
            {
                t2 = _mm_load_ps(mat[i] + j);
                t3 = _mm_load_ps(mat[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_store_ps(mat[i] + j, t2);
            }
            for (j; j < N; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}

void sse_lu_vectorized(float mat[][N])
{
    __m128 t1, t2, t3;
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
        for (j; j < N; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
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
/*
void sse_lu_aligned_vectorized(float mat[][N])
{
    __m128 t1, t2, t3;
    for(int k = 0;k<N;k++)
    {
        float temp1[4] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
        t1 = _mm_load_ps(temp1);
        for(int j = N - 4;j >= k + 1;j-=4)
        {
            t2 = _mm_load_ps(mat[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_store_ps(mat[k] + j, t3);
        }
        for(int j = (N % 4) - 1;j >= k + 1;j--)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
        for(int i = k + 1;i<N;i++)
        {
            float temp2[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm_load_ps(temp2);
            for(int j = N - 4;j >= k + 1;j-=4)
            {
                t2 = _mm_load_ps(mat[i] + j);
                t3 = _mm_load_ps(mat[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_store_ps(mat[i] + j, t2);
            }
            for(int j = (N % 4) - 1;j >= k + 1;j--)
                mat[i][j] = mat[i][j] - mat[i][k]*mat[k][j];
            mat[i][k] = 0;
        }
    }
}
*/
void avx_lu(float mat[][N])
{
    __m256 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp[8] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm256_loadu_ps(temp);
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
}

void avx_lu_aligned(float mat[][N])
{
    __m256 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            mat[k][j] = mat[k][j] / mat[k][k];
        mat[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            int j = k + 1;
            for (j; j % 8 != 0; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            float temp[8] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm256_load_ps(temp);
            for (j; j < N - 7; j += 8)
            {
                t2 = _mm256_load_ps(mat[i] + j);
                t3 = _mm256_load_ps(mat[k] + j);
                t3 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_store_ps(mat[i] + j, t2);
            }
            for (j; j < N; j++)
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            mat[i][k] = 0;
        }
    }
}

void avx_lu_vectorized(float mat[][N])
{
    __m256 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        float temp1[8] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
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
        for (int i = k + 1; i < N; i++)
        {
            float temp2[8] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm256_loadu_ps(temp2);
            j = k + 1;
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
}

/*
void avx_lu_aligned_vectorized(float mat[][N])
{
    __m256 t1, t2, t3;
    for(int k = 0;k<N;k++)
    {
        float temp1[8] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
        t1 = _mm256_load_ps(temp1);
        for(int j = N - 4;j >= k + 1;j-=4)
        {
            t2 = _mm256_load_ps(mat[k] + j);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(mat[k] + j, t3);
        }
        mat[k][k] = 1.0;
        for(int i = k + 1;i<N;i++)
        {
            float temp2[8] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
            t1 = _mm256_load_ps(temp2);
            for(int j = N - 8;j >= k + 1;j-=8)
            {
                t2 = _mm256_load_ps(mat[i] + j);
                t3 = _mm256_load_ps(mat[k] + j);
                t3 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_store_ps(mat[i] + j, t2);
            }
            for(int j = (N % 8) - 1;j >= k + 1;j--)
                mat[i][j] = mat[i][j] - mat[i][k]*mat[k][j];
            mat[i][k] = 0;
        }
    }
}
*/
void print_mat(float mat[][N])
{
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
    int epoch = 10;
    long long head, tail, freq; // timers
    long double timer;
    init_mat(mat);

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);   // start time
        naive_lu(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "naive lu cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);   // start time
        sse_lu(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse lu cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);   // start time
        sse_lu_aligned(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse lu aligned cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);   // start time
        sse_lu_vectorized(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse lu vectorized cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);   // start time
        avx_lu(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "avx lu cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        avx_lu_aligned(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "avx lu aligned cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset_mat(test, mat);
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq); // similar to CLOCKS_PER_SEC
        QueryPerformanceCounter((LARGE_INTEGER *)&head);   // start time
        avx_lu_vectorized(test);
        QueryPerformanceCounter((LARGE_INTEGER *)&tail); // end time
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "avx lu vectorized cost: " << timer / epoch << "ms" << endl;
    return 0;
}
