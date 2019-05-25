#include <iostream>
#include <windows.h>

using namespace std;

const int N = 1000;		// matrix size

double b[N][N];
double general_dot_product[N];
double cache_dot_product[N];
double vec[N];

void init(int n)			// generate a N*N matrix
{
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      b[i][j] = i + j;
}

void GeneralDotProduct(double mat[N][N], double vec[])
{
    for(int i = 0;i<N;i++)
    {
        general_dot_product[i] = 0.0;
        for(int j = 0;j<N;j++)
        {
            general_dot_product[i] += mat[j][i] * vec[j];
        }
    }
}

void CacheDotProduct(double mat[N][N], double vec[])
{
    for(int i = 0;i<N;i++)
    {
        cache_dot_product[i] = vec[i];
    }
    for(int i = 0;i<N;i++)
    {
        for(int j = 0;j<N;j++)
        {
            cache_dot_product[i] *= mat[i][j];
        }
    }
}

int main()
{
    long long head, tail, freq;        // timers

	init(N);
	for(int i = 0;i<N;i++)
        vec[i] = i+1;

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
    GeneralDotProduct(b, vec);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	double time1 = (tail - head) * 1000.0 / freq;
	cout << "General dot product: " << time1 << "ms" << endl;

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
    CacheDotProduct(b, vec);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	double time2 = (tail - head) * 1000.0 / freq;
	cout << "Cache dot product: " << time2 << "ms" << endl;

	cout << "Cache algorithm performs " << time1/time2 << " times than General algorithm"<<endl;
	return 0;
}
