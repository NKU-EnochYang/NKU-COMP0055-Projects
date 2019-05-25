#include <iostream>
#include <windows.h>

using namespace std;

const int N = 4096;

int serial_cal(int num[])
{
    int sum = 0;
    for(int i = 0;i<N;i++)
        sum += num[i];
    return sum;
}

int recursive_cal(int num[], int starts, int ends)
{
    if(starts == ends)
        return num[ends];
    else
    {
        int mid = (starts + ends)/2;
        return recursive_cal(num, starts, mid) + recursive_cal(num, mid + 1, ends);
    }
}

int new_cal(int a[], int n)
{
    int i, j;
    while(n >= 8)
    {
        n /= 2;
        j = 0;
        for(i = 0; j<n;i += 8)
        {
            a[j++] = a[i] + a[i+1];
            a[j++] = a[i+2] + a[i+3];
            a[j++] = a[i+4] + a[i+5];
            a[j++] = a[i+6] + a[i+7];
        }
    }
    return a[0] + a[1] + a[2] + a[3];
}
int iteration_cal(int num[])
{
    int cnt = N/2;
    while(cnt != 1)
    {
        for(int i = 0;i<cnt;i++)
            num[i] = num[2*i] + num[2*i + 1];
        cnt /= 2;
    }
    return num[0] + num[1];
}

int iteration_cal_unroll(int num[])
{
    int cnt = N/2;
    while(cnt != 2)
    {
        for(int i = 0;i<cnt;i+=4)
        {
            num[i] = num[2*i] + num[2*i + 1];
            num[i+1] = num[2*i + 2] + num[2*i + 3];
            num[i+2] = num[2*i + 4] + num[2*i + 5];
            num[i+3] = num[2*i + 6] + num[2*i + 7];
        }
        cnt /= 2;
    }
    return num[0] + num[1] + num[2] + num[3];
}

int main()
{
    int num[N];
    int num1[N];
    for(int i = 0;i<N;i++)
    {
        num[i] = 2*i;
        num1[i] = 2*i;
    }
    long long sum1 = 0;
    long long sum2 = 0;
    long long sum3 = 0;
    long long sum4 = 0;
    long long head, tail, freq;        // timers

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
    sum1 = serial_cal(num);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "Serial calculation results: " << sum1 <<endl;
	cout << "Serial calculation costs: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	/*
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
 	sum2 = recursive_cal(num, 0, N);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "Recursive calculation results: " << sum2 <<endl;
	cout << "Recursive calculation costs: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	*/
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
 	sum4 = new_cal(num, N);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "Unrolled iteration calculation results: " << sum4 <<endl;
	cout << "Unrolled iteration calculation costs: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	/*
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
 	sum3 = iteration_cal(num);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "Iteration calculation results: " << sum3 <<endl;
	cout << "Iteration calculation costs: " << (tail - head) * 1000.0 / freq << "ms" << endl;

	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);	// similar to CLOCKS_PER_SEC
 	QueryPerformanceCounter((LARGE_INTEGER *)&head);	// start time
 	sum4 = iteration_cal_unroll(num1);
	QueryPerformanceCounter((LARGE_INTEGER *)&tail);	// end time
	cout << "Unrolled iteration calculation results: " << sum4 <<endl;
	cout << "Unrolled iteration calculation costs: " << (tail - head) * 1000.0 / freq << "ms" << endl;
	*/
    return 0;
}
