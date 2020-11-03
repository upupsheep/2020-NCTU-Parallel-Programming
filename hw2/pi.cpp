#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <pthread.h>
using namespace std;

typedef long long int lli;

// global variables
int cpu_cores;

double MonteCarlo_Pi(lli number_of_tosses) {
    long thread;  // Use long in case of a 64-bit system
    pthread_t* thread_handles;
    thread_handles = (pthread_t*)malloc(cpu_cores * sizeof(pthread_t));

    srand(time(NULL));
    double number_in_circle = 0.0;
    for (int toss = 0; toss < (number_of_tosses / cpu_cores); toss++) {
        // random x and y
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4.0 * number_in_circle / ((double)number_of_tosses);
    return pi_estimate;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Error! There should be 3 arguments." << endl;
        return 0;
    }
    cpu_cores = atoi(argv[1]);
    lli number_of_tosses = atoll(argv[2]);

    double pi = MonteCarlo_Pi(number_of_tosses);
    cout << pi << endl;
    return 0;
}