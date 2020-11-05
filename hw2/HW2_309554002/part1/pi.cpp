#include <pthread.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

typedef long long int lli;

// global variables
int cpu_cores;
lli number_of_tosses;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
lli total_in_circle = 0;
// lli* thread_in_circle;

void* MonteCarlo_Pi(void* threadId) {
    // long long_tid;
    // long_tid = (long)threadId;
    int tid = (int)threadId;

    unsigned int seed = time(NULL) + 20 * tid;
    lli number_in_circle = 0;

    int toss;
    double x, y, distance_squared;
    for (toss = 0; toss < number_of_tosses / cpu_cores; toss++) {
        // random x and y
        // double x = (double)rand() / RAND_MAX;
        // double y = (double)rand() / RAND_MAX;
        x = rand_r(&seed) / ((double)RAND_MAX);
        y = rand_r(&seed) / ((double)RAND_MAX);

        distance_squared = x * x + y * y;
        if (distance_squared <= 1.0)
            number_in_circle++;
    }

    // Let thread 0 calculate the remaining part
    /*
    if (tid == 0) {
        int remainder = toss % cpu_cores;
        for (toss = 0; toss < remainder; toss++) {
            double x = rand_r(&seed) / ((double)RAND_MAX);
            double y = rand_r(&seed) / ((double)RAND_MAX);

            double distance_squared = x * x + y * y;
            if (distance_squared <= 1)
                number_in_circle++;
        }
    }
    */
    pthread_mutex_lock(&mutex);
    total_in_circle += number_in_circle;
    pthread_mutex_unlock(&mutex);
    // pthread_exit(EXIT_SUCCESS);
    // double pi_estimate = 4.0 * number_in_circle / ((double)number_of_tosses);
    // return pi_estimate;
}

int main(int argc, char* argv[]) {
    // if (argc != 3) {
    //     cout << "Error! There should be 3 arguments." << endl;
    //     return 0;
    // }
    cpu_cores = atoi(argv[1]);
    number_of_tosses = atoll(argv[2]);
    // srand(time(NULL));

    // in_circle count in each thread
    // thread_in_circle = new long long int[cpu_cores];
    pthread_t thread_id[cpu_cores];

    for (int i = 0; i < cpu_cores; i++) {
        pthread_create(&thread_id[i], NULL, MonteCarlo_Pi, (void*)i);
    }

    for (int i = 0; i < cpu_cores; i++) {
        pthread_join(thread_id[i], NULL);
    }

    // Calculate pi
    double pi_estimate = 4.0 * total_in_circle / ((double)number_of_tosses);
    // cout << pi_estimate << endl;
    printf("%lf\n", pi_estimate);

    //pthread_exit(NULL);
    return 0;
}
