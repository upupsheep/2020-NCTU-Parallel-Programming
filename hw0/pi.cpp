#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace std;

typedef long long int lli;

double MonteCarlo_Pi(lli number_of_tosses){
    srand(time(NULL)); 
    double number_in_circle = 0.0;
    for (int toss = 0; toss < number_of_tosses; toss ++) {
        // random x and y
        double x = (double) rand() / RAND_MAX; 
        double y = (double) rand() / RAND_MAX;
        
        double distance_squared = x * x + y * y;
        /*
        cout << "------toss #" << toss << "-----" << endl;
        cout << "x = " << x << ", y = " << y << endl;
        cout << "distance_squared: " << distance_squared << endl;
        cout << "--------------------------" << endl;
        */
        if ( distance_squared <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4.0 * number_in_circle /(( double ) number_of_tosses);
    return pi_estimate;
}

int main(){
    double pi = MonteCarlo_Pi(60000);
    cout << pi << endl;
    return 0;
}