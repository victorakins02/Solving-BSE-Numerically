#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <complex>
#include <iomanip>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace Eigen;

//Initial Values of stocks
VectorXd init_V(double K, double dS, int Nt, int Nx) {

    VectorXd V(Nt - 1);

    for(int j = 0; j < (Nt - 1); j++) {
        V[j] = std::max(K - dS * (j+1), 0.0);
    }

    return V;
}

//Thomas Algorithim
void thomas_tridiag(const VectorXd& a, const VectorXd& b, const VectorXd& c, const VectorXd& psi, VectorXd& u) {
    int j, n;
    double bet;

    n = a.size();

    VectorXd gam(n); 

    if (b[0] == 0.0) {
        throw("Error 1 in tridag");
        
    }

    u[0] = psi[0] / (bet = b[0]);

    for (j = 1; j < n; j++) { 

        gam[j] = c[j - 1] / bet;
        bet = b[j] - a[j] * gam[j];

        if (bet == 0.0)
            throw("Error in tridag"); 

        u[j] = (psi[j] - a[j] * u[j - 1]) / bet;
    }

    for (j = (n - 2); j >= 0; j--)
        u[j] -= gam[j + 1] * u[j + 1]; 
}


int main() {
//Variables
    int N  = 10;
    int M  = 20; 
    double T = 0.4167;
    double Smax = 100; 
    double r = 0.1; 
    double sd = 0.4; 
    double K = 50.0; 
    double dS = Smax/M;
    double dt = T/N;
    


//Initialize Matrices
    VectorXd a(M - 1);
    VectorXd b(M);
    VectorXd c(M - 1);

    double sd2 = std::pow(sd,2);
    for(int j = 0; j < (M-1); j++) {
        a[j] = 0.25 * (j+1) * dt * (sd2 * (j+1) - r); 
        c[j] = 0.25 * (j+1) * dt * (sd2 * (j+1) + r); 
        b[j] = (1 - 0.5 * std::pow(sd * (j+1), 2) * dt);
    }



//Empty Vector / Becomes Output Vector
    VectorXd f(M-1);
    f.setZero();
    VectorXd final(M-1);
    VectorXd start_psi(M);
    VectorXd k(M-1);
    k.setZero();

//Initial Values Set
    VectorXd psi = init_V(K, dS, M, N);
    MatrixXd T2 = MatrixXd::Zero(M-1, M-1);

    for(int j = 0; j < (M-1); j++) { 
        T2(j, j) = b[j];
        if (j < M-2) {
            T2(j, j+1) = c[j];
            T2(j+1, j) = a[j+1];
        }
    }

    k[0] = a[0] * (2.0 * K);

    VectorXd init_psi = T2 * psi;
    init_psi = init_psi + k;

//Matrix Coefficients
    VectorXd a2(M - 1);
    VectorXd d2(M);
    VectorXd c2(M - 1);    

    for (int i = 0; i < (M-1); i++) {
        a2[i] = -(0.25 * (i+1) * dt * (sd2 * (i+1) - r));
        d2[i] = (1 + (r + 0.5 * std::pow(sd * (i+1), 2)) * dt);
        c2[i] = -(0.25 * (i+1) * dt * (sd2 * (i+1) + r));
    }

    std::cout << std::setw(7) << "Stock:";
        for(int i = 0; i < (M+1); i++) {
        std::cout << std::setw(6) <<  std::fixed << std::setprecision(2) << std::setprecision(1) << i * dS;
        }

        std::cout << std::endl << std::setw(7) << "Time:";
        for(int i = 0; i < (M+1); i++) {
            std::cout << std::setw(6) << "------";
        }

        std::cout << std::endl;

        std::cout << std::setw(6) << std::setprecision(2) << N * dt << "| ";
        std::cout << std::setw(6) << K;

        for(int j = 0; j < (M-1); j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << psi[j];
        }

        std::cout << std::setw(6) << "0.00" << std::endl;

    
    for(int j = N; j > 0; j--){

        thomas_tridiag(a2, d2, c2, init_psi, f);

        init_psi = f;

        std::cout << std::setw(6) << std::setprecision(2) << (j-1) * dt << "| ";
        std::cout << std::setw(6) << K;

        
        for(int i = 0; i < (M - 1); i++) {
                init_psi[i] = max(init_psi[i], K - dS * (i+1));
                std::cout << std::setw(6) << std::fixed << std::setprecision(2) << init_psi[i];
            }
        std::cout << std::setw(6) << "0.00" << std::endl;

        init_psi = T2 * init_psi;
        init_psi = init_psi + k;

        }
    
    return 0;
}