#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <complex>
#include <iomanip>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace Eigen;

VectorXcd init_psi(int N, double x_0, double k, double sigma, double dx) {

    VectorXcd result(N);
    double norm, realPart, imagPart;

    norm = std::pow((2 * M_PI) * (std::pow(sigma, 2)), -0.25);

    for (int i = 0; i < N; i++) {
        realPart = -(i * dx - x_0) * (i * dx - x_0) / (4 * sigma * sigma);
        imagPart = k * i * dx;

        result(i) = norm * std::exp(std::complex<double>(realPart, 0)) * std::exp(std::complex<double>(0, imagPart));
    }

    return result;
}

VectorXcd exact_psi(double x_0, double k, double sigma, double dt, int Nx, int Nt, double dx) {
    VectorXcd result(Nx);
    complex<double> norm, gaussian_term, phase_term;
    double time;
    complex<double> zi(0.0, 1.0);

    for (int i = 0; i < Nx; i++) {
        time = Nt * dt;
        norm = pow(pow(2.0 * M_PI, 0.25) * sqrt(sigma + (time * zi) / (2.0 * sigma)), -1);
        gaussian_term = -(pow((dx * i - x_0 - k * time),2))/ (4 * pow(sigma, 2) + 2.0 * zi * time) + (zi*k*(dx * i - k*time/2.0));
        result[i] = norm * exp(gaussian_term);
        
    }
    return result;
}

VectorXd generateEvenlySpacedIntervals(double start, double end, double interval) {
    int size = static_cast<int>((end - start) / interval) + 1;
    VectorXd intervals(size);

    for (int i = 0; i < size; i++) {
        intervals(i) = start + i * interval;
    }

    return intervals;
}

void thomas_tridiag(const VectorXcd& a, const VectorXcd& b, const VectorXcd& c, const VectorXcd& psi, VectorXcd& u) {
    int j, n;
    complex<double> bet;

    n = a.size();

    VectorXcd gam(n); 

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

    std::complex<double> zi(0.0, 1.0);

    double x_0, sigma, k;

    x_0   = 5.0;
    k     = 5.0;
    sigma = 0.25;
    double h = 1;
    double m = 1;

    double start_x, end_x, dx;
    start_x   = 0;
    end_x     = 20;
    dx       = 0.05;

    double start_t, end_t, dt;
    start_t = 0;
    end_t  = 1;
    dt = dx/20;

    VectorXd x_i = generateEvenlySpacedIntervals(start_x, end_x, dx);
    VectorXd t_n = generateEvenlySpacedIntervals(start_t, end_t, dt);

    int Nx, Nt;
    Nx = x_i.size();
    Nt = t_n.size();

    VectorXcd a(Nx - 1);
    a.setConstant(complex<double>(0, -dt * h / (4 * m * pow(dx, 2))));

    VectorXcd b(Nx);
    //b.setConstant(1.0 + r);
    b.setConstant(complex<double>(1, dt * h / (2 * m * pow(dx, 2))));

    VectorXcd c(Nx - 1);
    c.setConstant(complex<double>(0, -dt * h / (4 * m * pow(dx, 2))));

    VectorXcd f(Nx);
    f.setZero(); 
    VectorXcd psi = init_psi(Nx, x_0, k, sigma, dx);

    VectorXcd final(Nx);


     for (int i = 0; i < Nt; ++i) {

        thomas_tridiag(a, b, c, psi, f);

        VectorXcd final = 2*f - psi;

        psi = final;
    }


    ofstream outfile;
    outfile.open("cn.csv");

    double nrm_t;
    VectorXd num_sol(Nx);

    for (int i=0; i < Nx; i++){ 
        nrm_t = abs(psi[i]) * abs(psi[i]);
        num_sol[i] = nrm_t;
        outfile << dx * i << "," << nrm_t << endl;
    }

    outfile.close();

    VectorXcd exact_psi_result = exact_psi(x_0, k, sigma, dt, Nx, Nt, dx);

    std::ofstream outputFile3("ex-psi-initial-norm2.csv");
    if (!outputFile3.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return 1;  
    }

    outputFile3 << std::scientific;

    double exact_psi_abs, realPart_tc, imagPart_tc;
    VectorXd exact_sol(Nx);
    for (int i = 0; i < Nx; i++){

      exact_psi_abs = abs(exact_psi_result[i]) * abs(exact_psi_result[i]); 
      exact_sol[i] = exact_psi_abs;
      
      outputFile3 << dx * i << ',' << exact_psi_abs << endl;
    }

    cout << "end" << endl;

    return 0;
}