#include <iostream>
#include "Eigen/Dense"
#include <complex>
#include <iomanip>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace Eigen;

VectorXcd init_psi(const VectorXd& x, double x_0, double k, double sigma) {
    VectorXcd result(x.size());
    double norm, realPart, imagPart;

    norm = std::pow((2 * M_PI) * (std::pow(sigma, 2)), -0.25);

    for (int i = 0; i < x.size(); i++) {
        realPart = -(x(i) - x_0) * (x(i) - x_0) / (4 * sigma * sigma);
        
        imagPart = k * x(i);

        result(i) = norm * std::exp(std::complex<double>(realPart, 0)) * std::exp(std::complex<double>(0, imagPart));
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

VectorXcd exact_psi(const VectorXd& x, double x_0, double k, double sigma, double dt, int Nx, int Nt) {
    VectorXcd result(Nx);
    complex<double> norm, gaussian_term, phase_term;
    double time;
    complex<double> zi(0.0, 1.0);

    for (int i = 0; i < Nx; i++) {
        time = Nt * dt;
        norm = pow(pow(2.0 * M_PI, 0.25) * sqrt(sigma + (time * zi) / (2.0 * sigma)), -1);
        
        gaussian_term = -(pow((x(i) - x_0 - k * time),2))/ (4 * pow(sigma, 2) + 2.0 * zi * time) + (zi*k*(x[i] - k*time/2.0));
        result[i] = norm * exp(gaussian_term);
        
    }
    return result;
}

int main() {
    std::complex<double> zi(0.0, 1.0);

    double x_0, sigma, k;
    x_0   = 5.0;
    k     = 5.0;
    sigma = 0.25;

    double start_x, end_x, dx;
    start_x   = 0;
    end_x     = 20;
    dx       = 0.05;

    double start_t, end_t, dt;
    start_t = 0;
    end_t  = 1;
    dt = dx/20;

    double hb, m;
    hb = 1;
    m = 1;

    VectorXd x_i = generateEvenlySpacedIntervals(start_x, end_x, dx);
    VectorXd t_n = generateEvenlySpacedIntervals(start_t, end_t, dt);

    int Nx, Nt;
    Nx = x_i.size();
    Nt = t_n.size();

    MatrixXcd iden = MatrixXcd::Identity(Nx, Nx);

    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(Nx, Nx);
    Eigen::MatrixXcd V = Eigen::MatrixXcd::Zero(Nx, Nx);

    T.diagonal().array() = -2.0;
    T.diagonal(1).array() = T.diagonal(-1).array() = 1.0;
    V.diagonal().array() = 0;

    T *= ((-hb * hb) / (2.0 * m * dx * dx));

    Eigen::MatrixXcd H = T + V;

    Eigen::MatrixXcd FTCS = iden - ((std::complex<double>(0.0, 1.0) * dt / (2.0 * hb)) * H);
    Eigen::MatrixXcd BTCS = iden + ((std::complex<double>(0.0, 1.0) * dt / (2.0 * hb)) * H);

    MatrixXcd CN2 = FTCS * BTCS.inverse();

    VectorXcd psi = init_psi(x_i, x_0, k, sigma);
    VectorXcd psi_t; 
    
    for (int it = 0; it < Nt; it++) {
        psi_t = CN2 * psi ;
        psi = psi_t;
	
    }

    std::ofstream outputFile("psi-t-norm-cn.csv");

    if (!outputFile.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return 1;  
    }

    int precision = 3;
    outputFile << std::scientific;

    double nrm_t;
    VectorXd num_sol(Nx);


    for (int i = 0; i < Nx; i++) {

      nrm_t = abs(psi_t[i]) * abs(psi_t[i]);
      outputFile << x_i[i] << ',' << nrm_t << std::endl;
      num_sol[i] = nrm_t;

    }  
    outputFile.close();


    VectorXcd exact_psi_result = exact_psi(x_i, x_0, k, sigma, dt, Nx, Nt);
  
    std::ofstream outputFile1("ex-psi-initial-norm.csv");
    if (!outputFile1.is_open()) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return 1;  
    }

    outputFile1.precision(precision);
    outputFile1 << std::scientific;


    double exact_psi_abs, realPart_tc, imagPart_tc;
    VectorXd exact_sol(Nx);
    for (int i = 0; i < Nx; i++){

      exact_psi_abs = abs(exact_psi_result[i]) * abs(exact_psi_result[i]); 
      exact_sol[i] = exact_psi_abs;
      
      outputFile1 << x_i[i] << ',' << exact_psi_abs << endl;
    }

    outputFile1.close();

    cout << "end" << endl;
}