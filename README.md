# Solving-BSE-Numerically

Project Overview
This project numerically solves the Black-Scholes Equation, widely used in financial mathematics for pricing options, and compares it with the Time-Dependent Schrödinger Equation (TDSE) from quantum mechanics. By using the Crank-Nicolson Method, the project solves both equations, highlighting their mathematical similarities and providing accurate results for option pricing and quantum wave function evolution.

Features
Implements C++ code for numerical solutions using the Crank-Nicolson Method.
Compares numerical results with analytical solutions for accuracy verification.
Supports both explicit and implicit methods for solving the Black-Scholes and Schrödinger equations.
Python is used to plot the results from CSV files generated by the C++ implementation.
Efficient matrix-based approach to minimize computational costs.
Key Methods
Crank-Nicolson Method: A stable and accurate finite-difference approach used to solve the Black-Scholes and Schrödinger equations.
LU Decomposition: Solves tridiagonal systems in Crank-Nicolson I.
Thomas Algorithm: Used in Crank-Nicolson II to optimize performance by eliminating matrix-vector operations.
Dependencies
Eigen Library: For matrix and vector operations.
C++11 or later.
Python: To generate plots from the CSV files. The plotting script uses matplotlib.
Usage
To run the project, compile and execute the C++ code, ensuring the Eigen library is installed. After running the simulation, use the provided Python script to visualize the results from the generated CSV files.

C++ Example:
bash
Copy code
g++ -std=c++11 -I /path/to/eigen your_code.cpp -o output
./output
Python Example:
bash
Copy code
python plot_results.py
Files
main.cpp: The main C++ implementation for solving the equations using Crank-Nicolson methods.
data/: Contains the CSV files with numerical and analytical solutions.
plot_results.py: A Python script to plot results from the CSV files using matplotlib.
README.md: Project documentation and setup instructions.
