#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd get_WCA_force(MatrixXd current_state, double epsilon, double int_strength, double sigma) {
    const int N = current_state.rows();
    const double R=(N*sigma/2)+epsilon; // R should be the number of disks plus epsilon, if each disk radius (current_state(0,0)) is 1
    VectorXd x = current_state.col(0);
    VectorXd y = current_state.col(1);
    VectorXd r = (x.array().square()+y.array().square()).array().sqrt(); // vector of the radial distances for each disk from the center

    // compute distances between all pairs of disks (including disk with itself, to make the math easy)
    MatrixXd distances(N, N); 
    VectorXd wall_dist(N);
    for (size_t i=0; i<N; ++i){
        //populate (i,i) elements with zeros (distance from disk to itself is zero)
        distances(i,i) = 0;

        // each disk's nearest distance to the enclosure is R-r
        wall_dist(i) = sigma/2+R-r(i);
    }

    // loop over all unique disk pairs
    for (size_t disk_i=0; disk_i < N; ++disk_i) {
        for (size_t disk_j=disk_i+1; disk_j < N; ++disk_j) {
            // calculate distance between disk_i and disk_j
            double rij = std::sqrt(std::pow((x(disk_i)-x(disk_j)),2)+std::pow((y(disk_i)-y(disk_j)),2));
            // populate (disk_i,disk_j) and (disk_j,disk_i) with distance
            distances(disk_i,disk_j) = rij;
            distances(disk_j,disk_i) = rij;
        }
    }

    // calculate x and y values of the force produced by the various WCA potentials, scaled by kT
    MatrixXd forces(N,2); 
    for (size_t disk_i=0; disk_i < N; ++disk_i){
        double sum_grad_Ux=0;
        double sum_grad_Uy=0;
        for (size_t disk_j=0; disk_j < N; ++disk_j) {
            if (!((disk_j==disk_i) || (distances(disk_i,disk_j)/sigma)>std::pow(2,1.0/6.0))) {
                // no contribution to the force for self-interactions, or if the disks are outside range of the WCA potential
                sum_grad_Ux += -4*int_strength*(-(12/distances(disk_i,disk_j))*std::pow((sigma/distances(disk_i,disk_j)),12) \
                +(6/distances(disk_i,disk_j))*std::pow((sigma/distances(disk_i,disk_j)),6))*((x(disk_j)-x(disk_i))/distances(disk_i,disk_j)); // minus sign out front is because my gradient calculates force FROM disk_i TO disk_j, and I need to add opposite
                sum_grad_Uy += -4*int_strength*(-(12/distances(disk_i,disk_j))*std::pow((sigma/distances(disk_i,disk_j)),12) \
                +(6/distances(disk_i,disk_j))*std::pow((sigma/distances(disk_i,disk_j)),6))*((y(disk_j)-y(disk_i))/distances(disk_i,disk_j));
            }
        }

        //now add the force produced by the wall
        if (!((wall_dist(disk_i)/sigma)>std::pow(2,1.0/6.0))) {
            double wall_xdist = (sigma/2+R)*x(disk_i)/r(disk_i);
            double wall_ydist = (sigma/2+R)*y(disk_i)/r(disk_i);
            sum_grad_Ux += -4*int_strength*(-(12/wall_dist(disk_i))*std::pow((sigma/wall_dist(disk_i)),12) \
            +(6/wall_dist(disk_i))*std::pow((sigma/wall_dist(disk_i)),6))*((wall_xdist-x(disk_i))/wall_dist(disk_i)); // minus sign out front is because my gradient calculates force FROM disk_i TO wall
            sum_grad_Uy += -4*int_strength*(-(12/wall_dist(disk_i))*std::pow((sigma/wall_dist(disk_i)),12) \
            +(6/wall_dist(disk_i))*std::pow((sigma/wall_dist(disk_i)),6))*((wall_ydist-y(disk_i))/wall_dist(disk_i));
        }
        forces(disk_i, 0) = -sum_grad_Ux;
        forces(disk_i, 1) = -sum_grad_Uy;
    }
    return forces;
}

// Function to compute the phi values at all nsamp times for all N disks
MatrixXd phi_active(MatrixXd X, double f, double Tmax, int nsamp, double Dr) {
    const int N = X.rows();
    const double h = Tmax/nsamp; // h is the same thing as dt=Tmax/nsamp
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> uni_dist(0, 1); // uniform distribution between 0 and 1
    static std::normal_distribution<double> norm_dist(0.0, 1.0); // defines standard normal distribution

    MatrixXd phi(nsamp,N);
    for (int disk_id=0; disk_id<N; disk_id++) {
        // randomly orient the initial phi
        phi(0,disk_id) = 2*M_PI*uni_dist(gen);

        // for a given disk, loop over all times, drawing dphi increments at each time and summing it to previous values
        for (int t=1; t<nsamp; t++) {
            double dphi = sqrt(2*Dr*h) * norm_dist(gen);
            double phi_new = phi(t-1,disk_id)+dphi;
            phi(t,disk_id) = phi_new;
        }
    }
    // Return the matrix of phi values at all times
    return phi;

}

// Given the phi values from phi_active(), calculate the active force on each disk at time t
MatrixXd get_active_force_at_time_t(MatrixXd phi, int t, double f, int N) {
    // Compute the active force matrix at time t from the matrix of phi values at all times
    MatrixXd forces(N,2);
    for (int disk_id=0; disk_id<N; disk_id++) {
        forces(disk_id,0) = f*std::cos(phi(t,disk_id));
        forces(disk_id,1) = f*std::sin(phi(t,disk_id));
    }
    return forces;
}

// Define SDE parameters of the SDE: dX = a(t,X)dt+b(t,X)dW
// Deterministic part a(t,X)
MatrixXd a(MatrixXd X, double epsilon, double int_strength, double sigma, double D0) {
    return D0*get_WCA_force(X, epsilon, int_strength, sigma); // Nx2 matrix
}

// Stochastic part b(t,X)
MatrixXd b(MatrixXd X, double D0) {
    const int N = X.rows();
    return MatrixXd::Constant(N, 2, sqrt(2*D0)); // for Brownian motion with diffusion coefficient D (mu*k_B*T=D), b=sqrt(2D)
}

// Code to determine Sk in Runge-Kutta method definition (+-1 with prob 1/2 for Ito SDE, 0 with probability 1 for Stratonovich SDE)
MatrixXd Generate_Sk(int N, std::string IntegrationType = "I") {
    // Check integration type
    if (!(IntegrationType == "I" || IntegrationType == "S")) {
        throw std::invalid_argument("IntegrationType argument must be 'S' or 'I'");
    }
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> uni_dist(0, 1); // uniform distribution between 0 and 1

    MatrixXd Sk(N,2);
    for (int disk_id=0; disk_id<N; disk_id++) {
        for (int coord=0; coord<2; coord++) {
            Sk(disk_id,coord) = 0; // Stratonovich integration sets Sk=0 always
            if (IntegrationType == "I") {
            // Generate a random number
            double rand_num = uni_dist(gen);
            
            // Assign 1 or -1 based on the probability
            Sk(disk_id,coord) = (rand_num < 0.5) ? 1 : -1;
            }
        }
    }
    return Sk;  
}

// Uses the strong order 0.5 Euler method from SDEs to solve dynamics of the system of equations
//                          dX = a(t,X)dt+b(t,X)dW
// Inputs: Nx5 MatrixXd current_state, integration_type (Ito vs Stratonovich), Tmax, nsamp, epsilon, 
// int_strength of WCA potential, integration type "S" for Stratonovich and "I" for Ito 
// Outputs: Nx3 vector of disk trajectories (t, x and y components of the trajectory form the columns)
std::vector<std::vector<std::tuple<double, double, double>>> solve_Euler(MatrixXd X, double Tmax, int nsamp, double epsilon, double int_strength, double D0=0.5, double f=0, std::string IntegrationType = "I") {    
    const int N = X.rows();
    const int num_dimensions = 2; // consider changing this later for multidimensional sims - would need to change in lots of other places
    const double sigma = 2*X(0,0); // sigma=disk diameter. 0th element of each row in X is the disk radius - same for all disks.
    const double h = Tmax/nsamp; // h is the same thing as dt=Tmax/nsamp
    const double Dr = 3*D0/pow(sigma,2); // Rotational diffusion coefficient related to translational from Stokes-Einstein
    // Code to generate random number from standard normal distribution for stochastic part
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> norm_dist(0.0, 1.0); // defines standard normal distribution

    // Initialize trajectories matrix
    MatrixXd x_trajectories(N, nsamp);
    MatrixXd y_trajectories(N, nsamp);
    MatrixXd current_positions(N,2);

    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Compute phi values at all times for active force
    MatrixXd phi_values = phi_active(X,f,Tmax,nsamp,Dr);

    // Integrate the SDE
    double t = 0;

    // Vector to store tuples for each set of trajectories
    std::vector<std::vector<std::tuple<double, double, double>>> trajectories(N);
    for (size_t i=0; i<nsamp; ++i) {
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            trajectories[disk_i].push_back(std::make_tuple(t, current_positions(disk_i,0), current_positions(disk_i,1)));
        }

        MatrixXd dW(N,2);
        for (int disk_id=0; disk_id<N; disk_id++) {
            for (int coord=0; coord<2; coord++) {
                dW(disk_id, coord) = sqrt(h) * norm_dist(gen); // Wiener process increment
            }
        }

        // Get the active force on each disk
        MatrixXd active_force = get_active_force_at_time_t(phi_values,i,f,N);

        // Use Euler's method to update the positions
        MatrixXd update = h*(a(current_positions, epsilon, int_strength, sigma, D0)+active_force) + (dW.array()*b(current_positions, D0).array()).matrix(); // the array conversion used to implement Hadamard product - noise terms should multiply to their specific dimension
        current_positions += update;
        t += h;
    }

    return trajectories;
}

// Uses the strong order 1 Runge-Kutta method from SDEs to solve dynamics of the system of equations
//                          dX = a(t,X)dt+b(t,X)dW
// Inputs: Nx5 MatrixXd current_state, integration_type (Ito vs Stratonovich), Tmax, nsamp, epsilon, 
// int_strength of WCA potential, integration type "S" for Stratonovich and "I" for Ito 
// Outputs: Nx3 vector of disk trajectories (t, x and y components of the trajectory form the columns)
std::vector<std::vector<std::tuple<double, double, double>>> solve_RungeKutta(MatrixXd X, double Tmax, int nsamp, double epsilon, double int_strength, double D0=0.5, double f=0, std::string IntegrationType = "I") {    
    const int N = X.rows();
    const int num_dimensions = 2; // consider changing this later for multidimensional sims - would need to change in lots of other places
    const double sigma = 2*X(0,0); // sigma=disk diameter. 0th element of each row in X is the disk radius - same for all disks.
    const double h = Tmax/nsamp; // h is the same thing as dt=Tmax/nsamp
    const double Dr = 3*D0/pow(sigma,2); // Rotational diffusion coefficient related to translational from Stokes-Einstein
    // Code to generate random number from standard normal distribution for stochastic part
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> norm_dist(0.0, 1.0); // defines standard normal distribution

    // Initialize trajectories matrix
    MatrixXd x_trajectories(N, nsamp);
    MatrixXd y_trajectories(N, nsamp);
    MatrixXd current_positions(N,2);

    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Compute phi values at all times for active force
    MatrixXd phi_values = phi_active(X,f,Tmax,nsamp,Dr);

    // Integrate the SDE
    double t = 0;

    // Vector to store tuples for each set of trajectories
    std::vector<std::vector<std::tuple<double, double, double>>> trajectories(N);
    for (size_t i=0; i<nsamp; ++i) {
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            trajectories[disk_i].push_back(std::make_tuple(t, current_positions(disk_i,0), current_positions(disk_i,1)));
        }

        MatrixXd dW(N,2);
        for (int disk_id=0; disk_id<N; disk_id++) {
            for (int coord=0; coord<2; coord++) {
                dW(disk_id, coord) = sqrt(h) * norm_dist(gen); // Wiener process increment
            }
        }
        //double dW = sqrt(h) * norm_dist(gen); // Wiener process increment
        MatrixXd Sk = Generate_Sk(N, IntegrationType);

        // Get the active force on each disk
        MatrixXd active_force = get_active_force_at_time_t(phi_values,i,f,N);

        // Compute Runge-Kutta coefficients
        MatrixXd K1 = h*(a(current_positions, epsilon, int_strength, sigma, D0)+active_force) + ((dW-Sk*sqrt(h)).array()*b(current_positions, D0).array()).matrix(); // the array conversion used to implement Hadamard product - noise terms should multiply to their specific dimension
        MatrixXd K2 = h*(a(current_positions+K1, epsilon, int_strength, sigma, D0)+active_force) + ((dW+Sk*sqrt(h)).array()*b(current_positions+K1, D0).array()).matrix();

        // Update x using the Runge-Kutta method
        current_positions += (K1+K2)/2;
        t += h;
    }

    return trajectories;
}

int main() {
    // Open the CSV file
    std::string dpath="C:/Users/jpopows/OneDrive - UvA/Desktop/Active Glasses/Weeks Cage Breaking Model/"; // path to initial configuration data file here
    std::string init_config="initial_state.csv";
    std::ifstream file(dpath+init_config);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    // Read data from CSV file into a vector of vectors
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
        data.push_back(row);
    }

    // Close the file
    file.close();

    // Convert vector of vectors to Eigen matrix
    MatrixXd initial_state(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            initial_state(i, j) = data[i][j];
        }
    }

    // Print initial state
    std::cout << "Initial state matrix loaded from CSV file:" << std::endl;
    std::cout << initial_state << std::endl;

    // Test integration scheme with normal Brownian motion, save data
    std::vector<std::vector<std::tuple<double, double, double>>> brownian = solve_RungeKutta(initial_state, 1000, pow(10,6), 1, 40, 0.5, 0, "I"); // (initial state, Tmax, nsamp, epsilon, int_strength,D0,active force magnitude f,integration type)  

    // std::cout << "Number of disks: " << brownian.size() << std::endl;
    // std::cout << "Tuple size of each disk's vector:" << std::endl;
    // for (const auto& set : brownian) {
    //     std::cout << set.size() << std::endl;
    // }

    // Write data to CSV file
    std::ofstream outfile("data.csv");
    if (outfile.is_open()) {
        for (size_t i = 0; i < brownian[0].size(); ++i) {
            for (int disk_i = 0; disk_i < brownian.size(); ++disk_i) { // Loop over each disk
                const auto& tuple = brownian[disk_i][i];
                outfile << std::get<0>(tuple) << "," // Write t value
                        << std::get<1>(tuple) << "," // Write x
                        << std::get<2>(tuple); // Write y
                    
                if (disk_i < brownian.size()-1) {
                    outfile << ","; // Add comma between disks
                }
            }
            outfile << std::endl; // End of line
        }
        outfile.close();
        std::cout << "Data saved to data.csv" << std::endl;
    } else {
        std::cerr << "Failed to open file for writing." << std::endl;
        return 1;
    }

    return 0;
}
