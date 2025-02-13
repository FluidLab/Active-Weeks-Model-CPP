#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <omp.h>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/////////////////////// Thread-safe random number generator class //////////////////////////////////
class ThreadsafeRNG {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<> uni_dist;
    std::normal_distribution<> norm_dist;

public:
    ThreadsafeRNG() : 
        gen(std::random_device{}()),
        uni_dist(0.0, 1.0),
        norm_dist(0.0, 1.0) {}

    // Copy constructor that ensures different threads get different seeds
    ThreadsafeRNG(const ThreadsafeRNG&) : 
        gen(std::random_device{}()),
        uni_dist(0.0, 1.0),
        norm_dist(0.0, 1.0) {}

    double uniform() {
        return uni_dist(gen);
    }

    double normal() {
        return norm_dist(gen);
    }
};

/////////////////////// Functions for defining the initial state ///////////////////////////////////
bool check_disk_overlap(const Vector2d& pos1, const Vector2d& pos2, double r1, double r2) {
    double dist = (pos1 - pos2).norm();
    return dist <= (r1 + r2);
}

bool check_wall_collision(const Vector2d& pos, double r, double R) {
    return pos.norm() >= (R - r);
}

MatrixXd generate_initial_state(int N, double r, double epsilon, double v, ThreadsafeRNG& rng) {
    double R = N*r + epsilon; // Size of enclosure
    MatrixXd state(N, 5); // Format: [radius, x, y, vx, vy]
    int max_attempts = 1000;
    int attempt = 0;
    
    while (attempt < max_attempts) {
        bool valid_state = true;
        
        // Generate positions for all disks
        for (int i = 0; i < N; ++i) {
            double r_pos = rng.uniform() * R;
            double theta = 2 * M_PI *rng.uniform();
            double v_direction = 2 * M_PI *rng.uniform();
            
            // Set disk properties
            state(i, 0) = r; // radius
            state(i, 1) = r_pos * cos(theta); // x
            state(i, 2) = r_pos * sin(theta); // y
            state(i, 3) = v * cos(v_direction); // vx
            state(i, 4) = v * sin(v_direction); // vy
            
            // Check overlap with previous disks
            for (int j = 0; j < i; ++j) {
                Vector2d pos_i(state(i, 1), state(i, 2));
                Vector2d pos_j(state(j, 1), state(j, 2));
                if (check_disk_overlap(pos_i, pos_j, state(i, 0), state(j, 0))) {
                    valid_state = false;
                    break;
                }
            }
            
            // Check wall collision
            if (valid_state) {
                Vector2d pos_i(state(i, 1), state(i, 2));
                if (check_wall_collision(pos_i, state(i, 0), R)) {
                    valid_state = false;
                }
            }
            
            if (!valid_state) break;
        }
        
        if (valid_state) {
            return state;
        }
        
        attempt++;
    }
    
    throw std::runtime_error("Failed to generate valid initial state after maximum attempts");
}

////////////////////// Functions for solving the Langevin dynamics of the disks ///////////////////////////////////
MatrixXd get_WCA_force(MatrixXd current_state, double epsilon, double int_strength, double sigma) {
    const int N = current_state.rows();
    const double R=(N*sigma*std::pow(2,1.0/6.0)/2)+epsilon; // R should be the number of disks plus epsilon, if each disk's effective radius (current_state(0,0)*2^(1/6)) is 1
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
        wall_dist(i) = sigma*std::pow(2,1.0/6.0)/2+R-r(i); // replace sigma by sigma*2^(1/6)?
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
            double wall_xdist = (sigma*std::pow(2,1.0/6.0)/2+R)*x(disk_i)/r(disk_i);
            double wall_ydist = (sigma*std::pow(2,1.0/6.0)/2+R)*y(disk_i)/r(disk_i);
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

// Given the phi values from phi_active(), calculate the active force on each disk at time t
MatrixXd get_active_force(VectorXd phi, double f, int N) {
    // Compute the active force matrix at time t from the matrix of phi values at all times
    MatrixXd forces(N,2);
    for (int disk_id=0; disk_id<N; disk_id++) {
        forces(disk_id,0) = f*std::cos(phi(disk_id));
        forces(disk_id,1) = f*std::sin(phi(disk_id));
    }
    return forces;
}

// Define parameters of the SDE: dX = a(t,X)dt+b(t,X)dW
// Deterministic part a(t,X)
MatrixXd a(MatrixXd X, double epsilon, double int_strength, double sigma) {
    return get_WCA_force(X, epsilon, int_strength, sigma); // Nx2 matrix
}

// Stochastic part b(t,X)
MatrixXd b(MatrixXd X, double T) {
    const int N = X.rows();
    return MatrixXd::Constant(N, 2, sqrt(2*T)); // for Brownian motion with D/(k_BT)=1 and defining k_BT = T, b=sqrt(2T)
}

// Code to determine Sk in Runge-Kutta method definition (+-1 with prob 1/2 for Ito SDE, 0 with probability 1 for Stratonovich SDE)
MatrixXd Generate_Sk(int N, ThreadsafeRNG& rng, std::string IntegrationType = "I") {
    // Check integration type
    if (!(IntegrationType == "I" || IntegrationType == "S")) {
        throw std::invalid_argument("IntegrationType argument must be 'S' or 'I'");
    }

    MatrixXd Sk(N,2);
    for (int disk_id=0; disk_id<N; disk_id++) {
        for (int coord=0; coord<2; coord++) {
            Sk(disk_id,coord) = 0; // Stratonovich integration sets Sk=0 always
            if (IntegrationType == "I") {
            // Generate random number (uniform dist.), assign 1 or -1 based on the probability
            Sk(disk_id,coord) = (rng.uniform() < 0.5) ? 1 : -1;
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
std::vector<std::vector<std::tuple<double, double, double>>> solve_Euler(MatrixXd X, double Tmax, int nsamp, double epsilon, double int_strength, ThreadsafeRNG& rng, double T=0.5, double f=0, std::string IntegrationType = "I") {    
    const int N = X.rows();
    const int num_dimensions = 2; // consider changing this later for multidimensional sims - would need to change in lots of other places
    const double sigma = 2*X(0,0); // sigma=disk diameter. 0th element of each row in X is the disk radius - same for all disks.
    const double h = Tmax/nsamp; // h is the same thing as dt=Tmax/nsamp
    const double Dr = 3*T/pow(sigma,2); // Rotational diffusion coefficient related to translational from Stokes-Einstein

    // Initialize trajectories matrix
    MatrixXd current_positions(N,2);

    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Initialize the directions for all disks with random values from 0 to 2pi radians
    VectorXd current_phi(N);
    for (int disk_i=0; disk_i<N; disk_i++) {
        current_phi(disk_i) = 2*M_PI*rng.uniform();
    }

    std::vector<std::vector<std::tuple<double, double, double>>> trajectories(N); // Vector to store tuples for each set of trajectories
    
    // Integrate the SDE
    double t = 0;    
    for (size_t i=0; i<nsamp; ++i) {
        // Get the active force on each disk
        MatrixXd active_force = get_active_force(current_phi,f,N);

        // Disk updates/draw noise term
        MatrixXd dW(N,2);
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            // Add current positions to trajectories
            trajectories[disk_i].push_back(std::make_tuple(t, current_positions(disk_i,0), current_positions(disk_i,1)));

            // Update phi value by drawing dphi increment from gaussian dist and summing it to previous phi value
            double dphi = sqrt(2*Dr*h) * rng.normal();
            double phi_new = current_phi(disk_i)+dphi;
            current_phi(disk_i) = phi_new;

            // Draw Wiener process increments for each disk's coordinate in the timestep
            for (int coord=0; coord<num_dimensions; coord++) {
                dW(disk_i, coord) = sqrt(h) * rng.normal(); // Wiener process increment
            }
        }

        // Use Euler's method to update the positions
        MatrixXd update = h*(a(current_positions, epsilon, int_strength, sigma)+active_force) + (dW.array()*b(current_positions, T).array()).matrix(); // the array conversion used to implement Hadamard product - noise terms should multiply to their specific dimension
        current_positions += update;
        t += h;

        // Progress update every nsamp/10 iterations
        if (i % (nsamp/10) == 0) {
            std::cout << "Completed " << i << " iterations (" << (i * 100 / nsamp) << "% of total)" << std::endl;
        }
    }

    return trajectories;
}

// Uses the strong order 1 Runge-Kutta method from SDEs to solve dynamics of the system of equations
//                          dX = a(t,X)dt+b(t,X)dW
// Inputs: Nx5 MatrixXd current_state, integration_type (Ito vs Stratonovich), Tmax, nsamp, epsilon, 
// int_strength of WCA potential, integration type "S" for Stratonovich and "I" for Ito 
// Outputs: Nx3 vector of disk trajectories (t, x and y components of the trajectory form the columns)
std::vector<std::vector<std::tuple<double, double, double>>> solve_RungeKutta(MatrixXd X, double Tmax, int nsamp, double epsilon, double int_strength, ThreadsafeRNG& rng, double T=0.5, double f=0, std::string IntegrationType = "I") {    
    const int N = X.rows();
    const int num_dimensions = 2; // consider changing this later for multidimensional sims - would need to change in lots of other places
    const double sigma = 2*X(0,0); // sigma=disk diameter. 0th element of each row in X is the disk radius - same for all disks.
    const double h = Tmax/nsamp; // h is the same thing as dt=Tmax/nsamp
    const double Dr = 3*T/pow(sigma,2); // Rotational diffusion coefficient related to translational from Stokes-Einstein

    // Initialize trajectories matrix
    MatrixXd current_positions(N,2);
    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Initialize the directions for all disks with random values from 0 to 2pi radians
    VectorXd current_phi(N);
    for (int disk_i=0; disk_i<N; disk_i++) {
        current_phi(disk_i) = 2*M_PI*rng.uniform();
    }

    std::vector<std::vector<std::tuple<double, double, double>>> trajectories(N); // Vector to store tuples for each set of trajectories
    
    // Integrate the SDE
    double t = 0;
    for (size_t i=0; i<nsamp; ++i) {
        // Get the active force on each disk
        MatrixXd active_force = get_active_force(current_phi,f,N);

        // Generate Sk values for each disk's coordinate
        MatrixXd Sk = Generate_Sk(N, rng, IntegrationType);

        // Disk updates/draw noise term
        MatrixXd dW(N,2);
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            // Add current positions to trajectories
            trajectories[disk_i].push_back(std::make_tuple(t, current_positions(disk_i,0), current_positions(disk_i,1)));

            // Update phi value by drawing dphi increment from gaussian dist and summing it to previous phi value
            double dphi = sqrt(2*Dr*h) * rng.normal();
            double phi_new = current_phi(disk_i)+dphi;
            current_phi(disk_i) = phi_new;

            // Draw Wiener process increments for each disk's coordinate in the timestep
            for (int coord=0; coord<num_dimensions; coord++) {
                dW(disk_i, coord) = sqrt(h) * rng.normal(); // Wiener process increment
            }
        }

        // Compute Runge-Kutta coefficients
        MatrixXd K1 = h*(a(current_positions, epsilon, int_strength, sigma)+active_force) + ((dW-Sk*sqrt(h)).array()*b(current_positions, T).array()).matrix(); // the array conversion used to implement Hadamard product - noise terms should multiply to their specific dimension
        MatrixXd K2 = h*(a(current_positions+K1, epsilon, int_strength, sigma)+active_force) + ((dW+Sk*sqrt(h)).array()*b(current_positions+K1, T).array()).matrix();

        // Update x using the Runge-Kutta method
        current_positions += (K1+K2)/2;
        t += h;

        // Progress update every nsamp/10 iterations
        if (i % (nsamp/10) == 0) {
            std::cout << "Completed " << i << " iterations (" << (i * 100 / nsamp) << "% of total)" << std::endl;
        }
    }

    return trajectories;
}

int main() {
    // Simulation parameters
    const int N = 3;
    const double r = 1.0/pow(2, 1.0/6.0); // this makes the effective radius of the disks 1, since the WCA potential kicks in at a separation of 2^1/6
    const double epsilon = 0.06;
    const double v = 0.0;
    const double Tmax = 100000;
    const int nsamp = (int)1e8;
    const double int_strength = 40;
    const double T = 0.5;
    const std::vector<double> f_vals = {1.7,2,3,4,5}; // Different f values 0,0.5,1,1.2,1.5,  1.7,2,3,4,5

    bool had_error = false;  // Flag to track errors

    // Parallel execution of simulations - schedule(dynamic) automatically distributes across available cores
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < f_vals.size(); i++) {
        if (!had_error) {  // Only continue if no errors have occurred
            ThreadsafeRNG rng;
            double f = f_vals[i];
            
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num() 
                          << " starting calculation for f = " << f << std::endl;
            }

            MatrixXd initial_state = generate_initial_state(N, r, epsilon, v, rng);
            std::vector<std::vector<std::tuple<double, double, double>>> brownian = 
                solve_RungeKutta(initial_state, Tmax, nsamp, epsilon, int_strength, rng, T, f, "I");

            // Write data to CSV file
            std::string dpath="C:/Users/jpopows/OneDrive - UvA/Desktop/Active Glasses/Weeks Cage Breaking Model/data/varying_f_epsilon=0.06/";
            std::stringstream filename;
            filename << N << "_particles_epsilon=" << epsilon << "_f=" << f << "_data.csv";
            std::string dynamic_filename = filename.str();

            #pragma omp critical 
            {
                std::ofstream outfile(dpath+dynamic_filename);
                if (outfile.is_open()) {
                    for (size_t j = 0; j < brownian[0].size(); ++j) {
                        for (int disk_i = 0; disk_i < brownian.size(); ++disk_i) {
                            const auto& tuple = brownian[disk_i][j];
                            outfile << std::get<0>(tuple) << ","
                                    << std::get<1>(tuple) << ","
                                    << std::get<2>(tuple);
                                
                            if (disk_i < brownian.size()-1) {
                                outfile << ",";
                            }
                        }
                        outfile << std::endl;
                    }
                    outfile.close();
                    std::cout << "Thread " << omp_get_thread_num() 
                              << " finished calculation and saved data for f = " << f << std::endl;
                } else {
                    std::cerr << "Thread " << omp_get_thread_num() 
                              << " failed to open file for f = " << f << std::endl;
                    had_error = true;  // Set error flag instead of returning
                }
            }
        }
    }

    return had_error ? 1 : 0;  // Return error code after parallel region
}