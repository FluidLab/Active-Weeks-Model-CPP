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

///////////////////// Function to calculate expansion parameter at a given time (ONLY WORKS FOR 3 PARTICLES, USE CONVEX HULL METHOD FOR MORE) /////////////////////////////////////
double calculate_A(const Eigen::MatrixXd& positions) {
    // Extract vertices as 3D vectors (z=0)
    Eigen::Vector3d v1(positions(0,0), positions(0,1), 0);
    Eigen::Vector3d v2(positions(1,0), positions(1,1), 0);
    Eigen::Vector3d v3(positions(2,0), positions(2,1), 0);
    
    // Calculate area using cross product
    return 0.5 * ((v2-v1).cross(v3-v1)).norm();
}

////////////////////// Function to calculate milling parameter at a given time //////////////////////////////////////
double calculate_M(const MatrixXd& positions, const VectorXd& current_angles) {
    const int N = current_angles.rows();
    Eigen::Vector2d com = positions.colwise().mean(); // (x,y) center of mass position
    
    // Loop over particles
    double milling = 0.0;
    for(int i = 0; i < N; i++) {
        // Calculate position of each disk with respect to the center of mass
        Eigen::Vector2d r_i = positions.row(i).transpose() - com;
        r_i.normalize(); // Normalize r_i
        
        // Create 3D vectors for cross product (z = 0)
        Eigen::Vector3d r_i_3d(r_i[0], r_i[1], 0.0);
        
        // Create active velocity vector from angle
        double phi = current_angles[i];
        Eigen::Vector3d v_i(std::cos(phi), std::sin(phi), 0.0);
        
        // Calculate cross product and take z component
        milling += r_i_3d.cross(v_i)[2];
    }
    
    // Take absolute value and normalize by N
    return std::abs(milling)/N;
}


////////////////////// Function to calculate polarization parameter at a given time /////////////////////////////////
double calculate_P(const VectorXd& current_angles) {
    const int N = current_angles.rows();
    Eigen::Vector2d sum_vector = Eigen::Vector2d::Zero();
    
    // Sum up all direction vectors
    for (int i = 0; i < N; ++i) {
        double phi = current_angles(i);
        Eigen::Vector2d direction;
        direction << std::cos(phi), std::sin(phi); // direction vector, given angle of active particle direction
        sum_vector += direction;
    }
    
    // polarization = magnitude of sum divided by N
    double polarization = sum_vector.norm()/N;
    
    return polarization;
}

/* Function to extract angles from force vectors for AOUP model */
VectorXd extract_phi_from_forces(const MatrixXd& active_forces) {
    const int N = active_forces.rows();
    VectorXd phi_extracted(N);
    
    for (int i = 0; i < N; ++i) {
        double fx = active_forces(i, 0);
        double fy = active_forces(i, 1);
        phi_extracted(i) = std::atan2(fy, fx);
    }
    
    return phi_extracted;
}

////////////////////// Function to calculate h at a given time ////////////////////////////////////////////////////
double calculate_h(const MatrixXd& positions, int hopping_disk_id = 2) {
    // Get indices of non-hopping disks
    std::vector<int> non_hop_indices;
    for (int i = 0; i < positions.rows(); i++) {
        if (i != hopping_disk_id) {
            non_hop_indices.push_back(i);
        }
    }
    
    // Get coordinates
    double x_hop = positions(hopping_disk_id, 0);
    double y_hop = positions(hopping_disk_id, 1);
    double x1 = positions(non_hop_indices[0], 0);
    double y1 = positions(non_hop_indices[0], 1);
    double x2 = positions(non_hop_indices[1], 0);
    double y2 = positions(non_hop_indices[1], 1);

    // Calculate vectors
    Vector2d v1_to_v_hop(x_hop - x1, y_hop - y1);
    Vector2d v1_to_v2(x2 - x1, y2 - y1);
    
    // Calculate cross product for sin(theta)
    double sin_of_angle = (v1_to_v2(0)*v1_to_v_hop(1) - v1_to_v2(1)*v1_to_v_hop(0)) / 
                         (v1_to_v_hop.norm() * v1_to_v2.norm());
    
    // Calculate h
    return v1_to_v_hop.norm() * sin_of_angle;
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

        // each disk's nearest distance to the extent of the potential felt by the wall is r_eff+R-r
        wall_dist(i) = sigma * std::pow(2,1.0/6.0) / 2 + R - r(i);
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

    // calculate x and y values of the force produced by the various WCA potentials
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

// Calculate the total potential energy (sum of WCA potentials) stored in the current state
double get_potential_energy(MatrixXd current_state, double epsilon, double int_strength, double sigma) {
    const int N = current_state.rows();
    const double R=(N*sigma*std::pow(2,1.0/6.0)/2)+epsilon; // R should be the number of disks plus epsilon, if each disk's effective radius (current_state(0,0)*2^(1/6)) is 1
    VectorXd x = current_state.col(0);
    VectorXd y = current_state.col(1);
    VectorXd r = (x.array().square()+y.array().square()).array().sqrt(); // vector of the radial distances for each disk from the center

    double total_energy = 0.0;

    // Calculate disk-disk interaction energies
    for (int disk_i = 0; disk_i < N; ++disk_i) {
        for (int disk_j = disk_i + 1; disk_j < N; ++disk_j) {
            // Calculate distance between disks
            double r_ij = std::sqrt(std::pow((x(disk_i) - x(disk_j)), 2) + 
                                   std::pow((y(disk_i) - y(disk_j)), 2));
                                   
            // Apply WCA potential if disks are within range
            if (r_ij <= sigma * std::pow(2, 1.0/6.0)) {
                double energy_ij = 4 * int_strength * (std::pow((sigma / r_ij), 12) - 
                                                        std::pow((sigma / r_ij), 6)) + 
                                  int_strength;
                total_energy += energy_ij;
            }
        }
        
        // Calculate disk-wall interaction energy
        // each disk's nearest distance to the enclosure potential is r_eff+R-r
        double dist_to_wall = sigma * std::pow(2, 1.0/6.0) / 2 + R - r(disk_i);
        if (dist_to_wall <= sigma * std::pow(2, 1.0/6.0)) {
            double energy_wall = 4 * int_strength * (std::pow((sigma / dist_to_wall), 12) - 
                                                     std::pow((sigma / dist_to_wall), 6)) + 
                               int_strength;
            total_energy += energy_wall;
        }
    }
    
    return total_energy;
}

// Given the phi values from phi_active(), calculate the active force on each disk at time t (for ABP model)
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

// Uses the strong order 1 Runge-Kutta method from SDEs to solve dynamics of the system of equations
//                          dX = a(t,X)dt+b(t,X)dW,
// only saving the h values of the 3-particle system rather than the whole trajectories.
// Inputs: Nx5 MatrixXd current_state, integration_type (Ito vs Stratonovich), Tmax, nsamp, epsilon, 
// int_strength of WCA potential, integration type "S" for Stratonovich and "I" for Ito 
// Outputs: Length nsamp vector of h values for the system
void solve_RungeKutta_chunks(MatrixXd X, double Tmax, int nsamp, double epsilon, 
                           double int_strength, ThreadsafeRNG& rng, const std::string& filename,
                           double T=0.0, double f=0, double tau_p=1, std::string IntegrationType="I", 
                           int chunk_size=1000000, std::string model_type="AOUP", double D_f=0.5) {    
    const int N = X.rows();
    const double sigma = 2*X(0,0);
    const double h = Tmax/nsamp;

    MatrixXd current_positions(N,2);
    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Initialize active force variables based on model type
    VectorXd current_phi;
    MatrixXd current_active_forces;

    if (model_type == "ABP") {
        // Initialize phi values for ABP
        current_phi = VectorXd(N);
        for (int disk_i = 0; disk_i < N; disk_i++) {
            current_phi(disk_i) = 2 * M_PI * rng.uniform();
        }
        // Initialize with zeros, will be calculated from phi
        current_active_forces = MatrixXd::Zero(N, 2);
    } else if (model_type == "AOUP") {
        // Initialize active forces for AOUP
        // We'll still initialize phi for compatibility, but won't use it
        current_phi = VectorXd::Zero(N);
        current_active_forces = MatrixXd::Zero(N, 2);
        
        // Initialize with random values drawn from the steady-state distribution
        for (int disk_i = 0; disk_i < N; disk_i++) {
            double steady_state_std = std::sqrt(D_f / tau_p); // From theory of Ornstein-Uhlenbeck processes
            current_active_forces(disk_i, 0) = steady_state_std * rng.normal();
            current_active_forces(disk_i, 1) = steady_state_std * rng.normal();
        }
    }

    // Open output file
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    // Write header
    outfile << "h,A,M,P";
    for (int i = 0; i < N; i++) {
        outfile << ",x" << i << ",y" << i << ",phi" << i;
    }
    outfile << "\n";

    // Vector to store calculated values for current chunk
    std::vector<double> h_values;
    std::vector<double> A_values;
    std::vector<double> M_values;
    std::vector<double> P_values;
    std::vector<MatrixXd> positions;  // Store positions for each timestep
    std::vector<VectorXd> phi_values;
    h_values.reserve(chunk_size);
    A_values.reserve(chunk_size);
    M_values.reserve(chunk_size);
    P_values.reserve(chunk_size);
    positions.reserve(chunk_size);
    phi_values.reserve(chunk_size);

    // Initial calculations of tracked values over sim
    VectorXd initial_phi;
    if (model_type == "ABP") {
        initial_phi = current_phi;
    } else if (model_type == "AOUP") {
        initial_phi = extract_phi_from_forces(current_active_forces);
    }

    h_values.push_back(calculate_h(current_positions));
    A_values.push_back(calculate_A(current_positions));
    M_values.push_back(calculate_M(current_positions, initial_phi));
    P_values.push_back(calculate_P(initial_phi));
    positions.push_back(current_positions);
    phi_values.push_back(initial_phi);

    // Integrate the SDE
    double t = 0;
    for (size_t i=0; i<nsamp; ++i) {
        // Get the active force on each disk - modify this call
        MatrixXd active_force;
        if (model_type == "ABP") {
            active_force = get_active_force(current_phi, f, N);
        } else if (model_type == "AOUP") {
            active_force = current_active_forces; // Use stored force values directly
        }
        
        // Generate Sk values
        MatrixXd Sk = Generate_Sk(N, rng, IntegrationType);
        
        // Draw noise and update phi/forces depending on model
        MatrixXd dW(N,2);
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            if (model_type == "ABP") {
                // Update phi for ABP
                double dphi = sqrt(h/tau_p) * rng.normal();
                current_phi(disk_i) += dphi;
            } else if (model_type == "AOUP") {
                // Update active forces for AOUP according to O-U process
                for (int coord = 0; coord < 2; coord++) {
                    double drift = -current_active_forces(disk_i, coord) / tau_p * h;
                    double diffusion = sqrt(2.0 * D_f * h) * rng.normal() / tau_p;
                    current_active_forces(disk_i, coord) += drift + diffusion;
                }
            }
            
            // Draw Wiener process increments for position dynamics (same for both models)
            for (int coord=0; coord<2; coord++) {
                dW(disk_i, coord) = sqrt(h) * rng.normal();
            }
        }

        // Compute Runge-Kutta coefficients
        MatrixXd K1 = h*(a(current_positions, epsilon, int_strength, sigma)+active_force) + 
                      ((dW-Sk*sqrt(h)).array()*b(current_positions, T).array()).matrix();
        MatrixXd K2 = h*(a(current_positions+K1, epsilon, int_strength, sigma)+active_force) + 
                      ((dW+Sk*sqrt(h)).array()*b(current_positions+K1, T).array()).matrix();

        // Update positions
        current_positions += (K1+K2)/2;
        t += h;

        // Calculate values and add to current chunk
        double h_value = calculate_h(current_positions);
        double A_value = calculate_A(current_positions);

        // Calculate M and P using appropriate phi values
        double M_value, P_value;
        if (model_type == "ABP") {
            M_value = calculate_M(current_positions, current_phi);
            P_value = calculate_P(current_phi);
        } else if (model_type == "AOUP") {
            VectorXd phi_from_forces = extract_phi_from_forces(current_active_forces);
            M_value = calculate_M(current_positions, phi_from_forces);
            P_value = calculate_P(phi_from_forces);
        }

        // When storing phi_values for output:
        VectorXd phi_for_output;
        if (model_type == "ABP") {
            phi_for_output = current_phi;
        } else if (model_type == "AOUP") {
            phi_for_output = extract_phi_from_forces(current_active_forces);
        }

        h_values.push_back(h_value);
        A_values.push_back(A_value);
        M_values.push_back(M_value);
        P_values.push_back(P_value);
        positions.push_back(current_positions);
        phi_values.push_back(phi_for_output);

        // When chunk is full, write to file and clear
        if (h_values.size() >= chunk_size) {
            for (size_t i = 0; i < h_values.size(); i++) {
                outfile << h_values[i] << "," << A_values[i] << "," << M_values[i] << "," << P_values[i];
                // Write positions, angles for each particle
                for (int p = 0; p < N; p++) {
                    outfile << "," << positions[i](p,0) << "," << positions[i](p,1) << "," << phi_values[i](p);
                }
                outfile << "\n";
            }
            outfile.flush();

            // Clear vectors
            h_values.clear();
            A_values.clear();
            M_values.clear();
            P_values.clear();
            positions.clear();
            phi_values.clear();

            // Reserve space for next chunk
            h_values.reserve(chunk_size);
            A_values.reserve(chunk_size);
            M_values.reserve(chunk_size);
            P_values.reserve(chunk_size);
            positions.reserve(chunk_size);
            phi_values.reserve(chunk_size);
        }

        // Progress update
        if (i % (nsamp/10) == 0) {
            std::cout << "Completed " << i << " iterations (" << (i * 100 / nsamp) << "% of total)" << std::endl;
        }
    }

    // Write any remaining values at the end
    for (size_t i = 0; i < h_values.size(); i++) {
        outfile << h_values[i] << "," << A_values[i] << "," << M_values[i] << "," << P_values[i];
        for (int p = 0; p < N; p++) {
            outfile << "," << positions[i](p,0) << "," << positions[i](p,1) << "," << phi_values[i](p);
        }
        outfile << "\n";
    }
    outfile.close();
}

/////////////////////////////// Main function ///////////////////////////////
int main() {
    // Simulation parameters
    const int N = 3;
    const double r = 1.0/pow(2, 1.0/6.0); // this makes the effective radius of the disks 1, since the WCA potential kicks in at a separation of 2^1/6
    const double epsilon = 1;
    const double v = 0.0;
    const double Tmax = (int)1e4;
    const int nsamp = (int)1e7;
    const double int_strength = 40;
    const double T = 0.0; // thermal or athermal 0.0
    const double f = 1;
    const std::string model_type = "AOUP"; // or "ABP"
    //Eigen::VectorXd active_f_vals = Eigen::VectorXd::LinSpaced(21, 0, 5); // For parallel calculations of dense active_f_values
    //const double tau_p = 1; // persistence time. For Stokes-Einstein, tau_p = 1/(2*Dr)=pow(2*r*std::pow(2,1.0/6.0),2)/(6*T)
    //Eigen::VectorXd tau_p_vals = Eigen::VectorXd::LinSpaced(41, 0, 2); // For parallel calculations of dense tau_p values - for spacing of 0.05 between 0 and 2, you need (2 - 0)/0.05 + 1 = 41 points
    const std::vector<double> tau_p_vals = {0.01, 0.1, 1, 5}; // Different tau_p values to run in parallel, {0.001, 0.0022, 0.0046, 0.01, 0.022, 0.046, 0.1, 0.22, 0.46, 1, 2, 3, 4, 5}
    const int chunk_size = (int)1e7; // Process 10 million steps at a time

    bool had_error = false;

    // Parallel execution of simulations - schedule(dynamic) automatically distributes across available cores
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < tau_p_vals.size(); i++) {
        if (!had_error) {
            ThreadsafeRNG rng;
            double tau_p = tau_p_vals[i];
            const double D_f = pow(f,2) * 2 * tau_p / M_PI; //pow(f,2)/(3*tau_p); // Diffusion coefficient for AOUP propulsion noise chosen to give same stationary force magnitude as ABP
            
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num() 
                          << " starting calculation for f = " << f << ", taup = " << tau_p << ", model " << model_type << std::endl;
            }

            try {
                MatrixXd initial_state = generate_initial_state(N, r, epsilon, v, rng);
                
                std::string dpath="Z:/Jared/Data/weeks_model_desktop/AOUP_trajectories_22_07_2025/";
                std::stringstream filename;
                filename << "T=" << T << "_epsilon=" << epsilon << "_f=" << f 
                        << "_taup=" << tau_p << "_model=" << model_type << "_same_stationary_force_norms_trajectories.csv";
                std::string output_file = dpath + filename.str();

                solve_RungeKutta_chunks(initial_state, Tmax, nsamp, epsilon, int_strength, 
                                      rng, output_file, T, f, tau_p, "I", chunk_size,
                                    model_type, D_f);

                #pragma omp critical
                {
                    std::cout << "Thread " << omp_get_thread_num() 
                              << " finished calculation for epsilon = " << epsilon << ", f = " << f 
                              << ", taup = " << tau_p << ", model " << model_type << std::endl;
                }
            }
            catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Error in thread " << omp_get_thread_num() 
                             << " for epsilon = " << epsilon << ", f = " << f 
                             << ", taup = " << tau_p << ", model " << model_type << ": " << e.what() << std::endl;
                    had_error = true;
                }
            }
        }
    }

    return had_error ? 1 : 0;
}