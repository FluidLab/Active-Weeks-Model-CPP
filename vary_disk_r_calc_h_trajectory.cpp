#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
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

MatrixXd generate_initial_state(int N, double r_geom, double r_eff, double epsilon, double v, ThreadsafeRNG& rng) {
    double R = N*r_geom + epsilon; // Size of enclosure
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
            state(i, 0) = r_eff/std::pow(2, 1.0/6.0); // actual disk radius is WCA effective radius divided by 2^(1/6)
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

////////////////////// Helper function to check if absolute value of h falls outside the allowed range /////////////
void validate_h_value(double h_value, double h_min, double h_max) {
    double abs_value = std::abs(h_value);
    if(abs_value < h_min || abs_value >= h_max) {
        throw std::runtime_error("ERROR: |h| value " + std::to_string(abs_value) + 
                               " outside bounds [" + std::to_string(h_min) + 
                               ", " + std::to_string(h_max) + "]");
    }
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

////////////////////// Function to calculate cage length lc at a given time ////////////////////////////////////////////////////
double calculate_lc(const MatrixXd& positions, double epsilon, double r_geom, double sigma, int hopping_disk_id = 2) {
    const int N = positions.rows();
    const double r_eff = sigma*std::pow(2,1.0/6.0)/2;
    const double R=(N*r_geom)+epsilon; // R should be the number of disks plus epsilon, if each disk's effective radius (sigma*2^(1/6)/2) is 1

    VectorXd x = positions.col(0);
    VectorXd y = positions.col(1);
    VectorXd r = (x.array().square()+y.array().square()).array().sqrt(); // vector of the radial distances for each disk from the center

    // Store distances from hopping_disk_id to edges of other disks
    std::vector<double> edge_distances;
    
    // Calculate distance from hopping_disk_id to all other disk edges
    for (int i = 0; i < N; ++i) {
        if (i != hopping_disk_id) {
            // Get center-to-center distance
            double rij = std::sqrt(std::pow((x(hopping_disk_id)-x(i)),2) + 
                                 std::pow((y(hopping_disk_id)-y(i)),2));
            // Distance from center of hopping disk to edge of disk i's potential
            double edge_dist = rij - r_eff;
            edge_distances.push_back(edge_dist);
        }
    }

    // Distance from hopping disk center to wall edge (where interaction starts)
    double wall_dist = R - r(hopping_disk_id);
    edge_distances.push_back(wall_dist);

    // Calculate average distance from disk center to edges of its neighbors (other disks and wall)
    double rnn = 0;
    for (const double& dist : edge_distances) {
        rnn += dist;
    }
    rnn /= edge_distances.size();

    // Cage length: average center-to-edge distance minus effective radius (this is equivalent to traditional definition: NN center distance minus effective sigma)
    double lc = rnn - r_eff;
    return lc;
}

////////////////////// Functions for solving the Langevin dynamics of the disks ///////////////////////////////////
MatrixXd get_WCA_force(MatrixXd current_state, double epsilon, double int_strength, double r_geom, double sigma) {
    const int N = current_state.rows();
    const double R=(N*r_geom)+epsilon; // R should be the number of disks plus epsilon, if each disk's effective radius (current_state(0,0)*2^(1/6)) is 1
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
MatrixXd a(MatrixXd X, double epsilon, double int_strength, double r_geom, double sigma) {
    return get_WCA_force(X, epsilon, int_strength, r_geom, sigma); // Nx2 matrix
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
double solve_RungeKutta_chunks(MatrixXd X, double r_geom, double Tmax, int nsamp, double epsilon, 
                           double int_strength, ThreadsafeRNG& rng, const std::string& filename,
                           double T=0.5, double f=0, double tau_p=1, std::string IntegrationType="I", 
                           int chunk_size=1000000) {    
    const int N = X.rows();
    const double sigma = 2*X(0,0);
    const double r_eff = sigma*std::pow(2,1.0/6.0)/2;
    const double h = Tmax/nsamp;

    // Calculate theoretical h_max and add small buffer for soft disks
    const double h_max_theory = (3 * r_geom - r_eff + epsilon) + sqrt(pow((3 * r_geom - r_eff + epsilon), 2) - pow(r_eff, 2)); // adapted from Hunter, Weeks PRE 2012, allowing nonequal r_geom and r_eff
    const double buffer = 0.07; // Add 7% buffer for soft interactions
    const double h_max = h_max_theory * (1 + buffer);
    const double h_min = 0.0;  // Starting from 0 since we're taking abs(h)

    MatrixXd current_positions(N,2);
    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Initialize phi values
    VectorXd current_phi(N);
    for (int disk_i=0; disk_i<N; disk_i++) {
        current_phi(disk_i) = 2*M_PI*rng.uniform();
    }

    // Open output file in binary mode
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    // Vector to store h values for current chunk
    std::vector<double> h_values;
    h_values.reserve(chunk_size);

    // Calculate initial h value and add to chunk
    h_values.push_back(calculate_h(current_positions));

    // Initialize running average for lc using Welford's algorithm
    double lc_mean = 0.0;
    size_t lc_count = 0;
    
    // Calculate and accumulate initial lc
    double lc_initial = calculate_lc(current_positions, epsilon, r_geom, sigma);
    lc_count++;
    lc_mean = lc_initial;

    // Integrate the SDE
    double t = 0; 
    for (size_t i=0; i<nsamp; ++i) {
        // Get the active force on each disk
        MatrixXd active_force = get_active_force(current_phi,f,N);

        // Generate Sk values
        MatrixXd Sk = Generate_Sk(N, rng, IntegrationType);

        // Draw noise and update phi values
        MatrixXd dW(N,2);
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            // Update phi
            double dphi = sqrt(h/tau_p) * rng.normal();
            current_phi(disk_i) += dphi;

            // Draw Wiener process increments
            for (int coord=0; coord<2; coord++) {
                dW(disk_i, coord) = sqrt(h) * rng.normal();
            }
        }

        // Compute Runge-Kutta coefficients
        MatrixXd K1 = h*(a(current_positions, epsilon, int_strength, r_geom, sigma)+active_force) + 
                      ((dW-Sk*sqrt(h)).array()*b(current_positions, T).array()).matrix();
        MatrixXd K2 = h*(a(current_positions+K1, epsilon, int_strength, r_geom, sigma)+active_force) + 
                      ((dW+Sk*sqrt(h)).array()*b(current_positions+K1, T).array()).matrix();

        // Update positions
        current_positions += (K1+K2)/2;
        t += h;

        // Calculate h and add to current chunk
        double h_value = calculate_h(current_positions);
        validate_h_value(h_value, h_min, h_max);
        h_values.push_back(h_value);

        // Calculate lc and update running average using Welford's algorithm
        double lc_current = calculate_lc(current_positions, epsilon, r_geom, sigma);
        lc_count++;
        double delta = lc_current - lc_mean;
        lc_mean += delta / lc_count;

        // If chunk is full, write to file and clear
        if (h_values.size() >= chunk_size) {
            // Create a buffer for the data (reduces number of write operations)
            std::vector<char> buffer(h_values.size() * sizeof(float)); // Space for h values
            char* ptr = buffer.data();
            
            // Convert to float and copy to buffer
            for (size_t i=0; i<h_values.size(); i++) {
                float h_float = static_cast<float>(h_values[i]);
                std::memcpy(ptr, &h_float, sizeof(float));
                ptr += sizeof(float);
            }
            
            // Write the entire buffer at once
            outfile.write(buffer.data(), buffer.size());
            outfile.flush();

            // Clear h vector
            h_values.clear();

            // Reserve space for next chunk
            h_values.reserve(chunk_size);
        }

        // Progress update
        if (i % (nsamp/10) == 0) {
            std::cout << "Completed " << i << " iterations (" << (i * 100 / nsamp) << "% of total)" << std::endl;
        }
    }

    // Write any remaining values at the end
    for (size_t i=0; i<h_values.size(); i++) {
        float h_float = static_cast<float>(h_values[i]);
        outfile.write(reinterpret_cast<char*>(&h_float), sizeof(float));
    }
    outfile.close();

    // Write average cage length to CSV file
    std::string csv_filename = filename.substr(0, filename.find_last_of('.')) + "_lc_avg.csv";
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + csv_filename);
    }
    
    // Write header
    csv_file << "epsilon,f,tau_p,T,int_strength,Tmax,nsamp,lc_avg\n";
    
    // Write data with full precision
    csv_file << std::scientific << std::setprecision(16);
    csv_file << epsilon << "," << f << "," << tau_p << "," << T << "," 
             << int_strength << "," << Tmax << "," << nsamp << "," << lc_mean << "\n";
    
    csv_file.close();
    
    return lc_mean;
}

/////////////////////////////// Main function ///////////////////////////////
int main() {
    // Simulation parameters
    const int N = 3;
    const int r_geom = 1; /* length scale defining the confinement size, R=3*r_geom + epsilon */
    const double r_eff = 1; /* effective radius from WCA potential (kicks in at separation of 2^(1/6)), actual disk radius is r_eff/2^(1/6) */
    const double epsilon = 1.1; // uncomment for constant-epsilon sims
    //const std::vector<double> eps_vals = {0.13,0.18,0.26,0.38,0.55,0.78}; //{0.01,0.014,0.021,0.03,0.043,0.062,0.089}; //{0.13,0.18,0.26,0.38,0.55,0.78}; //{1.1,1.6,2.3,3.4,4.8,7,10,14.8}; // uncomment for different values to run in parallel
    const double v = 0.0;
    const double Tmax = (int)1e5; // Change depending on the epsilon value!
    const int nsamp = (int)1e9;
    const double int_strength = 40;
    const double T = 0.0;
    const double f = 1;
    //const double tau_p = 3; // persistence time. For Stokes-Einstein, tau_p = 1/(2*Dr)=pow(2*r*std::pow(2,1.0/6.0),2)/(6*T)
    //Eigen::VectorXd tau_p_vals = Eigen::VectorXd::LinSpaced(41, 0, 2); // For parallel calculations of dense tau_p values - for spacing of 0.05 between 0 and 2, you need (2 - 0)/0.05 + 1 = 41 points
    const std::vector<double> tau_p_vals = {0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 2, 3, 5}; // uncomment for different values to run in parallel
    const int chunk_size = (int)5e7; // Process 50 million steps at a time

    bool had_error = false;

    // Parallel execution of simulations - schedule(dynamic) automatically distributes across available cores
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < tau_p_vals.size(); i++) {
    //for(size_t i = 0; i < eps_vals.size(); i++) {
        if (!had_error) {
            ThreadsafeRNG rng;
            double tau_p = tau_p_vals[i];
            //double epsilon = eps_vals[i];
            
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num() 
                          << " starting calculation for epsilon = " << epsilon << ", f =" << f 
                          << ", taup = " << tau_p << std::endl;
            }

            try {
                MatrixXd initial_state = generate_initial_state(N, r_geom, r_eff, epsilon, v, rng);
                
                std::string dpath="Z:/Jared/Data/weeks_model_desktop/ABP cage time optimum sims 15-11-2025/";
                std::stringstream filename;
                filename << "r_eff=" << r_eff << "_same_confine_geom_h_trajectories_epsilon=" << epsilon << "_f=" << f << "_taup=" << tau_p << "_model=ABP" << ".bin";
                std::string output_file = dpath + filename.str();

                double lc_avg = solve_RungeKutta_chunks(initial_state, r_geom, Tmax, nsamp, epsilon, int_strength, 
                                      rng, output_file, T, f, tau_p, "I", chunk_size);

                #pragma omp critical
                {
                    std::cout << "Thread " << omp_get_thread_num() 
                              << " finished calculation for epsilon = " << epsilon << ", f =" << f << ", taup =" << tau_p 
                              << ", <lc> = " << lc_avg << std::endl;
                }
            }
            catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Error in thread " << omp_get_thread_num() 
                             << " for epsilon = " << epsilon << ", f =" << f << ", taup =" << tau_p
                              << ": " << e.what() << std::endl;
                    had_error = true;
                }
            }
        }
    }

    return had_error ? 1 : 0;
}