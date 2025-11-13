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

/* Function to generate an initial configuration of disks. Make sure to put in the effective radius for r_eff to avoid overlaps */
MatrixXd generate_initial_state(int N, int r_eff, double epsilon, double v, ThreadsafeRNG& rng) {
    double R = N*r_eff + epsilon; // Size of enclosure
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

////////////////////// Function to calculate L (distance between two non-tracked disks) at a given time ////////////////////////////////////////////////////
double calculate_L(const MatrixXd& positions, int hopping_disk_id = 2) {
    // Get indices of non-hopping disks
    std::vector<int> non_hop_indices;
    for (int i = 0; i < positions.rows(); i++) {
        if (i != hopping_disk_id) {
            non_hop_indices.push_back(i);
        }
    }
    
    // Get coordinates
    double x1 = positions(non_hop_indices[0], 0);
    double y1 = positions(non_hop_indices[0], 1);
    double x2 = positions(non_hop_indices[1], 0);
    double y2 = positions(non_hop_indices[1], 1);

    // Calculate vectors
    Vector2d v1_to_v2(x2 - x1, y2 - y1);
    
    // Calculate h
    return v1_to_v2.norm();
}

////////////////////// Function to calculate cage length lc at a given time ////////////////////////////////////////////////////
double calculate_lc(const MatrixXd& positions, double epsilon, double sigma, int hopping_disk_id = 2) {
    const int N = positions.rows();
    const double r_eff = sigma*std::pow(2,1.0/6.0)/2;
    const double R=(N*r_eff)+epsilon; // R should be the number of disks plus epsilon, if each disk's effective radius (sigma*2^(1/6)/2) is 1

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
// tracking cage-breaking events and h-value histogram on-the-fly.
// Inputs: Nx5 MatrixXd current_state, integration_type (Ito vs Stratonovich), Tmax, nsamp, epsilon, 
// int_strength of WCA potential, integration type "S" for Stratonovich and "I" for Ito 
// Outputs: Files containing time intervals of h zero-crossings and h histogram data
void solve_RungeKutta_event_driven(MatrixXd X, double Tmax, long long nsamp, double epsilon, 
                           double int_strength, ThreadsafeRNG& rng, const std::string& output_base,
                           double T=0.0, double f=0, double tau_p=1, std::string IntegrationType="I", 
                           std::string model_type="AOUP", double D_f=0.5, double thresh_param=1.0) {    
    const int N = X.rows();
    const double sigma = 2*X(0,0);
    const double h = Tmax/nsamp;

    // Calculate theoretical h_max and add small buffer for soft disks
    const double h_max_theory = 2 + epsilon + sqrt(3 + 4*epsilon + pow(epsilon,2));
    const double buffer = 0.07; // Add 7% buffer for soft interactions
    const double h_max = h_max_theory * (1 + buffer);
    const double h_min = -h_max;

    // Define histogram parameters - full range from -h_max to h_max
    const int n_bins = 100;
    const double bin_width = (h_max - h_min) / n_bins;
   
    // Initialize histogram bins and counts
    std::vector<double> bin_edges(n_bins + 1);
    std::vector<long long> bin_counts(n_bins, 0);
    for(int i = 0; i <= n_bins; i++) {
        bin_edges[i] = h_min + i * bin_width;
    }
    
    // Helper function to find bin index
    auto get_bin_index = [&](double value) -> int {
        if(value < h_min || value >= h_max) {
            throw std::runtime_error("ERROR: h value " + std::to_string(value) +
                                " outside bounds [" + std::to_string(h_min) +
                                ", " + std::to_string(h_max) + "]");
        }
        return static_cast<int>((value - h_min) / bin_width);
    };

    MatrixXd current_positions(N,2);
    current_positions.col(0) = X.col(1);
    current_positions.col(1) = X.col(2);

    // Initialize active force variables based on model type
    VectorXd current_phi;
    MatrixXd stored_forces; // Only used for AOUP to store O-U force evolution
    MatrixXd active_forces; // the current matrix of active forces, regardless of the model
    std::vector<double> mean_force_magnitudes(N, 0.0); // Running mean of active force magnitudes over time
    int magnitude_count = 0;

    if (model_type == "ABP") {
        // Initialize phi values for ABP
        current_phi = VectorXd(N);
        for (int disk_i = 0; disk_i < N; disk_i++) {
            current_phi(disk_i) = 2 * M_PI * rng.uniform();
        }
    } else if (model_type == "AOUP") {
        // Initialize with zero phi for compatibility (not used for AOUP)
        current_phi = VectorXd::Zero(N);
        
        // Initialize stored forces with random values drawn from the steady-state distribution
        stored_forces = MatrixXd::Zero(N,2);
        for (int disk_i = 0; disk_i < N; disk_i++) {
            double steady_state_std = std::sqrt(D_f / tau_p); // From theory of Ornstein-Uhlenbeck processes
            stored_forces(disk_i, 0) = steady_state_std * rng.normal();
            stored_forces(disk_i, 1) = steady_state_std * rng.normal();
        }
    }

    // State machine variables for crossing detection
    enum CageState { UNKNOWN, POSITIVE_STABLE, NEGATIVE_STABLE };
    CageState current_state = UNKNOWN;
    bool initial_stable_found = false;
    bool in_potential_transition = false;
    size_t potential_crossing_index = 0;
    double last_h_value = 0.0;
    
    // Storage for validated crossing indices
    std::vector<size_t> crossing_indices;

    // Calculate initial h value and update histogram
    double h_value = calculate_h(current_positions);
    try {
        int bin_idx = get_bin_index(h_value);
        bin_counts[bin_idx]++;
    } catch (const std::runtime_error& e) {
        #pragma omp critical
        {
            std::cerr << "Warning at initial state: " << e.what() << std::endl;
        }
    }
    
    // Check if initial state is in a stable region
    if (h_value > thresh_param) {
        current_state = POSITIVE_STABLE;
        initial_stable_found = true;
    } else if (h_value < -thresh_param) {
        current_state = NEGATIVE_STABLE;
        initial_stable_found = true;
    }
    
    last_h_value = h_value;

    // Integrate the SDE
    double t = 0;
    for (size_t i=0; i<nsamp; ++i) {
        // Use current active forces for this timestep
        if (model_type == "ABP") {
            active_forces = get_active_force(current_phi, f, N);
        } else if (model_type == "AOUP") {
            active_forces = stored_forces;
        }

        // Draw Wiener process increments for position dynamics (same for both models)
        MatrixXd dW(N,2);
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            for (int coord=0; coord<2; coord++) {
                dW(disk_i, coord) = sqrt(h) * rng.normal();
            }
        }

        // Generate Sk values
        MatrixXd Sk = Generate_Sk(N, rng, IntegrationType);

        // Compute Runge-Kutta coefficients
        MatrixXd K1 = h*(a(current_positions, epsilon, int_strength, sigma)+active_forces) + 
                      ((dW-Sk*sqrt(h)).array()*b(current_positions, T).array()).matrix();
        MatrixXd K2 = h*(a(current_positions+K1, epsilon, int_strength, sigma)+active_forces) + 
                      ((dW+Sk*sqrt(h)).array()*b(current_positions+K1, T).array()).matrix();

        // Update positions
        current_positions += (K1+K2)/2;
        t += h;

        // Update phi/forces for each disk for the NEXT timestep
        for (int disk_i = 0; disk_i<N; ++disk_i) {
            if (model_type == "ABP") {
                // Update phi for ABP
                double dphi = sqrt(h/tau_p) * rng.normal();
                current_phi(disk_i) += dphi;
            } else if (model_type == "AOUP") {
                // Update stored active forces for AOUP according to O-U process
                for (int coord = 0; coord < 2; coord++) {
                    double drift = -stored_forces(disk_i, coord) / tau_p * h;
                    double diffusion = sqrt(2.0 * D_f * h) * rng.normal() / tau_p;
                    stored_forces(disk_i, coord) += drift + diffusion;
                }
            }
        }

        // Calculate h
        h_value = calculate_h(current_positions);

        // Update histogram
        try {
            int bin_idx = get_bin_index(h_value);
            bin_counts[bin_idx]++;
        } catch (const std::runtime_error& e) {
            #pragma omp critical
            {
                std::cerr << "Warning at timestep " << i << ": " << e.what() << std::endl;
            }
            // Continue simulation despite out-of-bounds h value
        }

        // STATE MACHINE LOGIC FOR CROSSING DETECTION
        
        // First determine initial stable state if not yet found
        if (!initial_stable_found) {
            if (h_value > thresh_param) {
                current_state = POSITIVE_STABLE;
                initial_stable_found = true;
            } else if (h_value < -thresh_param) {
                current_state = NEGATIVE_STABLE;
                initial_stable_found = true;
            }
            // If still not found, continue to next iteration
            if (!initial_stable_found) {
                last_h_value = h_value;
                continue;
            }
        }

        // Detect NEW zero crossings (only if not already tracking one)
        if (!in_potential_transition && h_value * last_h_value <= 0 && current_state != UNKNOWN) {
            // Record that we just crossed zero
            potential_crossing_index = i;
            in_potential_transition = true;
        }

        // If we're tracking a potential transition, check EVERY timestep if it resolves
        if (in_potential_transition) {
            if (current_state == POSITIVE_STABLE && h_value < -thresh_param) {
                // Successfully reached opposite stable state
                crossing_indices.push_back(potential_crossing_index);
                current_state = NEGATIVE_STABLE;
                in_potential_transition = false;
            } else if (current_state == NEGATIVE_STABLE && h_value > thresh_param) {
                // Successfully reached opposite stable state
                crossing_indices.push_back(potential_crossing_index);
                current_state = POSITIVE_STABLE;
                in_potential_transition = false;
            } else if ((current_state == POSITIVE_STABLE && h_value > thresh_param) ||
                       (current_state == NEGATIVE_STABLE && h_value < -thresh_param)) {
                // Returned to original stable state - failed crossing
                in_potential_transition = false;
                // Don't record this crossing
            }
            // Otherwise: still in transition, keep checking next timestep
        }

        // Store h_value for next iteration's zero-crossing detection
        last_h_value = h_value;

        // Calculate and update running mean of force magnitudes using Welford's online algorithm
        magnitude_count++;
        for (int disk_i = 0; disk_i < N; disk_i++) {
            double fx = active_forces(disk_i, 0);
            double fy = active_forces(disk_i, 1);
            double magnitude = std::sqrt(fx*fx + fy*fy);

            // Incremental mean update: mean_new = mean_old + (current sample-mean_old)/n
            double delta = magnitude - mean_force_magnitudes[disk_i];
            mean_force_magnitudes[disk_i] += delta / magnitude_count;
        }

        // Progress update
        if (i % (nsamp/10) == 0) {
            std::cout << "Completed " << i << " iterations (" << (i * 100 / nsamp) << "% of total)" << std::endl;
        }
    }

    // Calculate statistics
    std::cout << "\n=== Simulation Results ===" << std::endl;
    std::cout << "Total timesteps: " << nsamp << std::endl;
    std::cout << "Validated crossings: " << crossing_indices.size() << std::endl;
    
    if (crossing_indices.size() > 1) {
        std::vector<size_t> intervals;
        for (size_t i = 1; i < crossing_indices.size(); ++i) {
            intervals.push_back(crossing_indices[i] - crossing_indices[i-1]);
        }
        
        double mean_interval = std::accumulate(intervals.begin(), intervals.end(), 0.0) / intervals.size();
        double sum_sq = 0.0;
        for (auto interval : intervals) {
            sum_sq += (interval - mean_interval) * (interval - mean_interval);
        }
        double std_interval = std::sqrt(sum_sq / intervals.size());
        double std_error = std_interval / std::sqrt(intervals.size());
        
        std::cout << "Number of intervals: " << intervals.size() << std::endl;
        std::cout << "Mean interval: " << mean_interval << " ± " << std_error << " timesteps" << std::endl;
        std::cout << "Mean cage time: " << mean_interval * h << " ± " << std_error * h << " time units" << std::endl;
    } else {
        std::cout << "Insufficient crossings for interval statistics" << std::endl;
    }

    // Calculate entropic barrier
    long long n_max = *std::max_element(bin_counts.begin(), bin_counts.end());
    int max_bin_idx = std::distance(bin_counts.begin(), 
                                    std::max_element(bin_counts.begin(), bin_counts.end()));
    int zero_bin_idx = get_bin_index(0.0);
    long long n_0 = bin_counts[zero_bin_idx];
    double entropic_barrier = (n_0 > 0) ? -std::log(static_cast<double>(n_0) / n_max) : INFINITY;

    std::cout << "\nEntropic barrier: " << entropic_barrier 
              << "\n  n_0=" << n_0 << " at bin " << zero_bin_idx 
              << " [" << bin_edges[zero_bin_idx] << ", " << bin_edges[zero_bin_idx+1] << ")"
              << "\n  n_max=" << n_max << " at bin " << max_bin_idx 
              << " [" << bin_edges[max_bin_idx] << ", " << bin_edges[max_bin_idx+1] << ")" 
              << std::endl;

    // Print time-averaged force magnitudes
    printf("\nMean active force magnitudes over simulation (should match f value):\n");
    for (int disk_i = 0; disk_i < N; disk_i++) {
        printf("  Disk %d: %.6f\n", disk_i, mean_force_magnitudes[disk_i]);
    }
    printf("\n");

    // Write crossing intervals to file
    std::string intervals_file = output_base + "_intervals.bin";
    std::ofstream interval_out(intervals_file, std::ios::binary);
    if (!interval_out.is_open()) {
        throw std::runtime_error("Could not open intervals output file: " + intervals_file);
    }

    // Write number of intervals
    size_t n_intervals = (crossing_indices.size() > 1) ? crossing_indices.size() - 1 : 0;
    interval_out.write(reinterpret_cast<const char*>(&n_intervals), sizeof(size_t));

    // Write intervals as size_t
    for (size_t i = 1; i < crossing_indices.size(); ++i) {
        size_t interval = crossing_indices[i] - crossing_indices[i-1];
        interval_out.write(reinterpret_cast<const char*>(&interval), sizeof(size_t));
    }
    interval_out.close();
    std::cout << "Wrote " << n_intervals << " intervals to " << intervals_file << std::endl;

    // Write histogram to file
    std::string histogram_file = output_base + "_histogram.bin";
    std::ofstream hist_out(histogram_file, std::ios::binary);
    if (!hist_out.is_open()) {
        throw std::runtime_error("Could not open histogram output file: " + histogram_file);
    }

    // Write metadata
    hist_out.write(reinterpret_cast<const char*>(&n_bins), sizeof(int));
    hist_out.write(reinterpret_cast<const char*>(&h_min), sizeof(double));
    hist_out.write(reinterpret_cast<const char*>(&h_max), sizeof(double));
    hist_out.write(reinterpret_cast<const char*>(&entropic_barrier), sizeof(double));

    // Write bin counts
    for (int i = 0; i < n_bins; ++i) {
        hist_out.write(reinterpret_cast<const char*>(&bin_counts[i]), sizeof(long long));
    }
    hist_out.close();
    std::cout << "Wrote histogram to " << histogram_file << std::endl;
}

/////////////////////////////// Main function ///////////////////////////////
int main() {
    // Simulation parameters
    const int N = 3;
    const int r_eff = 1; /* effective radius from WCA potential (kicks in at separation of 2^(1/6)), actual disk radius is r_eff/2^(1/6) */
    //const double epsilon = 0.021;
    const std::vector<double> eps_vals = {1.1,1.6,2.3,3.4,4.8,7,10,14.8}; //{0.01,0.014,0.021,0.03,0.043,0.062,0.089}; //{0.13,0.18,0.26,0.38,0.55,0.78}; //{1.1,1.6,2.3,3.4,4.8,7,10,14.8}; // uncomment for different values to run in parallel
    const double v = 0.0;
    const double Tmax = (int)3e6;
    const long long nsamp = 30000000000LL; // Using LL suffix for literals
    const double int_strength = 40;
    //const std::vector<double> int_strength_vals = {40, 100, 200, 400};
    const double T = 0.0; // thermal (typically 0.5) or athermal 0.0
    const double f = 1;
    const std::string model_type = "ABP"; // "AOUP" or "ABP"
    const double tau_p = 0.005; // persistence time
    const double thresh_param = 1.0; // threshold for cage-breaking detection

    bool had_error = false;

    // Parallel execution of simulations - schedule(dynamic) automatically distributes across available cores
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < eps_vals.size(); i++) {
        if (!had_error) {
            ThreadsafeRNG rng;
            double epsilon = eps_vals[i];
            const double D_f = std::pow(f,2) * 2 * tau_p / M_PI; // Diffusion coefficient for AOUP propulsion noise chosen to give same stationary force magnitude as ABP
            
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num() 
                          << " starting calculation for epsilon = " << epsilon 
                          << ", int_strength = " << int_strength 
                          << ", f = " << f 
                          << ", taup = " << tau_p 
                          << ", model " << model_type << std::endl;
            }

            try {
                MatrixXd initial_state = generate_initial_state(N, r_eff, epsilon, v, rng);
                
                std::string dpath="Z:/Jared/Data/weeks_model_desktop/ABP particle softness testing 09-10-2025/";
                std::stringstream output_base;
                output_base << dpath << "longer_event_driven_epsilon=" << epsilon 
                           << "_int_strength=" << int_strength 
                           << "_f=" << f 
                           << "_taup=" << tau_p 
                           << "_thresh=" << thresh_param
                           << "_model=" << model_type;

                solve_RungeKutta_event_driven(initial_state, Tmax, nsamp, epsilon, int_strength, 
                                             rng, output_base.str(), T, f, tau_p, "I",
                                             model_type, D_f, thresh_param);

                #pragma omp critical
                {
                    std::cout << "Thread " << omp_get_thread_num() 
                              << " finished calculation for epsilon = " << epsilon 
                              << ", int_strength = " << int_strength 
                              << ", f = " << f 
                              << ", taup = " << tau_p 
                              << ", model " << model_type << std::endl;
                }
            }
            catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Error in thread " << omp_get_thread_num() 
                             << " for epsilon = " << epsilon 
                             << ", int_strength = " << int_strength 
                             << ", f = " << f 
                             << ", taup = " << tau_p 
                             << ", model " << model_type 
                             << ": " << e.what() << std::endl;
                    had_error = true;
                }
            }
        }
    }

    return had_error ? 1 : 0;
}