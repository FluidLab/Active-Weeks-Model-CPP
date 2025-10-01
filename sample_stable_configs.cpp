#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <omp.h>
#include <iomanip>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::string formatted_string(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

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

    /* Set seed explicitly */
    void set_seed(unsigned int seed) {
        gen.seed(seed);
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

/* Structure to hold stable configuration data */
struct StableConfiguration {
    double epsilon;
    double h_value;
    MatrixXd positions;
    double t_stabilized; // Time when stabilization occurred
    std::vector<double> initial_phi_values; // Initial active force angles
};

/* Function to check if h has stabilized */
bool is_h_stable(const std::vector<double>& recent_h_values, double stability_threshold) {
    if (recent_h_values.size() < 2) {
        return false;
    }
    
    double mean_h = 0.0;
    for (double h : recent_h_values) {
        mean_h += h;
    }
    mean_h /= recent_h_values.size();
    
    /* find maximum deviation of h from its mean value */
    double max_deviation = 0.0;
    for (double h : recent_h_values) {
        double deviation = std::abs(h - mean_h);
        if (deviation > max_deviation) {
            max_deviation = deviation;
        }
    }
    
    return max_deviation < stability_threshold;
}

/* Function to save stable configuration to file */
void save_stable_configuration(const StableConfiguration& config, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }
    
    /* Write epsilon */
    outfile.write(reinterpret_cast<const char*>(&config.epsilon), sizeof(double));
    
    /* Write h value */
    outfile.write(reinterpret_cast<const char*>(&config.h_value), sizeof(double));
    
    /* Write time when stabilized */
    outfile.write(reinterpret_cast<const char*>(&config.t_stabilized), sizeof(double));

    /* Write initial phi values */
    int vec_size = config.initial_phi_values.size();
    outfile.write(reinterpret_cast<const char*>(&vec_size), sizeof(int));
    
    for (int i = 0; i < vec_size; i++) {
        double value = config.initial_phi_values[i];
        outfile.write(reinterpret_cast<const char*>(&value), sizeof(double));
    }
    
    /* Write number of rows and columns in positions matrix */
    int rows = config.positions.rows();
    int cols = config.positions.cols();
    outfile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    
    /* Write positions matrix data */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value = config.positions(i, j);
            outfile.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }
    
    outfile.close();
}

/*
Explores phase space by:
1. Generating multiple random spatial configurations
2. For each spatial configuration, testing multiple random phi combinations
3. Only saving configurations that stabilize

Inputs: N (number of disks), epsilon, r_eff, Tmax, dt, int_strength, T, f, output_dir,
        num_spatial_configs (number of random spatial arrangements to test),
        num_phi_samples (number of random phi combinations per spatial config),
        stabilization_time, stability_window, stability_threshold
Outputs: Saves stable configurations to files
*/ 
void explore_phase_space(int N, double epsilon, int r_eff, double Tmax, double dt, double int_strength, double T, 
    double f, const std::string& output_dir, int num_spatial_configs, int num_phi_samples,
    double stabilization_time = 10.0, int stability_window = 1000, double stability_threshold = 0.0005) {
    
    /* Storage for h-value histogram */
    std::map<double, int> h_histogram;
    int bin_width_precision = 2; // Round h to this many decimal places for binning

    /* Storage for all stable h values (for analysis) */
    std::vector<double> all_stable_h_values;
    
    /* Storage for stable configurations */
    std::vector<StableConfiguration> stable_configs;
    
    /* Total number of configurations to test */
    long long total_configs = static_cast<long long>(num_spatial_configs) * num_phi_samples;
    int completed = 0;
    
    printf("Testing %d spatial configurations x %d phi samples = %lld total configurations for epsilon = %.2f\n", 
           num_spatial_configs, num_phi_samples, total_configs, epsilon);

    /* Calculate the number of steps to wait before stability checking */
    int stabilization_steps = static_cast<int>(stabilization_time / dt);
    printf("Will begin stability checking after %d timesteps (%.2f time units)\n", 
           stabilization_steps, stabilization_time);
    
    /* Loop over spatial configurations */
    for (int spatial_id = 0; spatial_id < num_spatial_configs; spatial_id++) {
        printf("Starting spatial configuration %d/%d for epsilon=%.2f\n", 
               spatial_id + 1, num_spatial_configs, epsilon);
        
        /* Generate a new random initial spatial configuration */
        ThreadsafeRNG spatial_rng;
        spatial_rng.set_seed(spatial_id * 1000000); // Ensure different spatial configs
        
        MatrixXd initial_state;
        try {
            initial_state = generate_initial_state(N, r_eff, epsilon, 0.0, spatial_rng);
            //printf("Successfully generated initial state for spatial config %d\n", spatial_id);
        } catch (const std::exception& e) {
            printf("ERROR: Failed to generate spatial configuration %d for epsilon=%.2f: %s\n", 
                   spatial_id, epsilon, e.what());
            continue; // Skip this spatial configuration and try the next one
        }
        
        //printf("About to start parallel phi sampling for spatial config %d\n", spatial_id);
        
        /* Process each phi configuration for this spatial arrangement */
        #pragma omp parallel for schedule(dynamic)
        for (int phi_sample_id = 0; phi_sample_id < num_phi_samples; phi_sample_id++) {
            /* Create a thread-local RNG with deterministic seed based on spatial and phi IDs */
            ThreadsafeRNG local_rng;
            local_rng.set_seed(spatial_id * 1000000 + phi_sample_id);
            
            /* Generate random phi values for each disk */
            VectorXd current_phi(N);
            for (int disk_id = 0; disk_id < N; disk_id++) {
                current_phi(disk_id) = 2.0 * M_PI * local_rng.uniform();
            }
            
            /* Create a copy of the initial state for this run */
            MatrixXd X = initial_state;
            
            /* Run simulation */
            MatrixXd current_positions(N, 2);
            current_positions.col(0) = X.col(1);
            current_positions.col(1) = X.col(2);
            
            const double sigma = 2*X(0,0);
            const int nsamp = static_cast<int>(Tmax/dt);
            
            /* Calculate theoretical h_max */
            const double h_max_theory = 2 + epsilon + sqrt(3 + 4*epsilon + pow(epsilon,2));
            const double buffer = 0.07; 
            const double h_max = h_max_theory * (1 + buffer);
            const double h_min = 0.0;
            
            /* Create vector for stability checking */
            std::vector<double> recent_h_values;
            recent_h_values.reserve(stability_window);
            
            /* Calculate initial h value */
            double h_value = calculate_h(current_positions);
            
            /* Integrate the SDE */
            double t = 0; 
            bool is_stable = false;
            double t_stabilized = 0;
            
            for (size_t i = 0; i < nsamp && !is_stable; ++i) {
                /* Get the active force on each disk */
                MatrixXd active_force = get_active_force(current_phi, f, N);
                
                /* Generate Sk values */
                MatrixXd Sk = Generate_Sk(N, local_rng, "I");
                
                /* Draw noise values */
                MatrixXd dW(N, 2);
                for (int disk_i = 0; disk_i < N; ++disk_i) {
                    for (int coord = 0; coord < 2; coord++) {
                        dW(disk_i, coord) = sqrt(dt) * local_rng.normal();
                    }
                }
                
                /* Compute Runge-Kutta coefficients */
                MatrixXd K1 = dt*(a(current_positions, epsilon, int_strength, sigma) + active_force) + 
                             ((dW-Sk*sqrt(dt)).array()*b(current_positions, T).array()).matrix();
                MatrixXd K2 = dt*(a(current_positions+K1, epsilon, int_strength, sigma) + active_force) + 
                             ((dW+Sk*sqrt(dt)).array()*b(current_positions+K1, T).array()).matrix();
                
                /* Update positions */
                current_positions += (K1+K2)/2;
                t += dt;
                
                /* Calculate h and check bounds */
                h_value = calculate_h(current_positions);
                validate_h_value(h_value, h_min, h_max);
                
                if (i >= stabilization_steps) {
                    /* Update recent h values for stability check */
                    if (recent_h_values.size() >= stability_window) {
                        recent_h_values.erase(recent_h_values.begin());
                    }
                    recent_h_values.push_back(h_value);
                    
                    /* Check if h has stabilized */
                    if (recent_h_values.size() >= stability_window) {
                        if (is_h_stable(recent_h_values, stability_threshold)) {
                            is_stable = true;
                            t_stabilized = t;
                        }
                    }
                }
            }
            
            /* Save result if stable */
            if (is_stable) {
                StableConfiguration config;
                config.epsilon = epsilon;
                config.h_value = h_value;
                config.positions = current_positions;
                config.t_stabilized = t_stabilized;
                
                /* Add the initial phi values */
                config.initial_phi_values.clear();
                for (int disk_id = 0; disk_id < N; disk_id++) {
                    config.initial_phi_values.push_back(current_phi(disk_id));
                }
                
                #pragma omp critical
                {
                    stable_configs.push_back(config);
                    all_stable_h_values.push_back(h_value);
                    
                    /* Update histogram */
                    double rounded_h = std::round(std::abs(h_value) * pow(10, bin_width_precision)) / pow(10, bin_width_precision);
                    h_histogram[rounded_h]++;
                    
                    /* Progress update */
                    // completed++;
                    // if (completed % (total_configs / 10) == 0 || completed == total_configs) {
                    //     printf("Completed %d/%lld configurations (%.1f%%) - Found %zu stable configs\n", 
                    //            completed, total_configs, 100.0 * completed / total_configs, stable_configs.size());
                    // }
                }
            }
        }
        
        //printf("Completed phi sampling for spatial config %d\n", spatial_id);
    }

    printf("Found %zu stable configurations out of %lld tested (%.1f%%)\n", 
        stable_configs.size(), total_configs, 100.0 * stable_configs.size() / total_configs);
 
    /* Save all h values to file (not histogram) */
    std::string h_values_file = output_dir + "/all_h_values_epsilon_" + formatted_string(epsilon,2) + ".txt";
    std::ofstream out_h(h_values_file);
    if (out_h.is_open()) {
        out_h << "# h_value (one per line)\n";
        out_h << std::setprecision(10);
        for (double h : all_stable_h_values) {
            out_h << h << "\n";
        }
        out_h.close();
        printf("All h values saved to %s\n", h_values_file.c_str());
    }
 
    /* Find the most common h values and save representative configurations */
    const int num_representatives = 5;
    std::vector<std::pair<double, int>> hist_entries(h_histogram.begin(), h_histogram.end());
    std::sort(hist_entries.begin(), hist_entries.end(), 
    [](const std::pair<double, int>& a, const std::pair<double, int>& b) { 
        return a.second > b.second; 
    });
 
    for (int i = 0; i < std::min(num_representatives, static_cast<int>(hist_entries.size())); i++) {
        double target_h = hist_entries[i].first;
     
        /* Find a configuration with this h value */
        for (const auto& config : stable_configs) {
            double rounded_h = std::round(std::abs(config.h_value) * pow(10, bin_width_precision)) / pow(10, bin_width_precision);
            if (std::abs(rounded_h - target_h) < 1e-5) {
                /* Save this configuration */
                std::string config_file = output_dir + "/config_h_" + 
                                        formatted_string(target_h,2) + "_epsilon_" + 
                                        formatted_string(epsilon,2) + ".bin";
             
                /* Save the configuration */
                save_stable_configuration(config, config_file);
             
                /* Also save a visual text version */
                std::string vis_file = output_dir + "/visual_config_h_" + 
                                     formatted_string(target_h,2) + "_epsilon_" + 
                                     formatted_string(epsilon,2) + ".txt";
             
                std::ofstream out_vis(vis_file);
                if (out_vis.is_open()) {
                    out_vis << "# Epsilon: " << epsilon << "\n";
                    out_vis << "# h_value: " << config.h_value << "\n";
                    out_vis << "# Time stabilized: " << config.t_stabilized << "\n";
                    out_vis << "# Initial phi values: ";
                    for (double phi : config.initial_phi_values) {
                        out_vis << phi << " ";
                    }
                    out_vis << "\n";
                    out_vis << "# Disk positions (x, y):\n";
                    for (int j = 0; j < config.positions.rows(); j++) {
                        out_vis << config.positions(j, 0) << " " << config.positions(j, 1) << "\n";
                    }
                    out_vis.close();
                }
             
                break;
            }
        }
    }
}

/* Main function */
int main() {
    /* Simulation parameters */
    const int N = 3; /* Code as currently written WON'T WORK if disk number is not 3, could be easily adapted though */
    const int r_eff = 1; /* effective radius from WCA potential (kicks in at separation of 2^(1/6)), actual disk radius is r_eff/2^(1/6) */
    const std::vector<double> eps_vals = {0.01, 0.1, 0.25, 0.6, 1, 10};
    const double Tmax = 100;
    const double dt = 1e-4; // simulation step size
    const double int_strength = 40;
    const double T = 0.0;
    const double f = 1;
    
    /* Phase space exploration parameters */
    const int num_spatial_configs = 100;  // Number of random spatial arrangements to test
    const int num_phi_samples = 1000;     // Number of random phi combinations per spatial config
    
    /* Stability checking parameters */
    const double stabilization_time = 10.0; // Run for 10 time units before checking stability
    const int stability_window = 1000;
    const double stability_threshold = 0.0005;
    
    /* Output directory */
    const std::string output_dir = "stable_configurations";
    
    /* Process each epsilon value */
    bool had_error = false;
    
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < eps_vals.size(); i++) {
        if (!had_error) {
            double epsilon = eps_vals[i];
            
            #pragma omp critical
            {
                printf("Thread %d starting calculation for epsilon = %.2f\n", 
                       omp_get_thread_num(), epsilon);
            }
            
            try {
                /* Explore phase space for this epsilon */
                explore_phase_space(N, epsilon, r_eff, Tmax, dt, int_strength, T, f, output_dir, 
                    num_spatial_configs, num_phi_samples, stabilization_time, stability_window, stability_threshold);
                
                #pragma omp critical
                {
                    printf("Thread %d finished calculation for epsilon = %.2f\n", 
                           omp_get_thread_num(), epsilon);
                }
            }
            catch (const std::exception& e) {
                #pragma omp critical
                {
                    printf("Error in thread %d for epsilon = %.2f: %s\n", 
                           omp_get_thread_num(), epsilon, e.what());
                    had_error = true;
                }
            }
        }
    }
    
    return had_error ? 1 : 0;
}