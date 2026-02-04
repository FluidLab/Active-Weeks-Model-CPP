#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <Eigen/Dense>
#include <omp.h>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/////////////////////// Peak structure for state assignment ///////////////////////////////
struct Peak {
    double h;
    double L;
    bool use_L;  // If false, only use h for assignment
};

/////////////////////// Structure to hold transition event data ///////////////////////////////
struct TransitionEvent {
    long long timestep;
    int from_state;
    int to_state;
};

struct TransitionData {
    int n_states;
    double threshold;
    std::vector<TransitionEvent> events;  // List of all transition events
    std::vector<long long> residence_timesteps;  // residence_timesteps[i] = TOTAL time spent in state i
    
    TransitionData(int n, double thresh) : n_states(n), threshold(thresh) {
        residence_timesteps.resize(n, 0);
    }
};

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

/* Function to generate an initial configuration of disks. Make sure to put in the effective radius for r to avoid overlaps */
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
    
    // Calculate distance vector
    Vector2d v1_to_v2(x2 - x1, y2 - y1);
    
    // Return distance (L)
    return v1_to_v2.norm();
}

/////////////////////// Function to assign state based on proximity to peaks ///////////////////////////////
int assign_state(double h_val, double L_val, const std::vector<Peak>& peaks, double threshold) {
    // Find closest peak
    int closest_peak = -1;
    double min_distance = std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < peaks.size(); ++i) {
        double distance;
        if (peaks[i].use_L) {
            // 2D distance in (h, L) space
            double dh = h_val - peaks[i].h;
            double dL = L_val - peaks[i].L;
            distance = std::sqrt(dh*dh + dL*dL);
        } else {
            // 1D distance in h space only
            distance = std::abs(h_val - peaks[i].h);
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            closest_peak = i;
        }
    }
    
    // Check if within threshold
    if (min_distance <= threshold) {
        return closest_peak;
    }
    
    return -1; // Not assigned to any state
}

////////////////////// Functions for solving the Langevin dynamics of the disks ///////////////////////////////////
MatrixXd get_WCA_force(MatrixXd current_state, double epsilon, double int_strength, double sigma) {
    const int N = current_state.rows();
    const double r_eff = sigma*std::pow(2,1.0/6.0)/2;
    const double R=(N*r_eff)+epsilon; // R should be the number of disks plus epsilon, if each disk's effective radius (current_state(0,0)*2^(1/6)) is 1
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
        wall_dist(i) = r_eff + R - r(i);
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
            double wall_xdist = (r_eff+R)*x(disk_i)/r(disk_i);
            double wall_ydist = (r_eff+R)*y(disk_i)/r(disk_i);
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

/* Function to initialize AOUP active forces with correct statistics */
MatrixXd initialize_AOUP_forces(int N, double f, double tau_p, ThreadsafeRNG& rng) {
    MatrixXd forces(N, 2);
    double D_f = std::pow(f, 2) * 2 * tau_p / M_PI; // Diffusion coefficient for AOUP propulsion noise chosen to give same stationary force magnitude as ABP
    double stationary_std = std::sqrt(D_f / tau_p);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 2; ++j) {
            forces(i, j) = stationary_std * rng.normal();
        }
    }
    return forces;
}

/////////////////////// Function to write metadata to file ///////////////////////////////
void write_metadata(const std::vector<Peak>& peaks, const std::vector<double>& thresholds,
                   const std::string& filename, int N, double epsilon, double f, 
                   double tau_p, const std::string& model_type, double T, 
                   double int_strength, long long nsamp, double Tmax, double dt) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open metadata file: " + filename);
    }
    
    out << std::fixed << std::setprecision(10);
    
    out << "# Transition analysis metadata\n\n";
    out << "## Simulation parameters\n";
    out << "N = " << N << "\n";
    out << "epsilon = " << epsilon << "\n";
    out << "f = " << f << "\n";
    out << "tau_p = " << tau_p << "\n";
    out << "model_type = " << model_type << "\n";
    out << "T = " << T << "\n";
    out << "interaction_strength = " << int_strength << "\n";
    out << "timestep_dt = " << dt << "\n";
    out << "n_timesteps = " << nsamp << "\n";
    out << "Tmax = " << Tmax << "\n\n";
    
    out << "## Peak definitions\n";
    out << "n_peaks = " << peaks.size() << "\n";
    for (size_t i = 0; i < peaks.size(); ++i) {
        out << "peak_" << i << "_h = " << peaks[i].h << "\n";
        if (peaks[i].use_L) {
            out << "peak_" << i << "_L = " << peaks[i].L << "\n";
            out << "peak_" << i << "_type = 2D\n";
        } else {
            out << "peak_" << i << "_type = 1D\n";
        }
    }
    out << "\n";
    
    out << "## Thresholds\n";
    out << "n_thresholds = " << thresholds.size() << "\n";
    for (size_t i = 0; i < thresholds.size(); ++i) {
        out << "threshold_" << i << " = " << thresholds[i] << "\n";
    }
    
    out.close();
}

/////////////////////// Function to write transition events for one threshold ///////////////////////////////
void write_transition_events(const TransitionData& data, const std::string& filename, double dt) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open events file: " + filename);
    }
    
    out << std::fixed << std::setprecision(10);
    
    // Header
    out << "# Transition events for threshold = " << data.threshold << "\n";
    out << "# dt = " << dt << "\n";
    out << "# n_states = " << data.n_states << "\n\n";
    
    // Write total residence times in each of the states
    out << "# Residence times (timesteps):\n";
    for (int i = 0; i < data.n_states; ++i) {
        out << "# State " << i << ": " << data.residence_timesteps[i] << "\n";
    }
    out << "\n";
    
    // Write all events
    out << "timestep,time,from_state,to_state\n";
    for (const auto& event : data.events) {
        double time = event.timestep * dt;
        out << event.timestep << "," << time << "," 
            << event.from_state << "," << event.to_state << "\n";
    }
    
    out.close();
}

////////////////////// Main Runge-Kutta solver with transition tracking ///////////////////////////////////
void solve_RungeKutta_transitions(const MatrixXd& initial_state, double Tmax, long long nsamp,
                                 double epsilon, double int_strength, ThreadsafeRNG& rng,
                                 const std::string& output_base, double T, double f, double tau_p,
                                 std::string IntegrationType, const std::string& model_type, 
                                 double D_f, const std::vector<Peak>& peaks,
                                 const std::vector<double>& threshold_multipliers,
                                 double base_threshold) {
    const int N = initial_state.rows();
    const double sigma = 2*initial_state(0,0);
    const double h = Tmax/nsamp;
    
    MatrixXd current_positions(N,2);
    current_positions.col(0) = initial_state.col(1);
    current_positions.col(1) = initial_state.col(2);

    // Initialize active force variables based on model type
    VectorXd current_phi;
    MatrixXd stored_forces; // Only used for AOUP to store O-U force evolution
    MatrixXd active_forces; // the current matrix of active forces, regardless of the model
    std::vector<double> mean_force_magnitudes(N, 0.0); // Vector to track time-averaged active force magnitudes
    long long magnitude_count = 0;

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

    // Initialize transition tracking structures
    std::vector<TransitionData> all_transition_data;
    for (double multiplier : threshold_multipliers) {
        double threshold = base_threshold * multiplier;
        all_transition_data.emplace_back(peaks.size(), threshold);
    }

    // Track last committed state SEPARATELY for each threshold
    std::vector<int> last_committed_states(all_transition_data.size(), -1);

    // Main integration loop for the SDE
    double t = 0;
    for (long long i = 0; i < nsamp; ++i) {
        // Use current active forces for this timestep
        if (model_type == "ABP") {
            // For ABP, calculate force from current phi values
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

        // Calculate h and L values
        double h_value = calculate_h(current_positions);
        double L_value = calculate_L(current_positions);
        
        // Update all threshold tracking - each threshold tracks independently
        for (size_t t = 0; t < all_transition_data.size(); ++t) {
            int current_state = assign_state(h_value, L_value, peaks, 
                                            all_transition_data[t].threshold);
            
            // Increment residence time if in a state
            if (current_state >= 0) {
                all_transition_data[t].residence_timesteps[current_state]++;
            }
            
            // Record transition if we moved from one peak to another (excluding self-transitions)
            if (current_state >= 0 && last_committed_states[t] >= 0 && 
                current_state != last_committed_states[t]) {
                TransitionEvent event;
                event.timestep = i;
                event.from_state = last_committed_states[t];
                event.to_state = current_state;
                all_transition_data[t].events.push_back(event);
            }
            
            // Update last committed state for THIS threshold
            if (current_state >= 0) {
                last_committed_states[t] = current_state;
            }
        }

        // Progress update
        if (i % (nsamp/10) == 0) {
            std::cout << "Completed " << i << " iterations (" << (i * 100 / nsamp) << "% of total)" << std::endl;
        }
    }

    // Write metadata file
    std::string metadata_file = output_base + "_metadata.txt";
    write_metadata(peaks, threshold_multipliers, metadata_file, N, epsilon, f, tau_p, model_type,
                   T, int_strength, nsamp, Tmax, h);
    std::cout << "Wrote metadata to " << metadata_file << std::endl;
    
    // Write event files for each threshold
    for (size_t t = 0; t < all_transition_data.size(); ++t) {
        std::stringstream event_file;
        event_file << output_base << "_thresh" << std::fixed << std::setprecision(2) 
                   << all_transition_data[t].threshold << "_events.csv";
        write_transition_events(all_transition_data[t], event_file.str(), h);
        std::cout << "Wrote " << all_transition_data[t].events.size() 
                  << " events for threshold " << all_transition_data[t].threshold 
                  << " to " << event_file.str() << std::endl;
    }
}

/////////////////////////////// Main function ///////////////////////////////
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <peak_positions_file>\n";
        std::cerr << "  peak_positions_file: file containing peak positions (h [L] per line)\n";
        return 1;
    }
    
    // Read peak positions from file
    std::string peak_file_path = argv[1];
    std::vector<Peak> peaks;
    std::ifstream peak_file(peak_file_path);
    if (!peak_file.is_open()) {
        std::cerr << "Error: Could not open peak positions file: " << peak_file_path << "\n";
        return 1;
    }
    
    std::string line;
    while (std::getline(peak_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Peak p;
        p.L = 0.0;
        
        if (iss >> p.h) {
            if (iss >> p.L) {
                p.use_L = true;
            } else {
                p.use_L = false;
            }
            peaks.push_back(p);
        }
    }
    peak_file.close();
    
    if (peaks.empty()) {
        std::cerr << "Error: No peaks found in file\n";
        return 1;
    }
    
    std::cout << "Loaded " << peaks.size() << " peaks:\n";
    for (size_t i = 0; i < peaks.size(); ++i) {
        std::cout << "  Peak " << i << ": h = " << peaks[i].h;
        if (peaks[i].use_L) {
            std::cout << ", L = " << peaks[i].L << " (2D)";
        } else {
            std::cout << " (1D h-only)";
        }
        std::cout << "\n";
    }
    
    // Simulation parameters (all hard-coded)
    const int N = 3;
    const int r_eff = 1; // Effective radius from WCA potential (kicks in at separation of 2^(1/6)), actual disk radius is r_eff/2^(1/6)
    const double epsilon = 0.25;
    const double v = 0.0;
    const double Tmax = (int)3e6;
    const long long nsamp = 30000000000LL; // 30 billion timesteps
    const double int_strength = 40;
    const double T = 0.0; // Thermal or athermal (0.0)
    const double f = 1;
    const std::string model_type = "ABP"; // "AOUP" or "ABP"
    const double tau_p = 0.005; // 0.005, std::pow(2, -1.0/6.0), 10;
    const double D_f = std::pow(f, 2) * 2 * tau_p / M_PI; // Diffusion coefficient for AOUP
    
    // Transition tracking parameters
    const std::vector<double> threshold_multipliers = {1.0, 1.5, 2.0, 2.5, 3.0, 4.0};
    const double base_threshold = 0.5; // 0.069 = ~half of min separation between basins (at h=1.59 and 1.73, L=2)
    
    std::cout << "\n=== Simulation Parameters ===" << std::endl;
    std::cout << "Model type: " << model_type << std::endl;
    std::cout << "N = " << N << ", r_eff = " << r_eff << ", epsilon = " << epsilon << std::endl;
    std::cout << "f = " << f << ", tau_p = " << tau_p << std::endl;
    std::cout << "Timesteps: " << nsamp << ", Tmax = " << Tmax << std::endl;
    std::cout << "Base threshold: " << base_threshold << std::endl;
    std::cout << "Threshold multipliers: ";
    for (auto m : threshold_multipliers) std::cout << m << " ";
    std::cout << "\n" << std::endl;

    try {
        ThreadsafeRNG rng;
        MatrixXd initial_state = generate_initial_state(N, r_eff, epsilon, v, rng);
        
        // Create output file base name
        std::string dpath="Z:/Jared/Data/weeks_model_desktop/ABP transition rate matrices 21-11-2025/";
        std::stringstream output_base;
        output_base << dpath << "T=" << T << "_epsilon=" << epsilon << "_f=" << f 
                   << "_taup=" << tau_p << "_model=" << model_type;

        solve_RungeKutta_transitions(initial_state, Tmax, nsamp, epsilon, int_strength, 
                                    rng, output_base.str(), T, f, tau_p, "I",
                                    model_type, D_f, peaks, threshold_multipliers, 
                                    base_threshold);

        std::cout << "\nSimulation completed successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
