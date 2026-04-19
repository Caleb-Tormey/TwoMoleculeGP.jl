#=using Statistics
using Base.Threads
using LinearAlgebra
using StaticArrays
using Plots
using AbstractGPs
using KernelFunctions=#

#println("Starting N2 Two-Molecule Direct Sampling...")
#println("Running on $(Threads.nthreads()) threads.")
#=
# ==========================================
# 1. Physical Parameters (N2 System)
# ==========================================
const T = 65.0              # Temperature in K
const kB = 0.0019872041     # Boltzmann constant in kcal/(mol*K)
const beta = 1.0 / (kB * T)

const sigma = 3.315         # Angstroms
const epsilon = 0.0739      # kcal/mol
const r_cut = (2.0^(1.0/6.0)) * sigma 
const bond_length = 1.1     # Angstroms
const half_bond = bond_length / 2.0

# ==========================================
# 2. Simulation Parameters
# ==========================================
const N_blocks = 20
const N_steps = 50_000      # Steps per block
const dr = 0.1              # Grid spacing (Angstroms)
const N_grid = 2048         # Number of r grid points
const r_max = N_grid * dr
=#
# ==========================================
# 3. Helper Functions
# ==========================================

# Repulsive WCA Lennard-Jones Potential
@inline function repulsive_LJ(r::Float64)
    if r <= r_cut
        return 4.0 * epsilon * ((sigma/r)^12 - (sigma/r)^6) + epsilon
    else
        return 0.0
    end
end

# Truncated and Shifted Lennard-Jones Potential
@inline function lj_potential(r::Float64)
    if r <= r_cut
        return 4.0 * epsilon * ((sigma/r)^12 - (sigma/r)^6) - V_shift
    else
        return 0.0
    end
end

# Generates a uniform random 3D unit vector using StaticArrays
@inline function random_unit_vector()
    theta = 2.0 * pi * rand()
    w = 2.0 * rand() - 1.0
    r_xy = sqrt(1.0 - w^2)
    return SVector{3, Float64}(r_xy * cos(theta), r_xy * sin(theta), w)
end

# Fast binning function
@inline function add_to_hist!(hist, norm_arr, dist::Float64, weight::Float64, z_val::Float64)
    bin = floor(Int, dist / dr) + 1
    if bin <= N_grid
        z2 = z_val^2
        @inbounds hist[bin] += weight * z2
        @inbounds norm_arr[bin] += z2
    end
end

@inline function get_W(r::Float64, W_arr::Vector{Float64}, dr::Float64)
    # If r is beyond our grid, the solvation potential is 0.0
    if r >= length(W_arr) * dr
        return 0.0
    end
    
    # Fast linear interpolation logic goes here...
    # ...
end

# Fast linear interpolation for W(r) on a shifted grid: r_i = (i - 0.5) * dr
@inline function get_W(r::Float64, W_arr::Vector{Float64}, dr::Float64, N_grid::Int)
    # 1. Handle Long-Range Tail (r beyond the simulated grid)
    if r >= (N_grid - 0.5) * dr
        return 0.0
    end
    
    # 2. Handle Hard Core (r smaller than the first grid point)
    if r <= 0.5 * dr
        @inbounds return W_arr[1]
    end
    
    # 3. Fast Index Math
    idx_float = (r / dr) + 0.5
    idx_lower = floor(Int, idx_float)
    
    # 4. Fractional distance between the lower and upper bin
    fraction = idx_float - idx_lower
    
    # 5. Extract and interpolate (using @inbounds to skip bounds-checking overhead)
    @inbounds w_low = W_arr[idx_lower]
    @inbounds w_up  = W_arr[idx_lower + 1]
    
    return w_low + fraction * (w_up - w_low)
end

function run_mc_sampling(N_steps::Int, N_blocks::Int, W_AA::Vector{Float64}, W_BB::Vector{Float64}, W_AB::Vector{Float64})
    println("Starting MC Sampling block...")
    
# [Initialize your raw_hist and raw_norm matrices here]
# ==========================================
# 4. Memory Allocation
# ==========================================
raw_hist_AA = zeros(Float64, N_grid, N_blocks)
raw_norm_AA = zeros(Float64, N_grid, N_blocks)

raw_hist_BB = zeros(Float64, N_grid, N_blocks)
raw_norm_BB = zeros(Float64, N_grid, N_blocks)

raw_hist_AB = zeros(Float64, N_grid, N_blocks)
raw_norm_AB = zeros(Float64, N_grid, N_blocks)

# ==========================================
# 5. Multi-Threaded Direct Sampling Loop
# ==========================================
@threads for m in 1:N_blocks
    
    # Thread-local arrays
    local_hist_AA = zeros(Float64, N_grid)
    local_norm_AA = zeros(Float64, N_grid)
    
    local_hist_BB = zeros(Float64, N_grid)
    local_norm_BB = zeros(Float64, N_grid)
    
    local_hist_AB = zeros(Float64, N_grid)
    local_norm_AB = zeros(Float64, N_grid)
    
    for step in 1:N_steps
        # Draw random center of mass displacement along Z
        z = rand() * r_max
        com2 = SVector{3, Float64}(0.0, 0.0, z)
        
        # Draw random orientations
        u1 = random_unit_vector()
        u2 = random_unit_vector()
        
        # Site coordinates (Dimer 1 at origin)
        r1A =  half_bond * u1
        r1B = -half_bond * u1
        r2A = com2 + (half_bond * u2)
        r2B = com2 - (half_bond * u2)
        
        # Distances
        r_AA_dist = norm(r1A - r2A)
        r_BB_dist = norm(r1B - r2B)
        r_AB_dist = norm(r1A - r2B)
        r_BA_dist = norm(r1B - r2A)
        
        #=# 1. Bare Lennard-Jones Potential
        V_bare = repulsive_LJ(r_AA_dist) + repulsive_LJ(r_BB_dist) + 
                 repulsive_LJ(r_AB_dist) + repulsive_LJ(r_BA_dist)
        =#    
        # 1. Bare Lennard-Jones Potential
        V_bare = lj_potential(r_AA_dist) + lj_potential(r_BB_dist) + 
                 lj_potential(r_AB_dist) + lj_potential(r_BA_dist)     

        # 2. Add Dynamic Solvation Potential W(r)
         W_total = get_W(r_AA_dist, W_AA, dr, N_grid) + 
                   get_W(r_BB_dist, W_BB, dr, N_grid) + 
                   get_W(r_AB_dist, W_AB, dr, N_grid) + 
                   get_W(r_BA_dist, W_AB, dr, N_grid) # BA uses AB array by symmetry
        
        # 3. Total Boltzmann Weight
        weight = exp(-beta * (V_bare + W_total))
        
        # Bin the data
        add_to_hist!(local_hist_AA, local_norm_AA, r_AA_dist, weight, z)
        add_to_hist!(local_hist_BB, local_norm_BB, r_BB_dist, weight, z)
        add_to_hist!(local_hist_AB, local_norm_AB, r_AB_dist, weight, z)
        add_to_hist!(local_hist_AB, local_norm_AB, r_BA_dist, weight, z) # AB/BA symmetry
    end
    
    # Save block to global array
    raw_hist_AA[:, m] = local_hist_AA
    raw_norm_AA[:, m] = local_norm_AA
    
    raw_hist_BB[:, m] = local_hist_BB
    raw_norm_BB[:, m] = local_norm_BB
    
    raw_hist_AB[:, m] = local_hist_AB
    raw_norm_AB[:, m] = local_norm_AB
end
# ==========================================
    # Process Block Statistics (Converting Raw Hists to h(r))
    # ==========================================
    h_blocks_AA = zeros(Float64, N_grid, N_blocks)
    h_blocks_BB = zeros(Float64, N_grid, N_blocks)
    h_blocks_AB = zeros(Float64, N_grid, N_blocks)

    for m in 1:N_blocks
        h_blocks_AA[:, m] = (raw_hist_AA[:, m] ./ (raw_norm_AA[:, m] .+ 1e-10)) .- 1.0
        h_blocks_BB[:, m] = (raw_hist_BB[:, m] ./ (raw_norm_BB[:, m] .+ 1e-10)) .- 1.0
        h_blocks_AB[:, m] = (raw_hist_AB[:, m] ./ (raw_norm_AB[:, m] .+ 1e-10)) .- 1.0
    end

    # Calculate means and variances across the blocks, and flatten them to 1D vectors
    h_AA_mean = vec(mean(h_blocks_AA, dims=2))
    h_AA_var  = vec(var(h_blocks_AA, dims=2))
    
    h_BB_mean = vec(mean(h_blocks_BB, dims=2))
    h_BB_var  = vec(var(h_blocks_BB, dims=2))
    
    h_AB_mean = vec(mean(h_blocks_AB, dims=2))
    h_AB_var  = vec(var(h_blocks_AB, dims=2))

    return h_AA_mean, h_AA_var, h_BB_mean, h_BB_var, h_AB_mean, h_AB_var
end # <-- End of run_mc_sampling function

#=
println("Sampling complete. Processing statistics...")

# ==========================================
# 6. Data Processing
# ==========================================
#r_grid = [ (i - 0.5) * dr for i in 1:N_grid ]

# --- Old Way: Global Average (AA only for the baseline plot) ---
global_hist_AA = sum(raw_hist_AA, dims=2)
global_norm_AA = sum(raw_norm_AA, dims=2)
h_AA_old = vec((global_hist_AA ./ (global_norm_AA .+ 1e-10)) .- 1.0)

# --- New Way: Block Averaging for all pairs ---
h_blocks_AA = zeros(Float64, N_grid, N_blocks)
h_blocks_BB = zeros(Float64, N_grid, N_blocks)
h_blocks_AB = zeros(Float64, N_grid, N_blocks)

for m in 1:N_blocks
    h_blocks_AA[:, m] = (raw_hist_AA[:, m] ./ (raw_norm_AA[:, m] .+ 1e-10)) .- 1.0
    h_blocks_BB[:, m] = (raw_hist_BB[:, m] ./ (raw_norm_BB[:, m] .+ 1e-10)) .- 1.0
    h_blocks_AB[:, m] = (raw_hist_AB[:, m] ./ (raw_norm_AB[:, m] .+ 1e-10)) .- 1.0
end

h_AA_mean = zeros(Float64, N_grid); h_AA_var = zeros(Float64, N_grid)
h_BB_mean = zeros(Float64, N_grid); h_BB_var = zeros(Float64, N_grid)
h_AB_mean = zeros(Float64, N_grid); h_AB_var = zeros(Float64, N_grid)

for i in 1:N_grid
    h_AA_mean[i] = mean(h_blocks_AA[i, :])
    h_AA_var[i]  = var(h_blocks_AA[i, :])
    
    h_BB_mean[i] = mean(h_blocks_BB[i, :])
    h_BB_var[i]  = var(h_blocks_BB[i, :])
    
    h_AB_mean[i] = mean(h_blocks_AB[i, :])
    h_AB_var[i]  = var(h_blocks_AB[i, :])
end



# ==========================================
# 7. Visualization
# ==========================================
println("Generating plot...")

# 1. Old Way: Scatter plot (raw data points)
p = scatter(r_grid, h_AA_old, label="Old Way (Global Sum)", 
            markersize=3, markercolor=:black, markerstrokewidth=0, alpha=0.4, 
            xlabel="r (Angstroms)", ylabel="h(r)", xlims=(0, 10))

# 2. New Way: Connected lines showing the Block Mean
plot!(p, r_grid, h_AA_mean, label="New Way (Block Mean)", 
      linewidth=1.5, linecolor=:red)

# 3. The Payoff: Plot the Standard Deviation as a shaded ribbon around the mean!
# We use sqrt(variance) to get standard deviation.
plot!(p, r_grid, h_AA_mean, ribbon=sqrt.(h_AA_var), 
      fillalpha=0.3, fillcolor=:red, label="± 1 Std Dev (Noise)", 
      linewidth=0) # linewidth=0 so it just draws the shading, not an extra line

savefig(p, "N2_Sampling_Baseline_Updated.png")
println("Done! Plot saved as 'N2_Sampling_Baseline_Updated.png'.")

# ==========================================
# 8. Phase 3: The Constrained GP (Zero-Tolerance Rules)
# ==========================================
println("Building Constrained GP with full RISM sum rules...")

r_val = r_grid
N_v = length(r_val)

y_AA = h_AA_mean; v_AA = h_AA_var .+ 1e-8
y_BB = h_BB_mean; v_BB = h_BB_var .+ 1e-8
y_AB = h_AB_mean; v_AB = h_AB_var .+ 1e-8

y_stack = vcat(y_AA, y_BB, y_AB)
v_stack = vcat(v_AA, v_BB, v_AB)

l_scale = 0.5 
sig_var = 1.0 
kernel = SqExponentialKernel() ∘ ScaleTransform(1.0 / l_scale)

K_base = sig_var .* kernelmatrix(kernel, r_val)
K_joint = zeros(Float64, 3*N_v, 3*N_v)
K_joint[1:N_v, 1:N_v]                 = K_base
K_joint[N_v+1:2*N_v, N_v+1:2*N_v]     = K_base
K_joint[2*N_v+1:3*N_v, 2*N_v+1:3*N_v] = K_base

# 3. Construct the Constraint Matrix (C_mat) - 3 Exact Rows
C_mat = zeros(Float64, 3, 3*N_v)
for i in 1:N_v
    w2_i = (r_val[i]^2) * dr
    w4_i = (r_val[i]^4) * dr
    
    # Constraint 1: O(1) [AA - AB = 0]
    C_mat[1, i]           =  w2_i
    C_mat[1, 2*N_v + i]   = -w2_i
    
    # Constraint 2: O(1) [BB - AB = 0]
    C_mat[2, N_v + i]     =  w2_i
    C_mat[2, 2*N_v + i]   = -w2_i
    
    # Constraint 3: O(k^2) [AA + BB - 2AB = 0]
    C_mat[3, i]           =  w4_i
    C_mat[3, N_v + i]     =  w4_i
    C_mat[3, 2*N_v + i]   = -2.0 * w4_i
end

# Normalize to keep matrix conditioning perfect
C_mat[1, :] ./= norm(C_mat[1, :])
C_mat[2, :] ./= norm(C_mat[2, :])
C_mat[3, :] ./= norm(C_mat[3, :])

# 4. Build Observation Matrix (H_tilde)
H_tilde = zeros(Float64, 3*N_v + 3, 3*N_v)
H_tilde[1:3*N_v, 1:3*N_v] = I(3*N_v)
H_tilde[3*N_v + 1 : 3*N_v + 3, :] = C_mat

y_tilde = vcat(y_stack, [0.0, 0.0, 0.0])

# VITAL FIX: Exact 0.0 for constraint variance!
Sigma_tilde = diagm(vcat(v_stack, [0.0, 0.0, 0.0]))

# 5. Calculate Exact Posterior Mean
println("Inverting well-conditioned matrix...")
S = H_tilde * K_joint * transpose(H_tilde) + Sigma_tilde
mu_post = K_joint * transpose(H_tilde) * (S \ y_tilde)

h_AA_smooth = mu_post[1:N_v]
h_BB_smooth = mu_post[N_v+1:2*N_v]
h_AB_smooth = mu_post[2*N_v+1:3*N_v]
println("Fully-Constrained GP calculation complete!")

# ==========================================
# 9. Verification & Plotting
# ==========================================
chk_AA_AB_O1 = zeros(Float64, 3*N_v)
chk_BB_AB_O1 = zeros(Float64, 3*N_v)
chk_Det_O1   = zeros(Float64, 3*N_v)
chk_Det_Ok2  = zeros(Float64, 3*N_v)

for i in 1:N_v
    w2 = (r_val[i]^2)*dr
    w4 = (r_val[i]^4)*dr
    
    chk_AA_AB_O1[i] = w2; chk_AA_AB_O1[2*N_v + i] = -w2
    chk_BB_AB_O1[N_v + i] = w2; chk_BB_AB_O1[2*N_v + i] = -w2
    
    chk_Det_O1[i] = w2; chk_Det_O1[N_v + i] = w2; chk_Det_O1[2*N_v + i] = -2.0*w2
    chk_Det_Ok2[i] = w4; chk_Det_Ok2[N_v + i] = w4; chk_Det_Ok2[2*N_v + i] = -2.0*w4
end

println("\n--- COMPREHENSIVE SUM RULE VERIFICATION ---")
println("O(1) [AA - AB]        Raw: ", sum(chk_AA_AB_O1 .* y_stack), " | Smooth: ", sum(chk_AA_AB_O1 .* mu_post))
println("O(1) [BB - AB]        Raw: ", sum(chk_BB_AB_O1 .* y_stack), " | Smooth: ", sum(chk_BB_AB_O1 .* mu_post))
println("O(1) [AA+BB-2AB]      Raw: ", sum(chk_Det_O1 .* y_stack),   " | Smooth: ", sum(chk_Det_O1 .* mu_post))
println("O(k^2) [AA+BB-2AB]    Raw: ", sum(chk_Det_Ok2 .* y_stack),  " | Smooth: ", sum(chk_Det_Ok2 .* mu_post))
println("-------------------------------------------\n")

# ==========================================
# 10. Phase 4: k-Space Verification
# ==========================================
println("\nPerforming 3D Fourier Transforms...")

# Define a high-resolution k-grid focusing on the low-k limit
# (Avoid k exactly 0 to prevent divide-by-zero, start at k=0.01)
#k_grid = range(0.01, 1.5, length=300)

# 3D Spherical Fourier Transform Function
function calc_h_hat(r_arr, h_arr, k_val)
    integral = 0.0
    # Simple Riemann sum over the grid
    for i in 1:length(r_arr)
        integral += r_arr[i] * h_arr[i] * sin(k_val * r_arr[i]) * dr
    end
    return (4.0 * pi / k_val) * integral
end

# Arrays to hold the linear combination \hat{\Delta}(k)
delta_k_raw = zeros(Float64, length(k_grid))
delta_k_smooth = zeros(Float64, length(k_grid))

for (idx, k) in enumerate(k_grid)
    # Raw Data FTs
    h_hat_AA_raw = calc_h_hat(r_grid, h_AA_mean, k)
    h_hat_BB_raw = calc_h_hat(r_grid, h_BB_mean, k)
    h_hat_AB_raw = calc_h_hat(r_grid, h_AB_mean, k)
    
    # Smooth GP FTs
    h_hat_AA_smooth = calc_h_hat(r_grid, h_AA_smooth, k)
    h_hat_BB_smooth = calc_h_hat(r_grid, h_BB_smooth, k)
    h_hat_AB_smooth = calc_h_hat(r_grid, h_AB_smooth, k)
    
    # The Critical Combination: AA + BB - 2AB
    delta_k_raw[idx] = h_hat_AA_raw + h_hat_BB_raw - 2.0 * h_hat_AB_raw
    delta_k_smooth[idx] = h_hat_AA_smooth + h_hat_BB_smooth - 2.0 * h_hat_AB_smooth
end

# ==========================================
# 11. Plotting the Divergence Cure
# ==========================================
println("Plotting k-space results...")

# We plot \hat{\Delta}(k) / k^2. 
# In the RISM equations, this term sits in the denominator as 1 / k^4. 
# If \hat{\Delta}(k) scales as k^2 (due to noise), dividing by k^2 gives a non-zero intercept.
# If \hat{\Delta}(k) scales as k^4 (our GP fix), dividing by k^2 drives it safely to ZERO at the origin.

p4 = plot(k_grid, delta_k_raw ./ (k_grid.^2), 
          label="Raw MC Noise (Divergent)", linewidth=2, color=:red, linestyle=:dash)

plot!(p4, k_grid, delta_k_smooth ./ (k_grid.^2), 
      label="Constrained GP (Stable)", linewidth=3, color=:blue)

plot!(p4, xlabel="k (1/Angstroms)", ylabel="Δh(k) / k²", 
      title="Resolution of the low-k Divergence", 
      xlims=(0, 1.5))

savefig(p4, "N2_k_space_Fix.png")
println("Done! Final k-space plot saved as 'N2_k_space_Fix.png'.")
=#