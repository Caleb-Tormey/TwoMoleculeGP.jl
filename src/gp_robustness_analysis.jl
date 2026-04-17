using Random, Statistics, LinearAlgebra, Plots, AbstractGPs, KernelFunctions, Base.Threads,StaticArrays

# ==========================================
# Global Parameters
# ==========================================
const N_grid = 2048
const dr = 0.1
const r_grid = [(i - 0.5) * dr for i in 1:N_grid]
const r_max = N_grid * dr
const N_blocks = 20
const N_trials = 20      # Number of independent MC runs
const base_seed = 42     # PRNG Seed for reproducibility

# k-grid for divergence analysis
const k_grid = range(0.01, 1.5, length=300)

# Physical parameters for the simulation
const sigma = 3.315
const epsilon = 65.0
const r_cut = (2.0^(1.0/6.0)) * sigma 
const bond_length = 1.1     # Angstroms
const half_bond = bond_length / 2.0

const T = 65.0              # Temperature in K
const kB = 0.0019872041     # Boltzmann constant in kcal/(mol*K)
const beta = 1.0 / (kB * T)
# ==========================================


const N_steps = 50000    # <--- ADD THIS HERE!
# const rho = 0.01851    # <--- (Add this too if your loop uses it!)
# const box_L = 40.0     # <--- (Add this too if your loop uses it!)
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
# ==========================================
# Trial Function
# ==========================================
function run_single_trial(trial_idx, seed_val)
    Random.seed!(seed_val)
    println("  -> Starting MC Sampling for Trial $trial_idx...")

    # ---------------------------------------------------------
    # STEP 1: INITIALIZE MATRICES AND RUN SAMPLING
    # ---------------------------------------------------------
    # These are pre-allocated so your threaded loop can see them!
    raw_hist_AA = zeros(Float64, N_grid, N_blocks)
    raw_norm_AA = zeros(Float64, N_grid, N_blocks)
    
    raw_hist_BB = zeros(Float64, N_grid, N_blocks)
    raw_norm_BB = zeros(Float64, N_grid, N_blocks)
    
    raw_hist_AB = zeros(Float64, N_grid, N_blocks)
    raw_norm_AB = zeros(Float64, N_grid, N_blocks)

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # PASTE YOUR MONTE CARLO THREADED LOOP HERE
    # (The part that calculates distances and calls add_to_hist!)
     
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
        
            # Calculate Boltzmann weight (Bare potential only, W(r) = 0 for baseline)
            V_total = repulsive_LJ(r_AA_dist) + repulsive_LJ(r_BB_dist) + 
                      repulsive_LJ(r_AB_dist) + repulsive_LJ(r_BA_dist)
        
            weight = exp(-beta * V_total)
        
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

    println("Sampling complete. Processing statistics...")


    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # ---------------------------------------------------------
    # STEP 2: Block Averaging
    # ---------------------------------------------------------
    h_blocks_AA = zeros(Float64, N_grid, N_blocks)
    h_blocks_BB = zeros(Float64, N_grid, N_blocks)
    h_blocks_AB = zeros(Float64, N_grid, N_blocks)

    for m in 1:N_blocks
        h_blocks_AA[:, m] = (raw_hist_AA[:, m] ./ (raw_norm_AA[:, m] .+ 1e-10)) .- 1.0
        h_blocks_BB[:, m] = (raw_hist_BB[:, m] ./ (raw_norm_BB[:, m] .+ 1e-10)) .- 1.0
        h_blocks_AB[:, m] = (raw_hist_AB[:, m] ./ (raw_norm_AB[:, m] .+ 1e-10)) .- 1.0
    end

    h_AA_mean = vec(mean(h_blocks_AA, dims=2)); h_AA_var = vec(var(h_blocks_AA, dims=2))
    h_BB_mean = vec(mean(h_blocks_BB, dims=2)); h_BB_var = vec(var(h_blocks_BB, dims=2))
    h_AB_mean = vec(mean(h_blocks_AB, dims=2)); h_AB_var = vec(var(h_blocks_AB, dims=2))

    # ---------------------------------------------------------
    # STEP 3: Constrained GP (Zero-Tolerance Rules)
    # ---------------------------------------------------------
    println("  -> Fitting Constrained GP...")
    y_stack = vcat(h_AA_mean, h_BB_mean, h_AB_mean)
    v_stack = vcat(h_AA_var .+ 1e-8, h_BB_var .+ 1e-8, h_AB_var .+ 1e-8)

    l_scale = 0.25  # Lower length scale to hug data tighter
    sig_var = 1.0 
    #kernel = SqExponentialKernel() ∘ ScaleTransform(1.0 / l_scale)
    kernel = Matern52Kernel() ∘ ScaleTransform(1.0 / l_scale) # Switched to Matern 5/2
    
    K_base = sig_var .* kernelmatrix(kernel, r_grid)
    K_joint = zeros(Float64, 3*N_grid, 3*N_grid)
    K_joint[1:N_grid, 1:N_grid]                 = K_base
    K_joint[N_grid+1:2*N_grid, N_grid+1:2*N_grid]     = K_base
    K_joint[2*N_grid+1:3*N_grid, 2*N_grid+1:3*N_grid] = K_base

    # Build Exact Constraints
    # --- REPLACE THE C_mat LOOP IN STEP 3 WITH THIS ---
    C_mat = zeros(Float64, 3, 3*N_grid)
    r_constraint_max = 40.0 # Ignore ideal gas noise beyond 40 Angstroms
    
    for i in 1:N_grid
        if r_grid[i] <= r_constraint_max
            w2_i = (r_grid[i]^2) * dr
            w4_i = (r_grid[i]^4) * dr
            
            # O(1) AA - AB = 0
            C_mat[1, i]              =  w2_i
            C_mat[1, 2*N_grid + i]   = -w2_i
            
            # O(1) BB - AB = 0
            C_mat[2, N_grid + i]     =  w2_i
            C_mat[2, 2*N_grid + i]   = -w2_i
            
            # O(k^2) AA + BB - 2AB = 0
            C_mat[3, i]              =  w4_i
            C_mat[3, N_grid + i]     =  w4_i
            C_mat[3, 2*N_grid + i]   = -2.0 * w4_i
        end
    end

    # Normalize for numerical stability
    C_mat[1, :] ./= norm(C_mat[1, :])
    C_mat[2, :] ./= norm(C_mat[2, :])
    C_mat[3, :] ./= norm(C_mat[3, :])

    H_tilde = zeros(Float64, 3*N_grid + 3, 3*N_grid)
    H_tilde[1:3*N_grid, 1:3*N_grid] = I(3*N_grid)
    H_tilde[3*N_grid + 1 : 3*N_grid + 3, :] = C_mat

    y_tilde = vcat(y_stack, [0.0, 0.0, 0.0])
    
    # 0.0 variance on constraints to force absolute physical compliance
    Sigma_tilde = diagm(vcat(v_stack, [0.0, 0.0, 0.0]))

    # Solve
    S = H_tilde * K_joint * transpose(H_tilde) + Sigma_tilde
    mu_post = K_joint * transpose(H_tilde) * (S \ y_tilde)

    h_AA_smooth = mu_post[1:N_grid]
    h_BB_smooth = mu_post[N_grid+1:2*N_grid]
    h_AB_smooth = mu_post[2*N_grid+1:3*N_grid]

    # Calculate exact un-normalized residual for the O(k^2) rule for statistics
    chk_O1_AA = zeros(Float64, 3*N_grid)
    chk_O1_BB = zeros(Float64, 3*N_grid)
    chk_Ok2   = zeros(Float64, 3*N_grid)
    
    for i in 1:N_grid
        if r_grid[i] <= r_constraint_max
            w2 = (r_grid[i]^2)*dr
            w4 = (r_grid[i]^4)*dr
            
            chk_O1_AA[i] = w2; chk_O1_AA[2*N_grid + i] = -w2
            chk_O1_BB[N_grid + i] = w2; chk_O1_BB[2*N_grid + i] = -w2
            chk_Ok2[i] = w4; chk_Ok2[N_grid + i] = w4; chk_Ok2[2*N_grid + i] = -2.0*w4
        end
    end
    
    raw_O1_AA_int = sum(chk_O1_AA .* y_stack)
    sm_O1_AA_int  = sum(chk_O1_AA .* mu_post)
    
    raw_O1_BB_int = sum(chk_O1_BB .* y_stack)
    sm_O1_BB_int  = sum(chk_O1_BB .* mu_post)
    
    raw_Ok2_int = sum(chk_Ok2 .* y_stack)
    sm_Ok2_int  = sum(chk_Ok2 .* mu_post)

    # ---------------------------------------------------------
    # STEP 4: k-space Transform
    # ---------------------------------------------------------
    println("  -> Verifying k-space...")
    delta_k_raw = zeros(Float64, length(k_grid))
    delta_k_smooth = zeros(Float64, length(k_grid))

    for (idx, k) in enumerate(k_grid)
        h_hat_AA_raw = (4.0 * pi / k) * sum(r_grid .* h_AA_mean .* sin.(k .* r_grid) .* dr)
        h_hat_BB_raw = (4.0 * pi / k) * sum(r_grid .* h_BB_mean .* sin.(k .* r_grid) .* dr)
        h_hat_AB_raw = (4.0 * pi / k) * sum(r_grid .* h_AB_mean .* sin.(k .* r_grid) .* dr)
        
        h_hat_AA_sm = (4.0 * pi / k) * sum(r_grid .* h_AA_smooth .* sin.(k .* r_grid) .* dr)
        h_hat_BB_sm = (4.0 * pi / k) * sum(r_grid .* h_BB_smooth .* sin.(k .* r_grid) .* dr)
        h_hat_AB_sm = (4.0 * pi / k) * sum(r_grid .* h_AB_smooth .* sin.(k .* r_grid) .* dr)
        
        delta_k_raw[idx] = h_hat_AA_raw + h_hat_BB_raw - 2.0 * h_hat_AB_raw
        delta_k_smooth[idx] = h_hat_AA_sm + h_hat_BB_sm - 2.0 * h_hat_AB_sm
    end

    return h_AA_mean, h_AA_smooth, delta_k_raw, delta_k_smooth, raw_O1_AA_int, sm_O1_AA_int, raw_O1_BB_int, sm_O1_BB_int, raw_Ok2_int, sm_Ok2_int
end

# ==========================================
# Master Execution & Data Collection
# ==========================================
println("Starting Ensemble Robustness Analysis ($N_trials trials)...")

all_h_raw = zeros(Float64, N_grid, N_trials)
all_h_smooth = zeros(Float64, N_grid, N_trials)
all_delta_raw = zeros(Float64, length(k_grid), N_trials)
all_delta_smooth = zeros(Float64, length(k_grid), N_trials)

stat_raw_O1_AA = zeros(Float64, N_trials); stat_sm_O1_AA = zeros(Float64, N_trials)
stat_raw_O1_BB = zeros(Float64, N_trials); stat_sm_O1_BB = zeros(Float64, N_trials)
stat_raw_Ok2   = zeros(Float64, N_trials); stat_sm_Ok2   = zeros(Float64, N_trials)
stat_low_k_intercept = zeros(Float64, N_trials)

for t in 1:N_trials
    println("\n=== TRIAL $t / $N_trials ===")
    seed = base_seed + t
    h_r, h_s, d_r, d_s, rO1A, sO1A, rO1B, sO1B, rOk2, sOk2 = run_single_trial(t, seed)
    
    all_h_raw[:, t] = h_r
    all_h_smooth[:, t] = h_s
    all_delta_raw[:, t] = d_r
    all_delta_smooth[:, t] = d_s
    
    stat_raw_O1_AA[t] = rO1A; stat_sm_O1_AA[t] = sO1A
    stat_raw_O1_BB[t] = rO1B; stat_sm_O1_BB[t] = sO1B
    stat_raw_Ok2[t]   = rOk2; stat_sm_Ok2[t]   = sOk2
    stat_low_k_intercept[t] = d_s[1] / (k_grid[1]^2) 
end

# ==========================================
# Statistical Processing & Output
# ==========================================
println("\n\n==========================================")
println("ENSEMBLE STATISTICS ACROSS $N_trials RUNS")
println("==========================================")

avg_raw_std = mean(std(all_h_raw, dims=2))
avg_smooth_std = mean(std(all_h_smooth, dims=2))
println("Average Raw Pointwise StdDev:    ", avg_raw_std)
println("Average Smooth Pointwise StdDev: ", avg_smooth_std)
println("Variance Reduction Ratio:        ", avg_raw_std / avg_smooth_std, "x")

println("\n--- EXACT SUM RULE RESIDUALS (Evaluated to r=40A) ---")
println("O(1) [AA - AB]:")
println("  Raw Error:       ", mean(stat_raw_O1_AA), " ± ", std(stat_raw_O1_AA))
println("  GP Constrained:  ", mean(stat_sm_O1_AA), " ± ", std(stat_sm_O1_AA))

println("\nO(1) [BB - AB]:")
println("  Raw Error:       ", mean(stat_raw_O1_BB), " ± ", std(stat_raw_O1_BB))
println("  GP Constrained:  ", mean(stat_sm_O1_BB), " ± ", std(stat_sm_O1_BB))

println("\nO(k^2) [AA + BB - 2AB]:")
println("  Raw Error:       ", mean(stat_raw_Ok2), " ± ", std(stat_raw_Ok2))
println("  GP Constrained:  ", mean(stat_sm_Ok2), " ± ", std(stat_sm_Ok2))

println("\n--- k-SPACE STABILITY ---")
println("Low-k Intercept Stability (Δh(k)/k² at k=0.01):")
println("  Mean Intercept:  ", mean(stat_low_k_intercept))
println("  Intercept StdDev:", std(stat_low_k_intercept))
println("==========================================\n")


# ==========================================
# Plotting the "Envelope" and "Funnel"
# ==========================================
println("Generating Publication Plots...")

# 1. The R-space Envelope Plot
h_smooth_mean = vec(mean(all_h_smooth, dims=2))
h_smooth_std  = vec(std(all_h_smooth, dims=2))

p_env = plot(r_grid, h_smooth_mean, ribbon=2.0.*h_smooth_std, 
             fillalpha=0.4, fillcolor=:blue, linecolor=:blue, linewidth=2,
             label="GP Mean ± 2σ", xlabel="r (Angstroms)", ylabel="h(r)", 
             xlims=(0, 10), title="GP Envelope Consistency")

scatter!(p_env, r_grid, all_h_raw[:, 1], color=:black, markersize=2, alpha=0.3, 
         markerstrokewidth=0, label="Sample Raw Data")

savefig(p_env, "GP_Envelope_rspace.png")


# 2. The K-space Funnel Plot
p_funnel = plot(xlabel="k (1/Angstroms)", ylabel="Δh(k) / k²", 
                title="k-Space Divergence Resolution", xlims=(0, 1.5))

# Plot all raw lines (red dashed)
for t in 1:N_trials
    lbl = t == 1 ? "Raw MC Trials" : ""
    plot!(p_funnel, k_grid, all_delta_raw[:, t] ./ (k_grid.^2), 
          color=:red, linestyle=:dash, alpha=0.3, linewidth=1, label=lbl)
end

# Plot all GP lines (solid blue)
for t in 1:N_trials
    lbl = t == 1 ? "GP Constrained Trials" : ""
    plot!(p_funnel, k_grid, all_delta_smooth[:, t] ./ (k_grid.^2), 
          color=:blue, alpha=0.5, linewidth=2, label=lbl)
end

savefig(p_funnel, "GP_Funnel_kspace.png")
println("Done! Outputs saved as 'GP_Envelope_rspace.png' and 'GP_Funnel_kspace.png'.")