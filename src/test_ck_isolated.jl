using Base.Threads, Statistics, LinearAlgebra, StaticArrays, Plots, AbstractGPs, KernelFunctions, Printf, DelimitedFiles

println("Starting Isolated c(k) Sandbox Test...")

# ==========================================
# 1. Parameters
# ==========================================
const T = 65.0
const kB = 0.0019872041
const beta = 1.0 / (kB * T)
const sigma = 3.315
const epsilon = 65.0 
const r_cut = (2.0^(1.0/6.0)) * sigma 
const V_shift = 4.0 * epsilon * ((sigma/r_cut)^12 - (sigma/r_cut)^6)
const bond_length = 1.1     
const half_bond = bond_length / 2.0
const rho = 0.01851 

const N_blocks = 20
const N_steps = 50_000
const dr = 0.1
const N_grid = 2048
const r_max = N_grid * dr
const r_grid = [(i - 0.5) * dr for i in 1:N_grid]
const k_grid = range(0.01, 1.5, length=300)

# ==========================================
# 2. Helper Functions
# ==========================================
@inline function lj_potential(r::Float64)
    if r <= r_cut return 4.0 * epsilon * ((sigma/r)^12 - (sigma/r)^6) - V_shift else return 0.0 end
end

@inline function random_unit_vector()
    theta = 2.0 * pi * rand(); w = 2.0 * rand() - 1.0; r_xy = sqrt(1.0 - w^2)
    return SVector{3, Float64}(r_xy * cos(theta), r_xy * sin(theta), w)
end

@inline function add_to_hist!(hist, norm_arr, dist::Float64, weight::Float64, z_val::Float64)
    bin = floor(Int, dist / dr) + 1
    if bin <= N_grid
        z2 = z_val^2
        @inbounds hist[bin] += weight * z2
        @inbounds norm_arr[bin] += z2
    end
end

# ==========================================
# 3. Sampling
# ==========================================
raw_hist_AA = zeros(Float64, N_grid, N_blocks); raw_norm_AA = zeros(Float64, N_grid, N_blocks)
raw_hist_BB = zeros(Float64, N_grid, N_blocks); raw_norm_BB = zeros(Float64, N_grid, N_blocks)
raw_hist_AB = zeros(Float64, N_grid, N_blocks); raw_norm_AB = zeros(Float64, N_grid, N_blocks)

@threads for m in 1:N_blocks
    loc_h_AA = zeros(N_grid); loc_n_AA = zeros(N_grid)
    loc_h_BB = zeros(N_grid); loc_n_BB = zeros(N_grid)
    loc_h_AB = zeros(N_grid); loc_n_AB = zeros(N_grid)
    
    for step in 1:N_steps
        z = rand() * r_max; com2 = SVector{3, Float64}(0.0, 0.0, z)
        u1 = random_unit_vector(); u2 = random_unit_vector()
        
        r1A = half_bond * u1; r1B = -half_bond * u1
        r2A = com2 + (half_bond * u2); r2B = com2 - (half_bond * u2)
        
        rAA = norm(r1A - r2A); rBB = norm(r1B - r2B)
        rAB = norm(r1A - r2B); rBA = norm(r1B - r2A)
        
        V = lj_potential(rAA) + lj_potential(rBB) + lj_potential(rAB) + lj_potential(rBA)
        w = exp(-beta * V)
        
        add_to_hist!(loc_h_AA, loc_n_AA, rAA, w, z); add_to_hist!(loc_h_BB, loc_n_BB, rBB, w, z)
        add_to_hist!(loc_h_AB, loc_n_AB, rAB, w, z); add_to_hist!(loc_h_AB, loc_n_AB, rBA, w, z)
    end
    raw_hist_AA[:, m] = loc_h_AA; raw_norm_AA[:, m] = loc_n_AA
    raw_hist_BB[:, m] = loc_h_BB; raw_norm_BB[:, m] = loc_n_BB
    raw_hist_AB[:, m] = loc_h_AB; raw_norm_AB[:, m] = loc_n_AB
end

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

# ==========================================
# 4. GP Smoothing
# ==========================================
println("Fitting GP...")
y_stack = vcat(h_AA_mean, h_BB_mean, h_AB_mean)
v_stack = vcat(h_AA_var .+ 1e-8, h_BB_var .+ 1e-8, h_AB_var .+ 1e-8)

l_scale = 0.25 
sig_var = 1.0 
kernel = Matern52Kernel() ∘ ScaleTransform(1.0 / l_scale) 

K_base = sig_var .* kernelmatrix(kernel, r_grid)
K_joint = zeros(Float64, 3*N_grid, 3*N_grid)
K_joint[1:N_grid, 1:N_grid]                       = K_base
K_joint[N_grid+1:2*N_grid, N_grid+1:2*N_grid]     = K_base
K_joint[2*N_grid+1:3*N_grid, 2*N_grid+1:3*N_grid] = K_base

C_mat = zeros(Float64, 3, 3*N_grid)
r_constraint_max = 40.0 

for i in 1:N_grid
    if r_grid[i] <= r_constraint_max
        w2_i = (r_grid[i]^2) * dr; w4_i = (r_grid[i]^4) * dr
        C_mat[1, i] = w2_i; C_mat[1, 2*N_grid + i] = -w2_i
        C_mat[2, N_grid + i] = w2_i; C_mat[2, 2*N_grid + i] = -w2_i
        C_mat[3, i] = w4_i; C_mat[3, N_grid + i] = w4_i; C_mat[3, 2*N_grid + i] = -2.0 * w4_i
    end
end

C_mat[1, :] ./= norm(C_mat[1, :])
C_mat[2, :] ./= norm(C_mat[2, :])
C_mat[3, :] ./= norm(C_mat[3, :])

H_tilde = zeros(Float64, 3*N_grid + 3, 3*N_grid)
H_tilde[1:3*N_grid, 1:3*N_grid] = I(3*N_grid)
H_tilde[3*N_grid + 1 : 3*N_grid + 3, :] = C_mat

y_tilde = vcat(y_stack, [0.0, 0.0, 0.0])
Sigma_tilde = diagm(vcat(v_stack, [0.0, 0.0, 0.0]))

S = H_tilde * K_joint * transpose(H_tilde) + Sigma_tilde
mu_post = K_joint * transpose(H_tilde) * (S \ y_tilde)

h_AA_sm = mu_post[1:N_grid]
h_BB_sm = mu_post[N_grid+1:2*N_grid]
h_AB_sm = mu_post[2*N_grid+1:3*N_grid]

# ==========================================
# 5. Fourier Transforms
# ==========================================
println("Calculating FTs and RISM c(k)...")
delta_k_raw = zeros(length(k_grid)); delta_k_sm = zeros(length(k_grid))
c_AA_raw = zeros(length(k_grid)); c_AA_sm = zeros(length(k_grid))

for (idx, k) in enumerate(k_grid)
    # FT Raw
    h_hat_AA_raw = (4.0 * pi / k) * sum(r_grid .* h_AA_mean .* sin.(k .* r_grid) .* dr)
    h_hat_BB_raw = (4.0 * pi / k) * sum(r_grid .* h_BB_mean .* sin.(k .* r_grid) .* dr)
    h_hat_AB_raw = (4.0 * pi / k) * sum(r_grid .* h_AB_mean .* sin.(k .* r_grid) .* dr)
    
    # FT Smooth
    h_hat_AA_sm = (4.0 * pi / k) * sum(r_grid .* h_AA_sm .* sin.(k .* r_grid) .* dr)
    h_hat_BB_sm = (4.0 * pi / k) * sum(r_grid .* h_BB_sm .* sin.(k .* r_grid) .* dr)
    h_hat_AB_sm = (4.0 * pi / k) * sum(r_grid .* h_AB_sm .* sin.(k .* r_grid) .* dr)
    
    delta_k_raw[idx] = h_hat_AA_raw + h_hat_BB_raw - 2.0 * h_hat_AB_raw
    delta_k_sm[idx]  = h_hat_AA_sm + h_hat_BB_sm - 2.0 * h_hat_AB_sm

    # RISM Inversion
    w_cross = sin(k * bond_length) / (k * bond_length)
    w_plus = 1.0 + w_cross; w_minus = 1.0 - w_cross

    # Raw c(k)
    h_sym_r = (h_hat_AA_raw + h_hat_BB_raw) / 2.0
    h_p_r = h_sym_r + h_hat_AB_raw; h_m_r = h_sym_r - h_hat_AB_raw
    c_p_r = h_p_r / (w_plus^2 + rho * w_plus * h_p_r)
    den_m_r = w_minus^2 + rho * w_minus * h_m_r
    c_m_r = abs(den_m_r) > 1e-12 ? h_m_r / den_m_r : 0.0
    c_AA_raw[idx] = (c_p_r + c_m_r) / 2.0

    # Smooth c(k)
    h_sym_s = (h_hat_AA_sm + h_hat_BB_sm) / 2.0
    h_p_s = h_sym_s + h_hat_AB_sm; h_m_s = h_sym_s - h_hat_AB_sm
    c_p_s = h_p_s / (w_plus^2 + rho * w_plus * h_p_s)
    den_m_s = w_minus^2 + rho * w_minus * h_m_s
    c_m_s = abs(den_m_s) > 1e-12 ? h_m_s / den_m_s : 0.0
    c_AA_sm[idx] = (c_p_s + c_m_s) / 2.0
end

# ==========================================
# 6. Plotting & Saving to CSV
# ==========================================
println("Generating Plots and CSV files...")

p1 = plot(k_grid, delta_k_raw ./ (k_grid.^2), label="Raw", title="Sum Rule Fix (1/k^2)", linewidth=2)
plot!(p1, k_grid, delta_k_sm ./ (k_grid.^2), label="Smooth", linewidth=2)
savefig(p1, "Test_01_Delta_k2.png")

p2 = plot(k_grid, c_AA_raw, label="Raw c(k)", title="Analytical RISM c(k)", linewidth=2)
plot!(p2, k_grid, c_AA_sm, label="Smooth c(k)", linewidth=2)
savefig(p2, "Test_02_ck.png")

# Write Test 01 CSV
open("Test_01_Delta_k2.csv", "w") do io
    println(io, "k,delta_raw_over_k2,delta_sm_over_k2")
    for i in 1:length(k_grid)
        @printf(io, "%.6f,%.6e,%.6e\n", k_grid[i], delta_k_raw[i]/(k_grid[i]^2), delta_k_sm[i]/(k_grid[i]^2))
    end
end

# Write Test 02 CSV
open("Test_02_ck.csv", "w") do io
    println(io, "k,c_AA_raw,c_AA_sm")
    for i in 1:length(k_grid)
        @printf(io, "%.6f,%.6e,%.6e\n", k_grid[i], c_AA_raw[i], c_AA_sm[i])
    end
end

println("Test Complete! Check the .png and .csv files.")