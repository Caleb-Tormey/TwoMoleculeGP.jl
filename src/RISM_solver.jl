# ==========================================
# RISM_solver.jl
# Solves the Molecular Ornstein-Zernike equation
# ==========================================
using LinearAlgebra

function solve_rism_equations(h_AA::Vector{Float64}, h_BB::Vector{Float64}, h_AB::Vector{Float64}, 
                              r_grid::Vector{Float64}, dr::Float64)
    
    N_grid = length(r_grid)
    N_k = length(k_grid)
    dk = k_grid[2] - k_grid[1]

    # Pre-allocate k-space arrays
    h_hat_AA = zeros(Float64, N_k); c_hat_AA = zeros(Float64, N_k)
    h_hat_BB = zeros(Float64, N_k); c_hat_BB = zeros(Float64, N_k)
    h_hat_AB = zeros(Float64, N_k); c_hat_AB = zeros(Float64, N_k)

    # ---------------------------------------------------------
    # 1. Forward Fourier Transform (Spherical Bessel, l=0)
    # ---------------------------------------------------------
    for (idx, k) in enumerate(k_grid)
        sum_AA = 0.0; sum_BB = 0.0; sum_AB = 0.0
        for i in 1:N_grid
            r = r_grid[i]
            integrand_kernel = r * sin(k * r) * dr
            sum_AA += h_AA[i] * integrand_kernel
            sum_BB += h_BB[i] * integrand_kernel
            sum_AB += h_AB[i] * integrand_kernel
        end
        
        prefactor = 4.0 * pi / k
        h_hat_AA[idx] = prefactor * sum_AA
        h_hat_BB[idx] = prefactor * sum_BB
        h_hat_AB[idx] = prefactor * sum_AB
    end

    # ---------------------------------------------------------
    # 2. Solve RISM Matrix Equation in k-space (Decoupled Basis)
    # Bypasses the singular matrix inversion at low-k
    # ---------------------------------------------------------
    for (idx, k) in enumerate(k_grid)
        w_cross = sin(k * bond_length) / (k * bond_length)
        w_plus = 1.0 + w_cross
        w_minus = 1.0 - w_cross

        # Enforce exact symmetry for the homonuclear math
        h_sym = (h_hat_AA[idx] + h_hat_BB[idx]) / 2.0
        
        # Transform to diagonal basis
        h_plus = h_sym + h_hat_AB[idx]
        h_minus = h_sym - h_hat_AB[idx]

        # Solve decoupled scalar RISM equations
        c_plus = h_plus / (w_plus^2 + rho * w_plus * h_plus)
        
        # Safe evaluation for c_minus (denominator goes to 0 as k^4, numerator goes to 0 as k^2)
        denom_minus = w_minus^2 + rho * w_minus * h_minus
        
        # If the GP sum rules hold perfectly, h_minus is pure k^2, making c_minus stable.
        # The small tolerance catches floating point drift at k < 0.05
        if abs(denom_minus) > 1e-12
            c_minus = h_minus / denom_minus
        else
            c_minus = 0.0 
        end

        # Transform back to site-site basis
        c_hat_AA[idx] = (c_plus + c_minus) / 2.0
        c_hat_BB[idx] = c_hat_AA[idx]
        c_hat_AB[idx] = (c_plus - c_minus) / 2.0
    end

    # ---------------------------------------------------------
    # 3. Inverse Fourier Transform
    # ---------------------------------------------------------
    c_AA = zeros(Float64, N_grid)
    c_BB = zeros(Float64, N_grid)
    c_AB = zeros(Float64, N_grid)

    for i in 1:N_grid
        r = r_grid[i]
        sum_AA = 0.0; sum_BB = 0.0; sum_AB = 0.0
        for (idx, k) in enumerate(k_grid)
            integrand_kernel = k * sin(k * r) * dk
            sum_AA += c_hat_AA[idx] * integrand_kernel
            sum_BB += c_hat_BB[idx] * integrand_kernel
            sum_AB += c_hat_AB[idx] * integrand_kernel
        end
        
        prefactor = 1.0 / (2.0 * pi^2 * r)
        c_AA[i] = prefactor * sum_AA
        c_BB[i] = prefactor * sum_BB
        c_AB[i] = prefactor * sum_AB
    end

    # ---------------------------------------------------------
    # 4. HNC Closure Relation to yield Target Solvation Potential
    # W(r) = -kT * [h(r) - c(r)]
    # ---------------------------------------------------------
    W_AA_new = zeros(Float64, N_grid)
    W_BB_new = zeros(Float64, N_grid)
    W_AB_new = zeros(Float64, N_grid)

    for i in 1:N_grid
        # gamma(r) = h(r) - c(r)
        W_AA_new[i] = -(1.0 / beta) * (h_AA[i] - c_AA[i])
        W_BB_new[i] = -(1.0 / beta) * (h_BB[i] - c_BB[i])
        W_AB_new[i] = -(1.0 / beta) * (h_AB[i] - c_AB[i])
    end

    return W_AA_new, W_BB_new, W_AB_new, c_hat_AA, c_hat_BB, c_hat_AB
end