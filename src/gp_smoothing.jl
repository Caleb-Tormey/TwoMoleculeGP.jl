# ==========================================
# gp_smoothing.jl
# ==========================================
using LinearAlgebra, AbstractGPs, KernelFunctions, Printf

function constrain_and_smooth_gp(h_AA_mean::Vector{Float64}, h_AA_var::Vector{Float64},
                                 h_BB_mean::Vector{Float64}, h_BB_var::Vector{Float64},
                                 h_AB_mean::Vector{Float64}, h_AB_var::Vector{Float64},
                                 r_grid::Vector{Float64}, dr::Float64)
    
    N_grid = length(r_grid)
    
    y_stack = vcat(h_AA_mean, h_BB_mean, h_AB_mean)
    v_stack = vcat(h_AA_var .+ 1e-8, h_BB_var .+ 1e-8, h_AB_var .+ 1e-8)

    l_scale = 0.5 
    sig_var = 1.0 
    kernel = SqExponentialKernel() ∘ ScaleTransform(1.0 / l_scale)
    
    K_base = sig_var .* kernelmatrix(kernel, r_grid)
    K_joint = zeros(Float64, 3*N_grid, 3*N_grid)
    K_joint[1:N_grid, 1:N_grid]                       = K_base
    K_joint[N_grid+1:2*N_grid, N_grid+1:2*N_grid]     = K_base
    K_joint[2*N_grid+1:3*N_grid, 2*N_grid+1:3*N_grid] = K_base

    # =======================================================
    # 2. YOUR FIX: Restrict constraint weights to the physical core
    # =======================================================
    C_mat = zeros(Float64, 3, 3*N_grid)
    r_constraint_max = 40.0 
    
    for i in 1:N_grid
        if r_grid[i] <= r_constraint_max
            w2_i = (r_grid[i]^2) * dr
            w4_i = (r_grid[i]^4) * dr
        else
            w2_i = 0.0
            w4_i = 0.0
        end
        
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

    # Normalization
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

    h_AA_smooth = mu_post[1:N_grid]
    h_BB_smooth = mu_post[N_grid+1:2*N_grid]
    h_AB_smooth = mu_post[2*N_grid+1:3*N_grid]

    # ==========================================
    # 3. VERIFICATION BLOCK (Also restricted to r < 40)
    # ==========================================
    chk_AA_AB_O1 = zeros(Float64, 3*N_grid)
    chk_BB_AB_O1 = zeros(Float64, 3*N_grid)
    chk_Det_O1   = zeros(Float64, 3*N_grid)
    chk_Det_Ok2  = zeros(Float64, 3*N_grid)

    for i in 1:N_grid
        if r_grid[i] <= r_constraint_max
            w2 = (r_grid[i]^2)*dr
            w4 = (r_grid[i]^4)*dr
        else
            w2 = 0.0
            w4 = 0.0
        end
        
        chk_AA_AB_O1[i] = w2; chk_AA_AB_O1[2*N_grid + i] = -w2
        chk_BB_AB_O1[N_grid + i] = w2; chk_BB_AB_O1[2*N_grid + i] = -w2
        
        chk_Det_O1[i] = w2; chk_Det_O1[N_grid + i] = w2; chk_Det_O1[2*N_grid + i] = -2.0*w2
        chk_Det_Ok2[i] = w4; chk_Det_Ok2[N_grid + i] = w4; chk_Det_Ok2[2*N_grid + i] = -2.0*w4
    end

    smooth_stack = vcat(h_AA_smooth, h_BB_smooth, h_AB_smooth)

    println("\n--- GP SUM RULE VERIFICATION (r <= 40A) ---")
    @printf("O(1) [AA - AB]      Raw: % .4e | Smooth: % .4e\n", sum(chk_AA_AB_O1 .* y_stack), sum(chk_AA_AB_O1 .* smooth_stack))
    @printf("O(1) [BB - AB]      Raw: % .4e | Smooth: % .4e\n", sum(chk_BB_AB_O1 .* y_stack), sum(chk_BB_AB_O1 .* smooth_stack))
    @printf("O(1) [AA+BB-2AB]    Raw: % .4e | Smooth: % .4e\n", sum(chk_Det_O1 .* y_stack),   sum(chk_Det_O1 .* smooth_stack))
    @printf("O(k^2) [AA+BB-2AB]  Raw: % .4e | Smooth: % .4e\n", sum(chk_Det_Ok2 .* y_stack),  sum(chk_Det_Ok2 .* smooth_stack))
    println("-------------------------------------------\n")

    return h_AA_smooth, h_BB_smooth, h_AB_smooth
end