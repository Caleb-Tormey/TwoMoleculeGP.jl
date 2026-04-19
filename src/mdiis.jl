# ==========================================
# mdiis.jl
# Modified Direct Inversion in the Iterative Subspace
# ==========================================
using LinearAlgebra

# Struct to hold the history of inputs and errors
mutable struct MDIIS_History
    depth::Int
    current_size::Int
    
    # We will stack [W_AA, W_BB, W_AB] into single 1D vectors for easy linear algebra
    W_history::Vector{Vector{Float64}}
    Err_history::Vector{Vector{Float64}}
    
    # Picard mixing fraction to apply to the optimal MDIIS solution (0.0 to 1.0)
    # Lower is safer but slower. 0.5 is a standard, robust starting point.
    alpha::Float64 
end

function initialize_mdiis_buffers(depth::Int=5, alpha::Float64=0.5)
    return MDIIS_History(depth, 0, Vector{Vector{Float64}}(), Vector{Vector{Float64}}(), alpha)
end

function update_mdiis!(history::MDIIS_History, 
                       W_AA_target::Vector{Float64}, W_BB_target::Vector{Float64}, W_AB_target::Vector{Float64},
                       residual_AA::Vector{Float64}, residual_BB::Vector{Float64}, residual_AB::Vector{Float64})
    
    N_grid = length(W_AA_target)
    
    # 1. Stack the current state and error into unified 1D vectors
    # Note: For MDIIS, the "current input" W(r) is exactly what we just fed into the MC.
    # W_target is what the RISM equation produced. 
    # Therefore, W_current = W_target - residual.
    W_current = vcat(W_AA_target .- residual_AA, 
                     W_BB_target .- residual_BB, 
                     W_AB_target .- residual_AB)
                     
    Err_current = vcat(residual_AA, residual_BB, residual_AB)

    # 2. Manage the History Buffers
    push!(history.W_history, W_current)
    push!(history.Err_history, Err_current)

    if length(history.W_history) > history.depth
        popfirst!(history.W_history)
        popfirst!(history.Err_history)
    end
    
    history.current_size = length(history.W_history)
    n = history.current_size

    # 3. Handle the first few steps (Standard Picard Mixing)
    if n < 2
        println("   [MDIIS] Building history (using Picard mixing, alpha=$(history.alpha))")
        W_next = W_current .+ history.alpha .* Err_current
        
        return W_next[1:N_grid], W_next[N_grid+1:2*N_grid], W_next[2*N_grid+1:3*N_grid]
    end

    # 4. Construct the MDIIS Gram Matrix (A) and Target Vector (b)
    # We solve: A * c = b
    # Where A is the dot product of error vectors, plus a row/col of -1 for the Lagrange multiplier
    A = zeros(Float64, n + 1, n + 1)
    b = zeros(Float64, n + 1)
    b[n + 1] = -1.0

    for i in 1:n
        for j in 1:n
            A[i, j] = dot(history.Err_history[i], history.Err_history[j])
        end
        A[i, n + 1] = -1.0
        A[n + 1, i] = -1.0
    end
    A[n + 1, n + 1] = 0.0

    # 5. Solve for the coefficients (c)
    # Wrap in a try-catch block to gracefully fallback to Picard if the matrix is singular
    coeffs = zeros(Float64, n)
    try
        x = A \ b
        coeffs = x[1:n]
    catch e
        println("   [MDIIS WARNING] Matrix inversion failed. Falling back to Picard.")
        W_next = W_current .+ history.alpha .* Err_current
        return W_next[1:N_grid], W_next[N_grid+1:2*N_grid], W_next[2*N_grid+1:3*N_grid]
    end

    # 6. Construct the Optimal Predicted Vectors
    W_opt = zeros(Float64, 3 * N_grid)
    Err_opt = zeros(Float64, 3 * N_grid)

    for i in 1:n
        W_opt .+= coeffs[i] .* history.W_history[i]
        Err_opt .+= coeffs[i] .* history.Err_history[i]
    end

    # 7. Apply Picard Step from the Optimal State
    W_next = W_opt .+ history.alpha .* Err_opt

    # 8. Unstack and return
    println("   [MDIIS] Applied subspace inversion mixing.")
    W_AA_next = W_next[1:N_grid]
    W_BB_next = W_next[N_grid+1:2*N_grid]
    W_AB_next = W_next[2*N_grid+1:3*N_grid]

    return W_AA_next, W_BB_next, W_AB_next
end