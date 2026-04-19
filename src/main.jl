# ==========================================
# main.jl : The Iterative Two-Molecule RISM Orchestrator
# ==========================================
using Random, Statistics, LinearAlgebra, Printf, Base.Threads, StaticArrays, Dates, Plots, DelimitedFiles

# ==========================================
# 1. Global Physical & Grid Parameters
# ==========================================
const N_grid = 2048
const dr = 0.1
const r_grid = [(i - 0.5) * dr for i in 1:N_grid]
const r_max = N_grid * dr
const k_grid = range(0.01, 1.5, length=300)

const N_blocks = 20
const N_steps = 50000
const sigma = 3.315
const epsilon = 65.0
const r_cut = (2.0^(1.0/6.0)) * sigma 
const V_shift = 4.0 * epsilon * ((sigma/r_cut)^12 - (sigma/r_cut)^6)
const bond_length = 1.1     
const half_bond = bond_length / 2.0
const T = 65.0              
const kB = 0.0019872041     
const beta = 1.0 / (kB * T)
const rho = 0.01851  

# Solver Tolerances & Diagnostics
const max_iterations = 2       # Lowered for testing
const convergence_tol = 1e-5
const save_step = 1            # Frequency of intermediate saves

# ==========================================
# 2. Include Worker Modules
# ==========================================
include("n2_dimer_sampling.jl")
include("gp_smoothing.jl")
include("RISM_solver.jl")
include("mdiis.jl")

function run_solver()
    # --- Output Directory Management ---
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    base_dir = joinpath("output", timestamp)
    dir_W = joinpath(base_dir, "potentials")
    dir_hraw = joinpath(base_dir, "h_raw")
    dir_hsmooth = joinpath(base_dir, "h_smooth")
    dir_ck = joinpath(base_dir, "c_k")
    dir_plots = joinpath(base_dir, "plots")
    
    mkpath(dir_W); mkpath(dir_hraw); mkpath(dir_hsmooth); mkpath(dir_ck); mkpath(dir_plots)

    println("==================================================")
    println(" INITIALIZING TWO-MOLECULE RISM SOLVER")
    println(" Diagnostic Output will be saved to: ", base_dir)
    println("==================================================")

    W_AA = zeros(Float64, N_grid); W_BB = zeros(Float64, N_grid); W_AB = zeros(Float64, N_grid)
    mdiis_history = initialize_mdiis_buffers() 
    error_log = Float64[]

    # Open error log file
    err_file = open(joinpath(base_dir, "error_tracking.txt"), "w")
    println(err_file, "Iteration  RMSE_Error")

    # To store final arrays for plotting
    final_h_raw_AA = zeros(N_grid); final_h_smooth_AA = zeros(N_grid); final_c_hat_AA = zeros(length(k_grid))

    # ==========================================
    # 3. The Self-Consistent Iteration Loop
    # ==========================================
    for iter in 1:max_iterations
        @printf("\n--- ITERATION %d ---\n", iter)

        println("1. Running Monte Carlo Sampler...")
        raw_h_AA, raw_var_AA, raw_h_BB, raw_var_BB, raw_h_AB, raw_var_AB = run_mc_sampling(N_steps, N_blocks, W_AA, W_BB, W_AB)

        println("2. Constraining and Smoothing via GP...")
        h_AA_smooth, h_BB_smooth, h_AB_smooth = constrain_and_smooth_gp(raw_h_AA, raw_var_AA, raw_h_BB, raw_var_BB, raw_h_AB, raw_var_AB, r_grid, dr)

        println("3. Solving RISM Equations...")
        W_AA_new, W_BB_new, W_AB_new, c_hat_AA, c_hat_BB, c_hat_AB = solve_rism_equations(h_AA_smooth, h_BB_smooth, h_AB_smooth, r_grid, dr)

        # Convergence Check
        residual_AA = W_AA_new .- W_AA; residual_BB = W_BB_new .- W_BB; residual_AB = W_AB_new .- W_AB
        err = sqrt(mean(residual_AA.^2) + mean(residual_BB.^2) + mean(residual_AB.^2))
        push!(error_log, err)
        @printf("   -> Current Solver Error: %.2e\n", err)
        @printf(err_file, "%d  %.6e\n", iter, err)
        flush(err_file) # Ensure it writes immediately in case of crash

        # Diagnostic Intermediate Saves
        if iter % save_step == 0 || iter == max_iterations
            open(joinpath(dir_W, "W_iter_$iter.dat"), "w") do io
                println(io, "r_Angstroms  W_AA  W_BB  W_AB")
                for i in 1:N_grid @printf(io, "%.4f  %.6e  %.6e  %.6e\n", r_grid[i], W_AA_new[i], W_BB_new[i], W_AB_new[i]) end
            end
            open(joinpath(dir_hraw, "h_raw_iter_$iter.dat"), "w") do io
                println(io, "r_Angstroms  h_raw_AA  h_raw_BB  h_raw_AB")
                for i in 1:N_grid @printf(io, "%.4f  %.6e  %.6e  %.6e\n", r_grid[i], raw_h_AA[i], raw_h_BB[i], raw_h_AB[i]) end
            end
            open(joinpath(dir_hsmooth, "h_smooth_iter_$iter.dat"), "w") do io
                println(io, "r_Angstroms  h_smooth_AA  h_smooth_BB  h_smooth_AB")
                for i in 1:N_grid @printf(io, "%.4f  %.6e  %.6e  %.6e\n", r_grid[i], h_AA_smooth[i], h_BB_smooth[i], h_AB_smooth[i]) end
            end
            open(joinpath(dir_ck, "c_k_iter_$iter.dat"), "w") do io
                println(io, "k_inv_Angstroms  c_hat_AA  c_hat_BB  c_hat_AB")
                for i in 1:length(k_grid) @printf(io, "%.4f  %.6e  %.6e  %.6e\n", k_grid[i], c_hat_AA[i], c_hat_BB[i], c_hat_AB[i]) end
            end
        end

        # Capture final states for the summary plots
        final_h_raw_AA .= raw_h_AA; final_h_smooth_AA .= h_AA_smooth; final_c_hat_AA .= c_hat_AA
        W_AA .= W_AA_new; W_BB .= W_BB_new; W_AB .= W_AB_new

        if err < convergence_tol
            println("\n*** SOLVER CONVERGED IN $iter ITERATIONS ***")
            break
        end

        println("4. Mixing potentials via MDIIS...")
        W_AA, W_BB, W_AB = update_mdiis!(mdiis_history, W_AA_new, W_BB_new, W_AB_new, residual_AA, residual_BB, residual_AB)
    end
    close(err_file)
    
    # ==========================================
    # 4. Generate Final Diagnostic Plots
    # ==========================================
    println("==================================================")
    println(" GENERATING FINAL PLOTS IN: ", dir_plots)
    println("==================================================")
    
    # 1. Error Convergence Plot
    p_err = plot(1:length(error_log), error_log, yaxis=:log, marker=:circle, label="RMSE", 
                 xlabel="Iteration", ylabel="Log(Error)", title="Solver Convergence")
    savefig(p_err, joinpath(dir_plots, "01_error_convergence.png"))

    # 2. Final W(r) Plot (Comparing AA, BB, AB to check symmetry)
    p_W = plot(r_grid, W_AA, label="W_AA", xlabel="r (Angstroms)", ylabel="W(r) / kT", linewidth=2, xlims=(0, 15), title="Converged Potentials")
    plot!(p_W, r_grid, W_BB, label="W_BB", linewidth=2, linestyle=:dot)
    plot!(p_W, r_grid, W_AB, label="W_AB", linewidth=2, linestyle=:dash)
    savefig(p_W, joinpath(dir_plots, "02_final_W_potentials.png"))

    # 3. Raw vs Smooth h(r) Plot (Zoomed in on the core)
    p_h = plot(r_grid, final_h_raw_AA, label="Raw MC h_AA", xlabel="r (Angstroms)", ylabel="h(r)", 
               linewidth=1, alpha=0.6, xlims=(0, 15), title="GP Smoothing Validation")
    plot!(p_h, r_grid, final_h_smooth_AA, label="Smooth h_AA", linewidth=2, color=:red)
    savefig(p_h, joinpath(dir_plots, "03_h_r_smoothing.png"))

    # 4. Low-k c(k) behavior
    p_ck = plot(k_grid, final_c_hat_AA, label="c_hat_AA", xlabel="k (1/Angstroms)", ylabel="c(k)", 
                linewidth=2, title="Direct Correlation c(k)")
    savefig(p_ck, joinpath(dir_plots, "04_c_hat_k.png"))

    println("Done! Ready for analysis.")
    return W_AA, W_BB, W_AB
end

converged_W_AA, converged_W_BB, converged_W_AB = run_solver()