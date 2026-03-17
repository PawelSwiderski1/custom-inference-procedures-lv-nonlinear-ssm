using Gen
using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf

include("nonlinear_ssm_custom_inference.jl")

# Small utilities
struct MethodResult
    name::String
    seconds::Float64
    samples::Vector{NamedTuple}          # unweighted posterior samples (after burnin for MCMC)
    weights::Union{Nothing, Vector{Float64}}  # normalized weights if IS, else nothing
    extra::Dict{Symbol,Any}
end

function _now_s()
    return time_ns() / 1e9
end

function weighted_mean(x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    @assert length(x) == length(w)
    return sum(w .* x)
end

function weighted_var(x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    μ = weighted_mean(x, w)
    return sum(w .* (x .- μ).^2)
end

function ess_from_norm_weights(w::AbstractVector{<:Real})
    # w must sum to 1
    return 1.0 / sum(w .^ 2)
end

# ESS using Geyer's initial positive sequence method
function ess_geyer(x::AbstractVector{<:Real}; maxlag::Int=2000)
    n = length(x)
    if n < 3
        return float(n)
    end
    maxlag = min(maxlag, n - 1)

    # compute autocorrelations
    μ = mean(x)
    v = mean((x .- μ).^2)
    if v == 0
        return float(n)
    end

    ρ = Vector{Float64}(undef, maxlag + 1)
    for k in 0:maxlag
        ρ[k + 1] = mean((x[1:(n - k)] .- μ) .* (x[(1 + k):n] .- μ)) / v
    end

    # Geyer's initial positive sequence
    s = 0.0
    m = 1
    while (2 * m) <= maxlag
        pair = ρ[2 * m] + ρ[2 * m + 1]
        if !(pair > 0)
            break
        end
        s += pair
        m += 1
    end

    τ = 1.0 + 2.0 * s
    if !(τ > 0) || isnan(τ) || isinf(τ)
        return 1.0
    end
    return n / τ
end

function ess_geyer_dict(samples::Vector{NamedTuple}; burnin::Int=0, thin::Int=1)
    idx = (burnin+1):thin:length(samples)
    s = samples[idx]
    keys = (:sigma_x, :sigma_y)

    out = Dict{Symbol,Float64}()
    for k in keys
        chain = Float64.(getfield.(s, k))
        out[k] = ess_geyer(chain)
    end
    return out
end

function ess_per_second_summary(samples::Vector{NamedTuple}, seconds::Float64; burnin::Int=0, thin::Int=1)
    essd = ess_geyer_dict(samples; burnin=burnin, thin=thin)
    ess_vals = [essd[k] for k in (:sigma_x, :sigma_y)]

    min_ess = minimum(ess_vals)
    med_ess = median(ess_vals)
    max_ess = maximum(ess_vals)

    return (
        min_ess = min_ess,
        med_ess = med_ess,
        max_ess = max_ess,
        min_ess_per_sec = min_ess / seconds,
        med_ess_per_sec = med_ess / seconds,
        max_ess_per_sec = max_ess / seconds,
    )
end

function summarize_chain(samples::Vector{NamedTuple}; burnin::Int=0, thin::Int=1)
    if isempty(samples)
        return Dict{Symbol,NamedTuple}()
    end

    start = burnin + 1
    if start > length(samples)
        return Dict{Symbol,NamedTuple}()
    end

    idx = start:thin:length(samples)
    s = samples[idx]
    if isempty(s)
        return Dict{Symbol,NamedTuple}()
    end

    keys = (:sigma_x, :sigma_y)
    out = Dict{Symbol,NamedTuple}()

    for k in keys
        vals = Float64.(getfield.(s, k))
        out[k] = (mean=mean(vals), std=std(vals), min=minimum(vals), max=maximum(vals), n=length(vals))
    end
    return out
end

function summarize_is(traces, log_norm_w)
    w = exp.(log_norm_w)
    keys = (:sigma_x, :sigma_y)

    vals = Dict{Symbol,Vector{Float64}}()
    for k in keys
        vals[k] = [Float64(tr[k]) for tr in traces]
    end

    out = Dict{Symbol,NamedTuple}()
    for k in keys
        x = vals[k]
        μ = weighted_mean(x, w)
        v = weighted_var(x, w)
        out[k] = (mean=μ, std=sqrt(max(v, 0.0)), min=minimum(x), max=maximum(x), n=length(x))
    end
    return out, w
end

function print_summary_table(title::String, summ::Dict{Symbol,NamedTuple})
    println("\n", title)
    for k in (:sigma_x, :sigma_y)
        st = summ[k]
        @printf("  %-5s  mean=% .6f  std=% .6f  min=% .6f  max=% .6f  (n=%d)\n",
                String(k), st.mean, st.std, st.min, st.max, st.n)
    end
end

function print_ess_table(title::String, essd::Dict{Symbol,Float64}, n_eff_base::Int)
    println("\n", title)
    for k in (:sigma_x, :sigma_y)
        e = essd[k]
        @printf("  %-5s  ESS≈%8.1f  ESS/n=% .4f\n", String(k), e, e / n_eff_base)
    end
end

# Log-parameterized model for HMC
@gen function nl_ssm_model_hmc(T::Int)
    log_sigma_x = {:log_sigma_x} ~ normal(log(1.0), 0.3)
    log_sigma_y = {:log_sigma_y} ~ normal(log(1.0), 0.3)

    sigma_x = exp(log_sigma_x)
    sigma_y = exp(log_sigma_y)

    x = Vector{Real}(undef, T)

    x[1] = {:x => 1} ~ normal(0.0, 2.0)
    {:y => 1} ~ normal(g_observe(x[1]), sigma_y)

    for t in 2:T
        μ = f_transition(x[t-1], t)
        x[t] = {:x => t} ~ normal(μ, sigma_x)
        {:y => t} ~ normal(g_observe(x[t]), sigma_y)
    end

    return x
end

# Helper: make observation constraints
function make_obs_constraints(y_obs::Vector{Float64})
    obs = choicemap()
    for t in 1:length(y_obs)
        obs[:y => t] = y_obs[t]
    end
    return obs
end

# Gen Importance Sampling
function run_gen_importance_sampling(y_obs::Vector{Float64}, T::Int;
    n_samples::Int=2000,
    rng::AbstractRNG=Random.default_rng()
)
    obs = make_obs_constraints(y_obs)
    t0 = _now_s()
    traces, log_norm_w, lml_est = Gen.importance_sampling(nl_ssm_model, (T,), obs, n_samples)
    secs = _now_s() - t0

    summ, w = summarize_is(traces, log_norm_w)
    ess_w = ess_from_norm_weights(w)

    res = MethodResult(
        "Gen Importance Sampling",
        secs,
        NamedTuple[],
        w,
        Dict(:lml_est => lml_est, :weighted_ess => ess_w, :summary => summ)
    )
    return res
end

# Gen Metropolis-Hastings (selection-based)
function run_gen_mh(y_obs::Vector{Float64}, T::Int;
    n_iters::Int=3000,
    burnin::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
    mh_sweeps_per_iter::Int=1
)
    obs = make_obs_constraints(y_obs)

    # Initialize trace
    tr, _ = generate(nl_ssm_model, (T,), obs)

    sel_sigma = select(:sigma_x, :sigma_y)

    # For latent states, we'll update them in blocks
    block_size = max(1, T ÷ 10)

    samples = Vector{NamedTuple}(undef, n_iters)
    acc = Dict(:sigma=>0, :states=>0)
    t0 = _now_s()

    for it in 1:n_iters
        for _ in 1:mh_sweeps_per_iter
            # Update sigma parameters
            tr, ok = Gen.mh(tr, sel_sigma; observations=obs)
            acc[:sigma] += ok ? 1 : 0

            # Update latent states in blocks
            for start_t in 1:block_size:T
                end_t = min(start_t + block_size - 1, T)
                sel_block = select([(:x => t) for t in start_t:end_t]...)
                tr, ok = Gen.mh(tr, sel_block; observations=obs)
                acc[:states] += ok ? 1 : 0
            end
        end

        samples[it] = (
            sigma_x = tr[:sigma_x],
            sigma_y = tr[:sigma_y],
            score = Gen.get_score(tr)
        )

        if it % 500 == 0
            @printf("MH it=%d acc_sigma=%.3f score=%.2f\n",
                    it, acc[:sigma]/(it*mh_sweeps_per_iter), samples[it].score)
        end
    end

    secs = _now_s() - t0
    n_blocks = ceil(Int, T / block_size)
    extra = Dict(
        :acceptance_rates => Dict(
            :sigma => acc[:sigma] / (n_iters*mh_sweeps_per_iter),
            :states => acc[:states] / (n_iters*mh_sweeps_per_iter*n_blocks)
        )
    )

    return MethodResult("Gen MH (selection)", secs, samples, nothing, extra)
end

# Gen HMC (on log-parameterized model)
function run_gen_hmc(y_obs::Vector{Float64}, T::Int;
    n_iters::Int=2000,
    burnin::Int=500,
    rng::AbstractRNG=Random.default_rng(),
    L::Int=10,
    eps::Float64=0.01,
    hmc_sweeps_per_iter::Int=1
)
    obs = make_obs_constraints(y_obs)

    # Initialize trace with HMC model
    tr, _ = generate(nl_ssm_model_hmc, (T,), obs)

    # Select all continuous variables for HMC
    sel_all = select(:log_sigma_x, :log_sigma_y, [(:x => t) for t in 1:T]...)

    samples = Vector{NamedTuple}(undef, n_iters)
    acc = 0
    t0 = _now_s()

    for it in 1:n_iters
        for _ in 1:hmc_sweeps_per_iter
            tr, ok = Gen.hmc(tr, sel_all; L=L, eps=eps, observations=obs, check=false)
            acc += ok ? 1 : 0
        end

        samples[it] = (
            sigma_x = exp(tr[:log_sigma_x]),
            sigma_y = exp(tr[:log_sigma_y]),
            score = Gen.get_score(tr)
        )

        if it % 200 == 0
            @printf("HMC it=%d acc_rate=%.3f score=%.2f\n",
                    it, acc/(it*hmc_sweeps_per_iter), samples[it].score)
        end
    end

    secs = _now_s() - t0
    extra = Dict(:acceptance_rate => acc / (n_iters*hmc_sweeps_per_iter), :L => L, :eps => eps)
    return MethodResult("Gen HMC (log-param)", secs, samples, nothing, extra)
end

# 4) Custom PGAS+MH wrapper
function run_custom_pgas_wrapper(y_obs::Vector{Float64}, T::Int;
    n_iters::Int=3000,
    burnin::Int=1000,
    N::Int=64,
    thin::Int=1,
    step_x::Float64=0.05,
    step_y::Float64=0.05,
    rng::AbstractRNG=Random.default_rng()
)
    t0 = _now_s()

    sigma_x = rand(rng, SIGMA_PRIOR)
    sigma_y = rand(rng, SIGMA_PRIOR)
    x_ref = [rand(rng, Normal(0.0, 2.0)) for _ in 1:T]

    # Storage for all samples (including burnin, we'll filter later)
    all_samples = Vector{NamedTuple}(undef, n_iters)

    accx_total = 0
    accy_total = 0

    for it in 1:n_iters
        # PGAS update of x
        x_ref = pgas_step(rng, y_obs, sigma_x, sigma_y, x_ref; N=N)

        # MH update of σ
        sigma_x, accx = mh_update_sigma_x(rng, x_ref, y_obs, sigma_x, sigma_y; step_x=step_x)
        sigma_y, accy = mh_update_sigma_y(rng, x_ref, y_obs, sigma_x, sigma_y; step_y=step_y)
        accx_total += accx ? 1 : 0
        accy_total += accy ? 1 : 0

        all_samples[it] = (
            sigma_x = sigma_x,
            sigma_y = sigma_y,
            score = log_joint(x_ref, y_obs, sigma_x, sigma_y)
        )

        if it % 500 == 0
            @printf("PGAS it=%d sigma_x=%.3f sigma_y=%.3f acc_x=%.3f acc_y=%.3f\n",
                    it, sigma_x, sigma_y, accx_total/it, accy_total/it)
        end
    end

    secs = _now_s() - t0

    extra = Dict(
        :acceptance_rates => Dict(
            :sigma_x => accx_total / n_iters,
            :sigma_y => accy_total / n_iters
        ),
        :N_particles => N
    )

    return MethodResult("Custom PGAS+MH", secs, all_samples, nothing, extra)
end

# Unified comparison runner
function compare_all_inference_methods(y_obs::Vector{Float64}, T::Int;
    # budgets
    n_iters::Int=3000,
    burnin::Int=1000,
    is_samples::Int=2000,

    # HMC params (tuned)
    hmc_L::Int=10,
    hmc_eps::Float64=0.01,

    # PGAS params
    pgas_N::Int=64,
    pgas_step_x::Float64=0.05,
    pgas_step_y::Float64=0.05,

    # sweeps
    mh_sweeps::Int=1,
    hmc_sweeps::Int=1,

    rng::AbstractRNG=Random.default_rng()
)
    @assert length(y_obs) == T

    results = MethodResult[]

    # 0) Custom PGAS+MH
    println("\nRunning Custom PGAS+MH...")
    push!(results, run_custom_pgas_wrapper(
        y_obs, T;
        n_iters=n_iters, burnin=burnin, N=pgas_N,
        step_x=pgas_step_x, step_y=pgas_step_y, rng=rng
    ))

    # 1) IS
    println("\nRunning Gen Importance Sampling...")
    push!(results, run_gen_importance_sampling(
        y_obs, T;
        n_samples=is_samples, rng=rng
    ))

    # 2) MH
    println("\nRunning Gen MH...")
    push!(results, run_gen_mh(
        y_obs, T;
        n_iters=n_iters, burnin=burnin,
        mh_sweeps_per_iter=mh_sweeps, rng=rng
    ))

    # 3) HMC
    println("\nRunning Gen HMC...")
    push!(results, run_gen_hmc(
        y_obs, T;
        n_iters=n_iters, burnin=burnin,
        L=hmc_L, eps=hmc_eps, hmc_sweeps_per_iter=hmc_sweeps, rng=rng
    ))

    # print comparison
    println("\n" * "="^50)
    println("Inference comparison (Nonlinear SSM)")
    println("="^50)
    println("Budget: n_iters=$n_iters, burnin=$burnin | IS samples=$is_samples")
    println("HMC: L=$hmc_L eps=$hmc_eps | PGAS: N=$pgas_N")

    for r in results
        println("\n" * "-"^40)
        println(r.name)
        @printf("Wall time: %.3f s\n", r.seconds)

        if r.name == "Gen Importance Sampling"
            summ = r.extra[:summary]
            wess = r.extra[:weighted_ess]
            lml  = r.extra[:lml_est]
            print_summary_table("Posterior (weighted) summary:", summ)
            @printf("Weighted ESS (from weights): %.1f / %d\n", wess, is_samples)
            println("log marginal likelihood estimate (lml_est): ", lml)
            @printf("IS weighted ESS/sec: %.3f\n", r.extra[:weighted_ess] / r.seconds)

        else
            epss = ess_per_second_summary(r.samples, r.seconds; burnin=burnin, thin=1)
            @printf("ESS/sec: min=%.3f  median=%.3f  max=%.3f\n",
                    epss.min_ess_per_sec, epss.med_ess_per_sec, epss.max_ess_per_sec)

            summ = summarize_chain(r.samples; burnin=burnin, thin=1)
            print_summary_table("Posterior summary:", summ)

            essd = ess_geyer_dict(r.samples; burnin=burnin, thin=1)
            n_eff_base = length((burnin+1):length(r.samples))
            print_ess_table("ESS (Geyer) by parameter:", essd, n_eff_base)

            if haskey(r.extra, :acceptance_rates)
                println("Acceptance rates: ", r.extra[:acceptance_rates])
            elseif haskey(r.extra, :acceptance_rate)
                println("Acceptance rate: ", r.extra[:acceptance_rate], " (L=", r.extra[:L], ", eps=", r.extra[:eps], ")")
            end
        end
    end

    return results
end

# Diagnostics
function nl_ssm_diagnostics(samples::Vector{NamedTuple}; burnin::Int=0, thin::Int=1)
    @assert thin >= 1
    @assert burnin >= 0
    idx = (burnin + 1):thin:length(samples)
    s2 = samples[idx]

    println("\n=== Nonlinear SSM ACF + ESS diagnostics ===")
    println("Using ", length(s2), " samples (burnin=", burnin, ", thin=", thin, ")")

    latents = [:sigma_x, :sigma_y]
    ess_vals = Float64[]

    for key in latents
        chain = Float64.(getfield.(s2, key))

        n = length(chain)
        e = ess_geyer(chain)

        println("\n", String(key), ":")
        println("  n = ", n)
        println("  mean = ", mean(chain), "   std = ", std(chain))
        println("  ESS ≈ ", round(e, digits=1), "   ESS/n = ", round(e/n, digits=3))

        push!(ess_vals, e)
    end

    println("\nESS summary:")
    println("  min    = ", minimum(ess_vals))
    println("  median = ", median(ess_vals))
    println("  max    = ", maximum(ess_vals))
end

# Main execution

rng = MersenneTwister(42)

T = 200

# Simulate ground truth data
true_trace = simulate(nl_ssm_model, (T,))
x_true = get_retval(true_trace)
y_obs = [true_trace[:y => t] for t in 1:T]

println("\n=== Ground truth parameters (comparison dataset) ===")
println("sigma_x = ", true_trace[:sigma_x])
println("sigma_y = ", true_trace[:sigma_y])
println("T  = ", T)

# Run comparison
results = compare_all_inference_methods(
    y_obs, T;
    n_iters=3000,
    burnin=1000,
    is_samples=2000,
    hmc_L=5,
    hmc_eps=0.025,
    pgas_N=64,
    pgas_step_x=0.08,
    pgas_step_y=0.08,
    mh_sweeps=1,
    hmc_sweeps=1,
    rng=rng
)

# Detailed diagnostics on each MCMC method
println("\n" * "="^50)
println("Detailed diagnostics")
println("="^50)

nl_ssm_diagnostics(results[1].samples; burnin=1000, thin=1)  # Custom PGAS
nl_ssm_diagnostics(results[3].samples; burnin=1000, thin=1)  # MH
nl_ssm_diagnostics(results[4].samples; burnin=1000, thin=1)  # HMC

