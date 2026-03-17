using Gen
using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf

include("lv_custom_inference.jl")

# Small utilities
@gen function lv_model_hmc(T::Int, dt::Float64)
    # unconstrained
    log_alpha = {:log_alpha} ~ normal(0.0, 0.3)
    log_beta  = {:log_beta}  ~ normal(-2.0, 0.3)
    log_gamma = {:log_gamma} ~ normal(0.0, 0.3)
    log_delta = {:log_delta} ~ normal(-2.0, 0.3)

    log_sigma = {:log_sigma} ~ normal(log(0.15), 0.3)

    log_x0 = {:log_x0} ~ normal(log(10.0), 0.2)
    log_y0 = {:log_y0} ~ normal(log(5.0), 0.2)

    # transform to positive
    alpha = exp(log_alpha)
    beta  = exp(log_beta)
    gamma = exp(log_gamma)
    delta = exp(log_delta)
    sigma = exp(log_sigma)
    x0    = exp(log_x0)
    y0    = exp(log_y0)

    xs, ys = simulate_lv(alpha, beta, gamma, delta, x0, y0, T; dt=dt)

    for t in 1:T
        {:y => t => 1} ~ log_normal(log(xs[t]), sigma)
        {:y => t => 2} ~ log_normal(log(ys[t]), sigma)
    end

    return (xs, ys)
end


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

function ess_per_second_summary(samples::Vector{NamedTuple}, seconds::Float64; burnin::Int=0, thin::Int=1)
    essd = ess_geyer_dict(samples; burnin=burnin, thin=thin)
    ess_vals = [essd[k] for k in (:alpha,:beta,:gamma,:delta,:x0,:y0,:sigma)]

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

    keys = (:alpha, :beta, :gamma, :delta, :x0, :y0, :sigma)
    out = Dict{Symbol,NamedTuple}()

    for k in keys
        vals = Float64.(getfield.(s, k))
        out[k] = (mean=mean(vals), std=std(vals), min=minimum(vals), max=maximum(vals), n=length(vals))
    end
    return out
end


function summarize_is(traces, log_norm_w)
    # log_norm_w are normalized log-weights (sum exp(logw) = 1)
    w = exp.(log_norm_w)
    keys = (:alpha, :beta, :gamma, :delta, :x0, :y0, :sigma)

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

function ess_geyer_dict(samples::Vector{NamedTuple}; burnin::Int=0, thin::Int=1)
    idx = (burnin+1):thin:length(samples)
    s = samples[idx]
    keys = (:alpha, :beta, :gamma, :delta, :x0, :y0, :sigma)

    out = Dict{Symbol,Float64}()
    for k in keys
        chain = Float64.(getfield.(s, k))
        out[k] = ess_geyer(chain)
    end
    return out
end

function print_summary_table(title::String, summ::Dict{Symbol,NamedTuple})
    println("\n", title)
    for k in (:alpha,:beta,:gamma,:delta,:x0,:y0,:sigma)
        st = summ[k]
        @printf("  %-5s  mean=% .6f  std=% .6f  min=% .6f  max=% .6f  (n=%d)\n",
                String(k), st.mean, st.std, st.min, st.max, st.n)
    end
end

function print_ess_table(title::String, essd::Dict{Symbol,Float64}, n_eff_base::Int)
    println("\n", title)
    for k in (:alpha,:beta,:gamma,:delta,:x0,:y0,:sigma)
        e = essd[k]
        @printf("  %-5s  ESS≈%8.1f  ESS/n=% .4f\n", String(k), e, e / n_eff_base)
    end
end

# Build init trace for MH
function init_trace_centered_for_mh(xobs, yobs, T, dt; smooth_w::Int=31)
    alpha_hat, beta_hat, gamma_hat, delta_hat =
        estimate_lv_params_from_obs(xobs, yobs, dt; smooth_w=smooth_w)

    x0_center = xobs[1]
    y0_center = yobs[1]
    sigma_center = estimate_sigma_center(xobs, yobs; smooth_w=smooth_w)
    sigma_center = max(sigma_center, 1e-9)

    constraints = make_obs_constraints(xobs, yobs)
    constraints[:alpha] = alpha_hat
    constraints[:beta]  = beta_hat
    constraints[:gamma] = gamma_hat
    constraints[:delta] = delta_hat
    constraints[:x0]    = x0_center
    constraints[:y0]    = y0_center
    constraints[:sigma] = sigma_center

    tr, _ = generate(lv_model, (T, dt), constraints)
    return tr
end

# Build init trace for HMC (unconstrained log-params)
function init_trace_centered_for_hmc(xobs, yobs, T, dt; smooth_w::Int=31)
    alpha_hat, beta_hat, gamma_hat, delta_hat =
        estimate_lv_params_from_obs(xobs, yobs, dt; smooth_w=smooth_w)

    x0_center = xobs[1]
    y0_center = yobs[1]
    sigma_center = max(estimate_sigma_center(xobs, yobs; smooth_w=smooth_w), 1e-9)

    constraints = make_obs_constraints(xobs, yobs)

    # constrain log-parameters
    constraints[:log_alpha] = log(alpha_hat)
    constraints[:log_beta]  = log(beta_hat)
    constraints[:log_gamma] = log(gamma_hat)
    constraints[:log_delta] = log(delta_hat)

    constraints[:log_x0]    = log(x0_center)
    constraints[:log_y0]    = log(y0_center)
    constraints[:log_sigma]= log(sigma_center)

    tr, _ = generate(lv_model_hmc, (T, dt), constraints)
    return tr
end


# Gen Importance Sampling
function run_gen_importance_sampling(xobs, yobs, T, dt;
    n_samples::Int=2000,
    rng::AbstractRNG=Random.default_rng()
)
    obs = make_obs_constraints(xobs, yobs)
    t0 = _now_s()
    traces, log_norm_w, lml_est = Gen.importance_sampling(lv_model, (T, dt), obs, n_samples)
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
function run_gen_mh(xobs, yobs, T, dt;
    n_iters::Int=6000,
    burnin::Int=1500,
    rng::AbstractRNG=Random.default_rng(),
    smooth_w::Int=31,
    mh_sweeps_per_iter::Int=1
)
    obs = make_obs_constraints(xobs, yobs)
    tr = init_trace_centered_for_mh(xobs, yobs, T, dt; smooth_w=smooth_w)

    sel_abgd = select(:alpha, :beta, :gamma, :delta)
    sel_xy   = select(:x0, :y0)
    sel_sig  = select(:sigma)

    samples = Vector{NamedTuple}(undef, n_iters)
    acc = Dict(:abgd=>0, :xy=>0, :sigma=>0)
    t0 = _now_s()

    for it in 1:n_iters
        for _ in 1:mh_sweeps_per_iter
            tr, ok = Gen.mh(tr, sel_abgd; observations=obs)
            acc[:abgd] += ok ? 1 : 0

            tr, ok = Gen.mh(tr, sel_xy; observations=obs)
            acc[:xy] += ok ? 1 : 0

            tr, ok = Gen.mh(tr, sel_sig; observations=obs)
            acc[:sigma] += ok ? 1 : 0
        end

        samples[it] = (
            alpha = tr[:alpha], beta = tr[:beta], gamma = tr[:gamma], delta = tr[:delta],
            x0 = tr[:x0], y0 = tr[:y0], sigma = tr[:sigma],
            score = Gen.get_score(tr)
        )
    end

    secs = _now_s() - t0
    extra = Dict(
        :acceptance_rates => Dict(
            :abgd => acc[:abgd] / (n_iters*mh_sweeps_per_iter),
            :xy   => acc[:xy]   / (n_iters*mh_sweeps_per_iter),
            :sigma=> acc[:sigma]/ (n_iters*mh_sweeps_per_iter)
        )
    )

    return MethodResult("Gen MH (selection)", secs, samples, nothing, extra)
end

# 3) Gen HMC (selection-based)
function run_gen_hmc(xobs, yobs, T, dt;
    n_iters::Int=2000,
    burnin::Int=500,
    rng::AbstractRNG=Random.default_rng(),
    smooth_w::Int=31,
    L::Int=5,
    eps::Float64=0.003,
    hmc_sweeps_per_iter::Int=1
)
    obs = make_obs_constraints(xobs, yobs)
    tr  = init_trace_centered_for_hmc(xobs, yobs, T, dt; smooth_w=smooth_w)

    sel_all = select(
        :log_alpha, :log_beta, :log_gamma, :log_delta,
        :log_x0, :log_y0, :log_sigma
    )

    samples = Vector{NamedTuple}(undef, n_iters)
    acc = 0
    t0 = _now_s()

    for it in 1:n_iters
        for _ in 1:hmc_sweeps_per_iter
            tr, ok = Gen.hmc(tr, sel_all; L=L, eps=eps, observations=obs, check=false)
            acc += ok ? 1 : 0
        end

        samples[it] = (
            alpha = exp(tr[:log_alpha]),
            beta  = exp(tr[:log_beta]),
            gamma = exp(tr[:log_gamma]),
            delta = exp(tr[:log_delta]),
            x0    = exp(tr[:log_x0]),
            y0    = exp(tr[:log_y0]),
            sigma = exp(tr[:log_sigma]),
            score = Gen.get_score(tr)
        )

        if it % 100 == 0
            @printf("HMC it=%d acc_rate=%.3f score=%.2f\n", it, acc/(it*hmc_sweeps_per_iter), samples[it].score)
        end
    end

    secs = _now_s() - t0
    extra = Dict(:acceptance_rate => acc / (n_iters*hmc_sweeps_per_iter), :L => L, :eps => eps)
    return MethodResult("Gen HMC (log-param)", secs, samples, nothing, extra)
end


# Custom Blocked ESS sampler wrapper
function run_custom_blocked_ess_wrapper(xobs, yobs, T, dt;
    n_iters::Int=6000,
    burnin::Int=1500,
    smooth_w::Int=31,
    n_sweeps_per_iter::Int=1,
    rng::AbstractRNG=Random.default_rng()
)
    t0 = _now_s()
    samples = run_lv_blocked_ess(
        xobs, yobs, T, dt;
        n_iters=n_iters,
        burnin=burnin,
        smooth_w=smooth_w,
        n_sweeps_per_iter=n_sweeps_per_iter,
        rng=rng
    )
    secs = _now_s() - t0
    return MethodResult("Custom Blocked ESS", secs, samples, nothing, Dict())
end

# Unified comparison runner
function compare_all_inference_methods(xobs::Vector{Float64}, yobs::Vector{Float64}, T::Int, dt::Float64;
    # budgets
    n_iters::Int=6000,
    burnin::Int=1500,
    is_samples::Int=2000,

    # HMC params (tuned)
    hmc_L::Int=5,
    hmc_eps::Float64=0.003,

    # ESS params
    ess_sweeps::Int=1,

    # MH/HMC sweeps
    mh_sweeps::Int=1,
    hmc_sweeps::Int=1,

    smooth_w::Int=31,
    rng::AbstractRNG=Random.default_rng()
)
    @assert length(xobs) == T
    @assert length(yobs) == T

    results = MethodResult[]

    # 0) Custom
    push!(results, run_custom_blocked_ess_wrapper(
        xobs, yobs, T, dt;
        n_iters=n_iters, burnin=burnin, smooth_w=smooth_w,
        n_sweeps_per_iter=ess_sweeps, rng=rng
    ))

    # 1) IS
    push!(results, run_gen_importance_sampling(
        xobs, yobs, T, dt;
        n_samples=is_samples, rng=rng
    ))

    # 2) MH
    push!(results, run_gen_mh(
        xobs, yobs, T, dt;
        n_iters=n_iters, burnin=burnin, smooth_w=smooth_w,
        mh_sweeps_per_iter=mh_sweeps, rng=rng
    ))

    # 3) HMC
    push!(results, run_gen_hmc(
        xobs, yobs, T, dt;
        n_iters=n_iters, burnin=burnin, smooth_w=smooth_w,
        L=hmc_L, eps=hmc_eps, hmc_sweeps_per_iter=hmc_sweeps, rng=rng
    ))

    # print comparison
    println("\n====================")
    println("Inference comparison")
    println("====================")
    println("Budget: n_iters=$n_iters, burnin=$burnin | IS samples=$is_samples")
    println("HMC: L=$hmc_L eps=$hmc_eps | MH sweeps/iter=$mh_sweeps | HMC sweeps/iter=$hmc_sweeps | ESS sweeps/iter=$ess_sweeps")

    for r in results
        println("\n----------------------------------------")
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

T = 500
dt = 0.01
true_trace = simulate(lv_model, (T, dt))
xobs = [true_trace[:y => t => 1] for t in 1:T]
yobs = [true_trace[:y => t => 2] for t in 1:T]

println("\n=== Ground truth parameters (comparison dataset) ===")
println("alpha = ", true_trace[:alpha])
println("beta  = ", true_trace[:beta])
println("gamma = ", true_trace[:gamma])
println("delta = ", true_trace[:delta])
println("sigma = ", true_trace[:sigma])
println("x0    = ", true_trace[:x0])
println("y0    = ", true_trace[:y0])

results = compare_all_inference_methods(
    xobs, yobs, T, dt;
    n_iters=6000,
    burnin=1500,
    is_samples=2000,
    hmc_L=5,
    hmc_eps=0.0027,
    mh_sweeps=1,
    hmc_sweeps=1,
    ess_sweeps=1,
    smooth_w=31,
    rng=Random.default_rng()
)

lv_diagnostics(results[1].samples; burnin=1500, thin=1)  # Custom ESS
lv_diagnostics(results[3].samples; burnin=1500, thin=1)  # MH
lv_diagnostics(results[4].samples; burnin=1500, thin=1)  # HMC
