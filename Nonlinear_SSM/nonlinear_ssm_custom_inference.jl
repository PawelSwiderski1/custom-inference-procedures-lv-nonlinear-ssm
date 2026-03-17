using Random
using Statistics
using Distributions
using Gen

# 1) Gen model
@dist log_normal(mu::Real, sigma::Real) = exp(normal(mu, sigma))

function f_transition(x_prev::Real, t::Int)
    return 0.5*x_prev + 25.0*x_prev/(1.0 + x_prev^2) + 8.0*cos(1.2*t)
end

function g_observe(x::Real)
    return (x^2) / 20.0
end


@gen function nl_ssm_model(T::Int)
    sigma_x = {:sigma_x} ~ log_normal(log(1.0), 0.3)
    sigma_y = {:sigma_y} ~ log_normal(log(1.0), 0.3)

    x = Vector{Float64}(undef, T)
    x[1] = {:x => 1} ~ normal(0.0, 2.0)
    {:y => 1} ~ normal(g_observe(x[1]), sigma_y)

    for t in 2:T
        μ = f_transition(x[t-1], t)
        x[t] = {:x => t} ~ normal(μ, sigma_x)
        {:y => t} ~ normal(g_observe(x[t]), sigma_y)
    end

    return x
end

# Utilities: stable categorical sampling from log-weights
function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

# sample index i ~ Categorical(proportional to exp(logw[i]))
function sample_categorical_logw(rng::AbstractRNG, logw::AbstractVector{<:Real})
    Z = logsumexp(logw)
    p = exp.(logw .- Z)
    return rand(rng, Categorical(p))
end

# PGAS step for x_{1:T} | y, sigma_x, sigma_y, reference trajectory
function pgas_step(
    rng::AbstractRNG,
    y::Vector{Float64},
    sigma_x::Float64,
    sigma_y::Float64,
    x_ref::Vector{Float64};
    N::Int = 64
)
    T = length(y)
    @assert length(x_ref) == T
    @assert N >= 2

    # particles: x[t, i]
    x = Array{Float64}(undef, T, N)

    # ancestors: a[t, i] for t>=2
    a = Array{Int}(undef, T, N)
    fill!(a, 1)

    # log weights for current time
    logw = Vector{Float64}(undef, N)

    # t = 1
    x[1, 1] = x_ref[1]  # reference particle at index 1
    prior_x1 = Normal(0.0, 2.0)
    for i in 2:N
        x[1, i] = rand(rng, prior_x1)
    end

    for i in 1:N
        logw[i] = Distributions.logpdf(Normal(g_observe(x[1, i]), sigma_y), y[1])
    end

    # t = 2..T
    for t in 2:T
        # ordinary particles i=2..N
        for i in 2:N
            parent = sample_categorical_logw(rng, logw)
            a[t, i] = parent
            μ = f_transition(x[t-1, parent], t)
            x[t, i] = rand(rng, Normal(μ, sigma_x))
        end

        # reference particle i=1: force value to reference trajectory
        x[t, 1] = x_ref[t]

        # ancestor sampling
        logas = Vector{Float64}(undef, N)
        for j in 1:N
            μj = f_transition(x[t-1, j], t)
            logas[j] = logw[j] + Distributions.logpdf(Normal(μj, sigma_x), x_ref[t])
        end
        a[t, 1] = sample_categorical_logw(rng, logas)

        # update weights using observation likelihood at time t
        for i in 1:N
            logw[i] = Distributions.logpdf(Normal(g_observe(x[t, i]), sigma_y), y[t])
        end
    end

    # sample final particle index, then backtrack to get trajectory
    k = sample_categorical_logw(rng, logw)

    x_new = Vector{Float64}(undef, T)
    x_new[T] = x[T, k]
    for t in T:-1:2
        k = a[t, k]
        x_new[t-1] = x[t-1, k]
    end

    return x_new
end


const SIGMA_PRIOR = LogNormal(log(1.0), 0.3)

function log_joint(x::Vector{Float64}, y::Vector{Float64}, sigma_x::Float64, sigma_y::Float64)
    T = length(y)
    @assert length(x) == T
    @assert sigma_x > 0 && sigma_y > 0

    # prior on sigma_x, sigma_y
    lj  = Distributions.logpdf(SIGMA_PRIOR, sigma_x)
    lj += Distributions.logpdf(SIGMA_PRIOR, sigma_y)

    # prior on x1
    lj += Distributions.logpdf(Normal(0.0, 2.0), x[1])

    # observation likelihoods + transitions
    for t in 1:T
        lj += Distributions.logpdf(Normal(g_observe(x[t]), sigma_y), y[t])
        if t >= 2
            μ = f_transition(x[t-1], t)
            lj += Distributions.logpdf(Normal(μ, sigma_x), x[t])
        end
    end
    return lj
end

function mh_update_sigma_x(rng, x, y, sigma_x, sigma_y; step_x=0.08)
    cur = log_joint(x, y, sigma_x, sigma_y)
    logsigma_x_p = log(sigma_x) + step_x * randn(rng)
    sigma_x_p = exp(logsigma_x_p)
    prop = log_joint(x, y, sigma_x_p, sigma_y)
    if log(rand(rng)) < (prop - cur)
        return sigma_x_p, true
    else
        return sigma_x, false
    end
end

function mh_update_sigma_y(rng, x, y, sigma_x, sigma_y; step_y=0.08)
    cur = log_joint(x, y, sigma_x, sigma_y)
    logsigma_y_p = log(sigma_y) + step_y * randn(rng)
    sigma_y_p = exp(logsigma_y_p)
    prop = log_joint(x, y, sigma_x, sigma_y_p)
    if log(rand(rng)) < (prop - cur)
        return sigma_y_p, true
    else
        return sigma_y, false
    end
end


# Gen-connection helpers
function make_obs_choicemap(y_obs::Vector{Float64})
    obs = choicemap()
    for t in 1:length(y_obs)
        obs[:y => t] = y_obs[t]
    end
    return obs
end

function make_latent_choicemap(x::Vector{Float64}, sigma_x::Float64, sigma_y::Float64)
    cm = choicemap()
    cm[:sigma_x] = sigma_x
    cm[:sigma_y] = sigma_y
    for t in 1:length(x)
        cm[:x => t] = x[t]
    end
    return cm
end

# Full run: simulate y in Gen, then PGAS+MH
function run_pgas_mcmc_gen_connected(
    rng::AbstractRNG;
    T::Int = 200,
    iters::Int = 3000,
    N::Int = 64,
    burn::Int = 1000,
    thin::Int = 1,
    step_x::Float64 = 0.05,
    step_y::Float64 = 0.05
)
    # simulate data from Gen
    tr_true = simulate(nl_ssm_model, (T,))
    x_true = get_retval(tr_true)
    y_obs = [tr_true[:y => t] for t in 1:T]

    println("True sigma_x = ", tr_true[:sigma_x], "   True sigma_y = ", tr_true[:sigma_y])

    # build observation constraints
    obs = make_obs_choicemap(y_obs)

    # build initial Gen trace consistent with observations
    tr_gen, _ = generate(nl_ssm_model, (T,), obs)

    sigma_x = rand(rng, SIGMA_PRIOR)
    sigma_y = rand(rng, SIGMA_PRIOR)
    x_ref = [rand(rng, Normal(0.0, 2.0)) for _ in 1:T]

    xsamps = Vector{Vector{Float64}}()
    sigs   = Vector{Tuple{Float64,Float64}}()

    accx_total = 0
    accy_total = 0

    for it in 1:iters
        # 1) PGAS update of x
        x_ref = pgas_step(rng, y_obs, sigma_x, sigma_y, x_ref; N=N)

        # 2) MH update of sigmas
        sigma_x, accx = mh_update_sigma_x(rng, x_ref, y_obs, sigma_x, sigma_y; step_x=step_x)
        sigma_y, accy = mh_update_sigma_y(rng, x_ref, y_obs, sigma_x, sigma_y; step_y=step_y)
        accx_total += accx ? 1 : 0
        accy_total += accy ? 1 : 0

        # 3) Connect to Gen
        latent_cm = make_latent_choicemap(x_ref, sigma_x, sigma_y)
        constraints = merge(obs, latent_cm)

        tr_gen, _, _, _ = update(tr_gen, (T,), (NoChange(),), constraints)

        # We compare our manual log joint to gen score to verify correctness
        if it % 200 == 0
            gen_score = get_score(tr_gen)
            manual = log_joint(x_ref, y_obs, sigma_x, sigma_y)
            @info "iter=$it  sigma_x=$(round(sigma_x, sigdigits=4))  sigma_y=$(round(sigma_y, sigdigits=4))  acc_x=$(round(accx_total/it, sigdigits=3))  acc_y=$(round(accy_total/it, sigdigits=3))"
            @info "   Gen score=$(round(gen_score, sigdigits=6))  manual log_joint=$(round(manual, sigdigits=6))  diff=$(round(manual - gen_score, sigdigits=6))"
        end

        # store
        if it > burn && ((it - burn) % thin == 0)
            push!(xsamps, copy(x_ref))
            push!(sigs, (sigma_x, sigma_y))
        end
    end

    return (x_true=x_true, y_obs=y_obs, xsamps=xsamps, sigs=sigs)
end


# Diagnostics
function autocorr_at(x::AbstractVector{<:Real}, k::Int)
    n = length(x)
    @assert 0 <= k < n
    μ = mean(x)
    v = mean((x .- μ).^2)
    if v == 0
        return NaN
    end
    return mean((x[1:(n-k)] .- μ) .* (x[(1+k):n] .- μ)) / v
end

function acf(x::AbstractVector{<:Real}; maxlag::Int=50)
    n = length(x)
    maxlag = min(maxlag, n-1)
    out = Vector{Float64}(undef, maxlag+1)
    for k in 0:maxlag
        out[k+1] = autocorr_at(x, k)
    end
    return out
end

function ess_geyer(x::AbstractVector{<:Real}; maxlag::Int=2000)
    n = length(x)
    if n < 3
        return n
    end
    maxlag = min(maxlag, n-1)

    ρ = acf(x; maxlag=maxlag) 

    s = 0.0
    m = 1
    while (2m) <= maxlag
        pair = ρ[2m] + ρ[2m+1]   
        if !(pair > 0)          
            break
        end
        s += pair
        m += 1
    end

    τ = 1.0 + 2.0*s
    if !(τ > 0) || isnan(τ) || isinf(τ)
        return 1.0
    end
    return n / τ
end

function print_chain_stats(name::String, chain::Vector{Float64})
    ac = acf(chain; maxlag=min(50, length(chain)-1))
    e  = ess_geyer(chain; maxlag=min(2000, length(chain)-1))

    println("\n$name:")
    println("  n = $(length(chain))")
    println("  mean = $(mean(chain))   std = $(std(chain))")
    println("  ESS ≈ $(round(e, digits=1))   ESS/n = $(round(e/length(chain), digits=3))")
    println("  ACF lags 0..10 = ", [round(ac[k+1], digits=3) for k in 0:10])
end

# Example run
rng = MersenneTwister(1)
res = run_pgas_mcmc_gen_connected(rng; T=200, iters=3000, N=128, burn=1000, thin=1, step_x=0.16, step_y=0.12)

sigma_x_samps = first.(res.sigs)
sigma_y_samps = last.(res.sigs)

println("Posterior mean sigma_x = ", mean(sigma_x_samps), "   sigma_y = ", mean(sigma_y_samps))

# Path-degeneracy / mixing sanity: early vs late state variability
x5   = [x[5] for x in res.xsamps]
x150 = [x[150] for x in res.xsamps]
println("std x[5]   = ", std(x5))
println("std x[150] = ", std(x150))


sigma_x_chain = first.(res.sigs)
sigma_y_chain = last.(res.sigs)

# pick a few time indices to diagnose
ts = [5, 50, 100, 150, 200]
xchains = Dict(t => [x[t] for x in res.xsamps] for t in ts)

for t in ts
    print_chain_stats("x[$t]", xchains[t])
end

println("\n=== ACF + ESS diagnostics ===")
print_chain_stats("sigma_x", sigma_x_chain)
print_chain_stats("sigma_y", sigma_y_chain)
