
using Gen
using LinearAlgebra
using Random
using Statistics
using Distributions

# Helper distribution
@dist log_normal(mu::Real, sigma::Real) = exp(normal(mu, sigma))

# LV dynamics (RK4)
@inline function lv_rhs(alpha, beta, gamma, delta, x, y)
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return dx, dy
end

function simulate_lv(alpha, beta, gamma, delta, x0, y0, T::Int; dt::Real = 0.1)
    # to make compatible with HMC later
    TT = promote_type(typeof(alpha), typeof(beta), typeof(gamma), typeof(delta),
                      typeof(x0), typeof(y0), typeof(dt))

    xs = Vector{TT}(undef, T)
    ys = Vector{TT}(undef, T)

    x = TT(x0)
    y = TT(y0)
    dtT = TT(dt)

    tiny = TT(1e-9)

    for t in 1:T
        xs[t] = x
        ys[t] = y

        k1x, k1y = lv_rhs(alpha, beta, gamma, delta, x, y)

        x2 = x + (dtT/TT(2)) * k1x
        y2 = y + (dtT/TT(2)) * k1y
        k2x, k2y = lv_rhs(alpha, beta, gamma, delta, x2, y2)

        x3 = x + (dtT/TT(2)) * k2x
        y3 = y + (dtT/TT(2)) * k2y
        k3x, k3y = lv_rhs(alpha, beta, gamma, delta, x3, y3)

        x4 = x + dtT * k3x
        y4 = y + dtT * k3y
        k4x, k4y = lv_rhs(alpha, beta, gamma, delta, x4, y4)

        x = x + (dtT / TT(6)) * (k1x + TT(2)*k2x + TT(2)*k3x + k4x)
        y = y + (dtT / TT(6)) * (k1y + TT(2)*k2y + TT(2)*k3y + k4y)

        # clamp to keep log() valid, but avoid Float64 conversion
        x = ifelse(x < tiny, tiny, x)
        y = ifelse(y < tiny, tiny, y)
    end

    return xs, ys
end

# Gen model 
@gen function lv_model(T::Int, dt::Float64)
    alpha = {:alpha} ~ log_normal(0.0, 0.3)
    beta  = {:beta}  ~ log_normal(-2.0, 0.3)
    gamma = {:gamma} ~ log_normal(0.0, 0.3)
    delta = {:delta} ~ log_normal(-2.0, 0.3)

    sigma = {:sigma} ~ log_normal(log(0.15), 0.3)

    x0 = {:x0} ~ log_normal(log(10.0), 0.2)
    y0 = {:y0} ~ log_normal(log(5.0), 0.2)

    xs, ys = simulate_lv(alpha, beta, gamma, delta, x0, y0, T; dt = dt)

    for t in 1:T
        {:y => t => 1} ~ log_normal(log(xs[t]), sigma)
        {:y => t => 2} ~ log_normal(log(ys[t]), sigma)
    end

    return (xs, ys)
end


# Regression estimator
function moving_average(v::AbstractVector{<:Real}, w::Int)
    @assert isodd(w)
    n = length(v)
    h = (w - 1) ÷ 2
    out = similar(float.(v))
    for i in 1:n
        lo = max(1, i - h)
        hi = min(n, i + h)
        out[i] = mean(@view v[lo:hi])
    end
    return out
end

function central_diff(v::AbstractVector{<:Real}, dt::Real)
    n = length(v)
    dv = zeros(Float64, n)
    dv[1] = (v[2] - v[1]) / dt
    for i in 2:(n - 1)
        dv[i] = (v[i + 1] - v[i - 1]) / (2 * dt)
    end
    dv[n] = (v[n] - v[n - 1]) / dt
    return dv
end

function linreg_intercept_slope(x::Vector{Float64}, y::Vector{Float64})
    X = hcat(ones(length(x)), x)
    theta = X \ y
    return theta[1], theta[2]
end

function estimate_lv_params_from_obs(xobs, yobs, dt; smooth_w = 21, eps = 1e-6)
    x = collect(Float64, xobs)
    y = collect(Float64, yobs)

    x .= max.(x, eps)
    y .= max.(y, eps)

    lx = log.(x)
    ly = log.(y)

    lx_s = moving_average(lx, smooth_w)
    ly_s = moving_average(ly, smooth_w)

    dlx = central_diff(lx_s, dt)
    dly = central_diff(ly_s, dt)

    x_s = exp.(lx_s)
    y_s = exp.(ly_s)

    n = length(x_s)
    h = (smooth_w - 1) ÷ 2
    idx = (h + 1):(n - h)

    # prey: dlx = alpha - beta*y
    alpha_hat, slope1 = linreg_intercept_slope(y_s[idx], dlx[idx])
    beta_hat = -slope1

    # predator: dly = delta*x - gamma
    intercept2, delta_hat = linreg_intercept_slope(x_s[idx], dly[idx])
    gamma_hat = -intercept2

    return max(alpha_hat, eps), max(beta_hat, eps), max(gamma_hat, eps), max(delta_hat, eps)
end

function estimate_sigma_center(xobs, yobs; smooth_w = 31, eps = 1e-12)
    lx = log.(max.(xobs, eps))
    ly = log.(max.(yobs, eps))

    lx_s = moving_average(lx, smooth_w)
    ly_s = moving_average(ly, smooth_w)

    r = vcat(lx .- lx_s, ly .- ly_s)
    return std(r)
end

function make_obs_constraints(xobs::AbstractVector, yobs::AbstractVector)
    obs = choicemap()
    for t in 1:T
        obs[:y => t => 1] = float(xobs[t])
        obs[:y => t => 2] = float(yobs[t])
    end
    return obs
end

# Loglikelihood of observations
function lv_loglik(
    alpha::Float64, beta::Float64, gamma::Float64, delta::Float64,
    x0::Float64, y0::Float64, sigma::Float64,
    xobs::Vector{Float64}, yobs::Vector{Float64},
    T::Int, dt::Float64
)
    xs, ys = simulate_lv(alpha, beta, gamma, delta, x0, y0, T; dt = dt)

    ll = 0.0
    for t in 1:T
        ll += Distributions.logpdf(LogNormal(log(xs[t]), sigma), xobs[t])
        ll += Distributions.logpdf(LogNormal(log(ys[t]), sigma), yobs[t])
    end
    return ll
end

# Helper: params from z
# z = log-params in order: alpha, beta, gamma, delta, x0, y0, sigma
@inline function z_to_params(z::Vector{Float64})
    return exp(z[1]), exp(z[2]), exp(z[3]), exp(z[4]), exp(z[5]), exp(z[6]), exp(z[7])
end

# Blocked ESS step on u = z - m.
# ESS applies because the prior in u-space is Gaussian, because priors of latents are log normal:
# u ~ N(0, Σ), where in our model Σ = diag(s.^2) due to independent log-parameter priors.
function ess_step_block!(
    rng::AbstractRNG,
    u::Vector{Float64},
    loglik_cur::Float64,
    m::Vector{Float64},
    s::Vector{Float64},
    idxs::Vector{Int},
    loglik_fn_from_z::Function;
    max_bracket_steps::Int = 50_000
)
    # ν ~ N(0, diag(s.^2)) but only on idxs
    ν = zeros(length(u))
    for i in idxs
        ν[i] = s[i] * randn(rng)
    end

    logy = loglik_cur + log(rand(rng))

    θ = 2π * rand(rng)
    θ_min = θ - 2π
    θ_max = θ

    for _ in 1:max_bracket_steps
        θtry = θ_min + rand(rng) * (θ_max - θ_min)

        u_try = copy(u)
        for i in idxs
            u_try[i] = u[i] * cos(θtry) + ν[i] * sin(θtry)
        end

        z_try = u_try .+ m
        ll_try = loglik_fn_from_z(z_try)

        if ll_try >= logy
            return u_try, ll_try, true
        else
            # shrink bracket around 0 (current state corresponds to θ=0)
            if θtry < 0
                θ_min = θtry
            else
                θ_max = θtry
            end
        end
    end

    # fallback no-move (should be rare)
    return u, loglik_cur, false
end


function run_lv_blocked_ess(
    xobs::Vector{Float64}, yobs::Vector{Float64}, T::Int, dt::Float64;
    n_iters::Int = 6000,
    burnin::Int = 1500,
    smooth_w::Int = 31,
    n_sweeps_per_iter::Int = 3,          
    global_every::Int = 10,             
    rng::AbstractRNG = Random.default_rng()
)
    @assert length(xobs) == T
    @assert length(yobs) == T

    # priors in log-space
    # means
    m = Float64[
        0.0, -2.0, 0.0, -2.0,  # alpha,beta,gamma,delta
        log(10.0), log(5.0),   # x0,y0
        log(0.15),             # sigma
    ]
    # stds
    s = Float64[
        0.3, 0.3, 0.3, 0.3,  # alpha, beta, gamma, delta
        0.2, 0.2,            # x0,y0
        0.3                  # sigma
    ]

    # init at estimated "center"
    alpha_hat, beta_hat, gamma_hat, delta_hat =
        estimate_lv_params_from_obs(xobs, yobs, dt; smooth_w = smooth_w)

    x0_center = xobs[1]
    y0_center = yobs[1]
    sigma_center = max(estimate_sigma_center(xobs, yobs; smooth_w = smooth_w), 1e-9)

    z0 = Float64[
        log(alpha_hat),
        log(beta_hat),
        log(gamma_hat),
        log(delta_hat),
        log(x0_center),
        log(y0_center),
        log(sigma_center)
    ]
    u = z0 .- m

    loglik_from_z = function(z::Vector{Float64})
        alpha, beta, gamma, delta, x0, y0, sigma = z_to_params(z)
        return lv_loglik(alpha, beta, gamma, delta, x0, y0, sigma, xobs, yobs, T, dt)
    end

    loglik_cur = loglik_from_z(u .+ m)

    # blocks
    blocks = Vector{Vector{Int}}([
        [1,2],     # alpha,beta
        [3,4],     # gamma,delta
        [5,6],     # x0,y0
        [7]        # sigma
    ])
    global_dyn = [1,2,3,4]

    samples = Vector{NamedTuple}(undef, n_iters)
    n_fallback = 0
    n_fallback_global = 0

    for it in 1:n_iters
        for _ in 1:n_sweeps_per_iter
            for b in randperm(rng, length(blocks))
                idxs = blocks[b]
                u, loglik_cur, ok = ess_step_block!(rng, u, loglik_cur, m, s, idxs, loglik_from_z)
                n_fallback += ok ? 0 : 1
            end
        end

        # occasional global dynamics move
        if global_every > 0 && (it % global_every) == 0
            u, loglik_cur, ok = ess_step_block!(rng, u, loglik_cur, m, s, global_dyn, loglik_from_z)
            n_fallback_global += ok ? 0 : 1
        end

        z = u .+ m
        alpha, beta, gamma, delta, x0, y0, sigma = z_to_params(z)

        samples[it] = (
            alpha = alpha, beta = beta, gamma = gamma, delta = delta,
            x0 = x0, y0 = y0, sigma = sigma,
            loglik = loglik_cur
        )
    end

    return samples
end



# Diagnostics 
function autocorr_at(x::AbstractVector{<:Real}, k::Int)
    n = length(x)
    @assert 0 <= k < n
    μ = mean(x)
    v = mean((x .- μ) .^ 2)
    if v == 0
        return (k == 0) ? 1.0 : 0.0
    end
    return mean((x[1:(n - k)] .- μ) .* (x[(1 + k):n] .- μ)) / v
end

function acf(x::AbstractVector{<:Real}; maxlag::Int = 50)
    n = length(x)
    maxlag = min(maxlag, n - 1)
    out = Vector{Float64}(undef, maxlag + 1)
    for k in 0:maxlag
        out[k + 1] = autocorr_at(x, k)
    end
    return out
end

function ess_geyer(x::AbstractVector{<:Real}; maxlag::Int = 2000)
    n = length(x)
    if n < 3
        return float(n)
    end
    maxlag = min(maxlag, n - 1)
    ρ = acf(x; maxlag = maxlag)
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

function print_chain_stats(name::String, chain::AbstractVector{<:Real})
    x = Float64.(chain)
    n = length(x)
    ac = acf(x; maxlag = min(50, n - 1))
    e  = ess_geyer(x; maxlag = min(2000, n - 1))

    println("\n$name:")
    println("  n = $n")
    println("  mean = $(mean(x))   std = $(std(x))")
    println("  ESS ≈ $(round(e, digits = 1))   ESS/n = $(round(e / n, digits = 3))")
    println("  ACF lags 0..10 = ", [round(ac[k + 1], digits = 3) for k in 0:10])
end

function lv_diagnostics(samples; burnin::Int = 0, thin::Int = 1)
    @assert thin >= 1
    @assert burnin >= 0
    idx = (burnin + 1):thin:length(samples)
    s2 = samples[idx]

    println("\n=== LV ACF + ESS diagnostics ===")
    println("Using ", length(s2), " samples (burnin=", burnin, ", thin=", thin, ")")

    latents = [:alpha, :beta, :gamma, :delta, :x0, :y0, :sigma]
    ess_vals = Float64[]

    for key in latents
        chain = getfield.(s2, key)
        print_chain_stats(String(key), chain)
        push!(ess_vals, ess_geyer(chain))
    end

    println("\nESS summary:")
    println("  min    = ", minimum(ess_vals))
    println("  median = ", median(ess_vals))
    println("  max    = ", maximum(ess_vals))
end


# Example run
# T = 1000
# dt = 0.01

# true_trace = simulate(lv_model, (T, dt))
# xs_true, ys_true = get_retval(true_trace)

# xobs = [true_trace[:y => t => 1] for t in 1:T]
# yobs = [true_trace[:y => t => 2] for t in 1:T]

# println("True parameters:")
# println("alpha = ", true_trace[:alpha])
# println("beta  = ", true_trace[:beta])
# println("gamma = ", true_trace[:gamma])
# println("delta = ", true_trace[:delta])
# println("sigma = ", true_trace[:sigma])

# K = 500

# # ---- ESS run ----
# samples_ridge = run_lv_blocked_ess(
#     xobs[1:K], yobs[1:K], K, dt;
#     n_iters = 6000,
#     burnin = 1500,
#     smooth_w = 31,
#     n_sweeps_per_iter = 4
# )

# println("\nPosterior summary (Blocked ESS):")
# for name in (:alpha, :beta, :gamma, :delta, :sigma, :x0, :y0)
#     vals = getfield.(samples_ridge, name)
#     println(name, ": mean=", mean(vals), " std=", std(vals))
# end

# lv_diagnostics(samples_ridge; burnin = 1500, thin = 1)
