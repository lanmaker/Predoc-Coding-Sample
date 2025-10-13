# CAPM Efficient Frontier Analysis

using LinearAlgebra
using Plots
using DataFrames

# ------------------------------
# Input Data
# ------------------------------

const MEAN_RETURNS = [0.26, 0.17, 0.08, 0.06, 0.02]
const COV_MATRIX = [
    0.0146  0.0057  0.0042  0.0049  0.0004;
    0.0057  0.0172  0.0044  0.0036  0.0008;
    0.0042  0.0044  0.0068  0.0034  0.0009;
    0.0049  0.0036  0.0034  0.0085  0.0011;
    0.0004  0.0008  0.0009  0.0011  0.0037;
]

const RISK_FREE_RATE = 0.01
const RETURN_GRID = range(0.01, 0.5, length=100)

# ------------------------------
# Helper Functions
# ------------------------------

"""
    efficient_frontier(mean_returns, cov_matrix; rf=RISK_FREE_RATE, return_levels=RETURN_GRID)

Return a tuple `(risk, return, weights)` describing the efficient frontier when a risk-free asset is available.
The `risk` vector contains portfolio standard deviations for each target return in `return_levels`.
"""
function efficient_frontier(mean_returns::AbstractVector, cov_matrix::AbstractMatrix;
    rf::Float64 = RISK_FREE_RATE, return_levels = RETURN_GRID)

    @assert length(mean_returns) == size(cov_matrix, 1) == size(cov_matrix, 2)

    excess = mean_returns .- rf
    scaled_weights = cov_matrix \ excess
    denominator = dot(excess, scaled_weights)

    risk = similar(return_levels, Float64)
    weights = Array{Float64}(undef, length(mean_returns), length(return_levels))

    for (idx, target_return) in enumerate(return_levels)
        xi = (target_return - rf) / denominator
        weights[:, idx] = xi .* scaled_weights
        risk[idx] = abs(target_return - rf) / sqrt(denominator)
    end

    return risk, return_levels, weights
end

"""
    market_portfolio(mean_returns, cov_matrix; rf=RISK_FREE_RATE)

Compute tangency portfolio statistics with a risk-free asset. Returns `(weights, expected_return, variance, std_dev, betas)`.
"""
function market_portfolio(mean_returns::AbstractVector, cov_matrix::AbstractMatrix;
    rf::Float64 = RISK_FREE_RATE)

    excess = mean_returns .- rf
    scaled_weights = cov_matrix \ excess
    weights = scaled_weights ./ sum(scaled_weights)

    port_return = dot(weights, mean_returns)
    port_variance = dot(weights, cov_matrix * weights)
    betas = (cov_matrix * weights) ./ port_variance

    return weights, port_return, port_variance, sqrt(port_variance), betas
end

function plot_frontier!(plot_obj, risks, returns; label)
    plot!(plot_obj, risks, returns, linestyle=:dash, lw=2, label=label)
    return plot_obj
end

# ------------------------------
# Main Routine
# ------------------------------

function analyze_frontier(mean_returns::AbstractVector, cov_matrix::AbstractMatrix;
    rf::Float64 = RISK_FREE_RATE, return_levels = RETURN_GRID)

    risks, returns, _ = efficient_frontier(mean_returns, cov_matrix; rf=rf, return_levels=return_levels)
    weights, port_return, port_variance, port_std, betas = market_portfolio(mean_returns, cov_matrix; rf=rf)

    result_df = DataFrame(Asset = 1:length(mean_returns), Weight = weights, Beta = betas)

    stats = (
        expected_return = port_return,
        variance = port_variance,
        std_dev = port_std,
        weights = weights,
        betas = betas,
        table = result_df,
    )

    return risks, returns, stats
end

function main()
    full_risks, full_returns, full_stats = analyze_frontier(MEAN_RETURNS, COV_MATRIX)

    # Remove the first asset (Apple) for comparison
    no_apple_returns = MEAN_RETURNS[2:end]
    no_apple_cov = COV_MATRIX[2:end, 2:end]
    trimmed_risks, trimmed_returns, trimmed_stats = analyze_frontier(no_apple_returns, no_apple_cov)

    plt = plot(full_risks, full_returns, linestyle=:dash, lw=2, label="All assets",
        xlabel="Risk (Std Dev)", ylabel="Expected Return", legend=:bottomright)
    plot_frontier!(plt, trimmed_risks, trimmed_returns; label="Without Apple")
    savefig(plt, "ps7_fig2.pdf")
    savefig(plot(full_risks, full_returns, linestyle=:dash, lw=2,
        label="Efficient frontier", xlabel="Risk (Std Dev)", ylabel="Expected Return", legend=:bottomright), "ps7_fig1.pdf")

    println("Market Portfolio (all assets)")
    show(full_stats.table)
    println("\nExpected Return: $(round(full_stats.expected_return, digits=4))")
    println("Variance: $(round(full_stats.variance, digits=4))")
    println("Standard Deviation: $(round(full_stats.std_dev, digits=4))")

    println("\nMarket Portfolio (without Apple)")
    show(trimmed_stats.table)
    println("\nExpected Return: $(round(trimmed_stats.expected_return, digits=4))")
    println("Variance: $(round(trimmed_stats.variance, digits=4))")
    println("Standard Deviation: $(round(trimmed_stats.std_dev, digits=4))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
