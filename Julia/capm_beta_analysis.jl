# CAPM Beta Estimation Pipeline

using CSV
using DataFrames
using Statistics
using GLM
using Plots
using PrettyTables

# ------------------------------
# File Locations
# ------------------------------

const MARKET_RETURNS_PATH = "crsp_market_returns.csv"
const RISK_FREE_PATH = "fred_rf.csv"
const STOCK_RETURNS_PATH = "crsp_stock_returns.csv"
const BETA_OUTPUT_PATH = "capm_beta_estimates.csv"
const SUMMARY_TABLE_PATH = "capm_beta_summary.tex"
const HISTOGRAM_PATH = "capm_beta_distribution.pdf"

# ------------------------------
# Data Loading Helpers
# ------------------------------

function load_market_returns(path::AbstractString)
    df = CSV.read(path, DataFrame)
    rename!(df, Dict("caldt" => "date", "vwretd" => "MKT_RET"))
    return df
end

function load_risk_free(path::AbstractString)
    df = CSV.read(path, DataFrame)
    df.DGS1MO = df.DGS1MO ./ 100
    rename!(df, Dict("observation_date" => "date", "DGS1MO" => "RF"))
    return df
end

function load_stock_returns(path::AbstractString)
    df = CSV.read(path, DataFrame)
    df = dropmissing(df)
    df.RET = [tryparse(Float64, x) === nothing ? missing : tryparse(Float64, x) for x in df.RET]
    df = filter(row -> (row.EXCHCD == 1 || row.EXCHCD == 3) && row.SHRCD == 11, df)
    df = unique(df, [:date, :PERMCO])
    return filter(row -> !ismissing(row.RET), df)
end

# ------------------------------
# Transformation Helpers
# ------------------------------

function merge_inputs(stocks::DataFrame, market::DataFrame, risk_free::DataFrame)
    merged = innerjoin(stocks, market, on=:date)
    merged = innerjoin(merged, risk_free, on=:date)
    merged.Excess_Return = merged.RET .- merged.RF
    merged.Market_Excess_Return = merged.MKT_RET .- merged.RF
    return merged
end

function estimate_betas(df::DataFrame; min_obs::Int = 24)
    permco_type = eltype(skipmissing(df.PERMCO))
    results = DataFrame(PERMCO = Vector{permco_type}(), Beta = Float64[])

    for subdf in groupby(df, :PERMCO)
        if nrow(subdf) > min_obs
            model = lm(@formula(Excess_Return ~ Market_Excess_Return), subdf)
            push!(results, (PERMCO = subdf.PERMCO[1], Beta = coef(model)[2]))
        end
    end

    return results
end

function attach_company_names(betas::DataFrame, df::DataFrame)
    companies = unique(df[:, [:PERMCO, :COMNAM]])
    companies = unique(companies, [:PERMCO])
    return innerjoin(betas, companies, on=:PERMCO)
end

function beta_summary(betas::DataFrame)
    return describe(betas[:, [:Beta]], :min, :q25, :median, :mean, :q75, :max, :std)
end

function plot_beta_histogram(betas::DataFrame; output_path::AbstractString = HISTOGRAM_PATH)
    pos = findall(x -> -2.5 <= x <= 4.5, betas.Beta)
    plt = histogram(betas.Beta[pos], bins=30,
        xlabel="Beta", ylabel="Frequency", title="Distribution of CAPM Betas")
    savefig(plt, output_path)
    return plt
end

# ------------------------------
# Pipeline
# ------------------------------

function run_analysis(; recompute_betas::Bool = false)
    market = load_market_returns(MARKET_RETURNS_PATH)
    risk_free = load_risk_free(RISK_FREE_PATH)
    stocks = load_stock_returns(STOCK_RETURNS_PATH)
    merged = merge_inputs(stocks, market, risk_free)

    betas = if recompute_betas
        new_betas = estimate_betas(merged)
        CSV.write(BETA_OUTPUT_PATH, new_betas)
        new_betas
    else
        CSV.read(BETA_OUTPUT_PATH, DataFrame)
    end

    betas = attach_company_names(betas, merged)
    summary = beta_summary(betas)

    pretty_table(summary)
    open(SUMMARY_TABLE_PATH, "w") do io
        pretty_table(io, summary, backend = Val(:latex))
    end

    plot_beta_histogram(betas)

    return (; betas, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_analysis()
end
