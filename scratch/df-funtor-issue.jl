using DataFrames
using Statistics: mean, std

abstract type Projector <: Function end

struct Normalizer <: Projector
    μ
    σ
end

struct Quantilizer <: Projector
    quantiles
end

Normalizer(x::AbstractVector) = Normalizer(mean(x), std(x))

function (m::Normalizer)(x::Real)
    return (x - m.μ) / m.σ
end

function (m::Normalizer)(x::AbstractVector)
    return (x .- m.μ) ./ m.σ
end

df = DataFrame(:v1 => rand(5), :v2 => rand(5))
feat_names = names(df)
norms = map((feat) -> Normalizer(df[:, feat]), feat_names)

# ByRow works
transform(df, feat_names .=> ByRow.(norms) .=> feat_names)

# Does not work
transform(df, feat_names[1] => norms[1] => feat_names[1])
# Does work
transform(df, feat_names[1] => (x -> norms[1](x)) => feat_names[1])

# Does not work
transform(df, feat_names .=> norms .=> feat_names)

# Does work
norms_f = map(f -> (x) -> f(x), norms)
transform(df, feat_names .=> norms_f .=> feat_names)



#################
# multi-function pattern
using DataFrames
using BenchmarkTools
nrows = 10_000_000
df = DataFrame(id = rand(["A"], nrows), v1 = rand(nrows), v2 = rand(nrows), v3 = rand(nrows), v4 = rand(nrows))

fun1(x) = x^2
fun2(x) = sin(x)
fun3(x) = cos(x)
fun4(x) = exp(x)

f1  = "v1" => ByRow(fun1) => "new1"
f2 = "v2" => ByRow(fun2) => "new2"
f3 = "v2" => ByRow(fun3) => "new3"
f4 = "v2" => ByRow(fun4) => "new4"

funs = [f1, f2, f3, f4]

function df_trans_A(df, funs)
    transform!(df, funs[1])
    transform!(df, funs[2])
    transform!(df, funs[3])
    transform!(df, funs[4])
end

function df_trans_B(df, funs)
    transform!(df, funs)
end

@btime df_trans_A($df, $funs);
@btime df_trans_B($df, $funs);


gdf = groupby(df, "id")
function gdf_trans_A(gdf, funs)
    transform!(gdf, funs[1])
    transform!(gdf, funs[2])
    transform!(gdf, funs[3])
    transform!(gdf, funs[4])
end

function gdf_trans_B(gdf, funs)
    transform!(gdf, funs)
end

@time gdf_trans_A(gdf, funs);
@time gdf_trans_B(gdf, funs);

@btime gdf_trans_A($gdf, $funs);
@btime gdf_trans_B($gdf, $funs);
