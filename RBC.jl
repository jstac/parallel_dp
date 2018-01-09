
workspace()

mutable struct RBC{T <: Real}
    α::Float64
    β::Float64
    z::Matrix{T}
    T::Matrix{T}
    K::Float64
    out::Float64
    cs::Float64
    k_grid::Vector{T}
    output::Matrix{Float64}
    f_valueN::Matrix{Float64}
    f_value::Matrix{Float64}
    f_policy::Matrix{Float64}
    n::Int64
    m::Int64

end

function RBC(;α = 1/3,
                   β = 0.95,
                   z = [0.9792 0.9896 1.0000 1.0106 1.0212],
                   T = [0.9727 0.0273 0.0000 0.0000 0.0000;
                        0.0041 0.9806 0.0153 0.0000 0.0000;
                        0.0000 0.0082 0.9837 0.0082 0.0000;
                        0.0000 0.0000 0.0153 0.9806 0.0041;
                        0.0000 0.0000 0.0000 0.0273 0.9727])
    
    K = (α * β)^(1 / (1 - α))
    out = K^α
    cs = out - K
    k_grid = collect(0.5*K:0.00001:1.5*K)
    output = (k_grid.^α) * z;
    n = length(k_grid)
    m = length(z)
    f_valueN = zeros(n, m)
    f_value = zeros(n, m)
    f_policy = zeros(n, m)
    RBC(α, β, z, T, K, out, cs, k_grid, output, f_valueN, f_value, f_policy, n, m)
    
end

function vfi!(rbc::RBC,
              bellman_operator!::Function; tol::Float64=1e-7,error::Float64=10.0)
        iter = 0
    while error > tol 
        bellman_operator!(rbc)
        error  = maximum(abs.(rbc.f_valueN - rbc.f_value))
        rbc.f_value = rbc.f_valueN
        rbc.f_valueN = zeros(rbc.n, rbc.m)
        iter = iter + 1
         if mod(iter,10)==0 || iter == 1
                println(" Iteration = ", iter, " Sup Diff = ", error)
        end 

    end
        println(" My check = ", rbc.f_policy[1000, 3])
end

function bellman_operator!(rbc::RBC)
       
        E = rbc.f_value * rbc.T'
        for i = 1: rbc.m
            kNext = 1
            for j = 1:rbc.n
                v_max = -1000.0
                k_choice  = rbc.k_grid[1]
                    for l = kNext : rbc.n
                        c = rbc.output[j, i] - rbc.k_grid[l]
                        v = (1 - rbc.β) * log(c) + rbc.β * E[l, i]
                        if v > v_max
                            v_max = v
                            k_choice = rbc.k_grid[l]
                            kNext = l
                        else
                            break 
                        end
                                 
                    end
                    rbc.f_valueN[j, i] = v_max
                    rbc.f_policy[j, i] = k_choice
              end
        end    
     
end


rbc=RBC();

@time vfi!(rbc, bellman_operator!)

