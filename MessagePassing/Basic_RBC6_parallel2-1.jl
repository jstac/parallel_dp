using QuantEcon
#====================================#
#  K : capitalSteadyState
#  cs : consumptionSteadyState
#  z : vProductivity
#  k_grid : GridCapital
#  out : outputSteadyState
#  T : transition matrix
#====================================#

@everywhere type BasicRBC{T <: Real}
    α::Float64
    β::Float64
    z::Vector{T}
    T::Matrix{T}
    K::Float64
    out::Float64
    cs::Float64
    k_grid::Vector{T}
end


@everywhere function BasicRBC(;α = 1/3,
                   β = 0.95)
    mc = QuantEcon.tauchen(100, 0.9, 0.1, 0.5)
    z =  mc.state_values
    T = mc.p
    K = (α*β)^(1/(1-α))
    out = K^α
    cs = out-K
    println("Output = ",out," Capital = ",K," Consumption = ",cs)
    k_grid = collect(0.5*K:0.00001:1.5*K)
    BasicRBC(α,β,z,T,K,out,cs,k_grid)
   
end
    


@everywhere _unpack(p::BasicRBC) = (p.α,p.β,p.k_grid,p.z,p.T )


#====================================#
# i : nProductivity
# j : nCapital
# l : nCapitalNextPeriod
# c : consumption
# v : valueProvisional
# v_max : valueHighSoFar
# k_choice : capitalChoice
# kNext : gridCapitalNextPeriod
#====================================#

@everywhere function bellman_operator!(p::BasicRBC,
                           output::Matrix,
                           f_vp::SharedArray,
                           E::Matrix)
    
    α,β,k_grid,z =_unpack(p)
    n = length(k_grid)
    m = length(z)    
    #-----------------------------------------------------------------
    @sync @parallel for i = 1:m
    # We start from previous choice (monotonicity of policy function)
        kNext = 1
        for j = 1:n
            v_max = -1000.0
            k_choice  = k_grid[1]
                for l = kNext : n
                    c = output[j,i]-k_grid[l]
                    v = (1-β)*log(c)+β*E[l,i]
                    if v > v_max
                        v_max = v
                        k_choice = k_grid[l]
                        kNext = l
                    else
                        break # We break when we have achieved the max
                    end
                                 
                end
                 f_vp[j,i]=(v_max,k_choice)
          end
    end
    #-----------------------------------------------------------------
    
end

#====================================#
#-------------PARSING----------------#
#====================================#
function parsing(f_valueN, f_policy, f_vp)
       @inbounds for i=1:size(f_policy, 2)
           for j=1:size(f_valueN, 1)
               f_valueN[j,i] = f_vp[j,i][1]
               f_policy[j,i] = f_vp[j,i][2]
           end     
        end
end

#====================================#
# m : length of GridProductivity(z)
# n : length of GridCapital(k_grid)
# tol : tolerance
# E : expectedValueFunction
# f_value : mValueFunction
# f_valueN : mValueFunctionNew
# f_policy : mPolicyFunction
# error : maxDifference
#iter : iteration
#====================================#

function main()

        p=BasicRBC()
        α,β,k_grid,z,T = _unpack(p)
        n = length(k_grid)
        m = length(z)
        #-----------------------------------------------------------------
        ouput    = zeros(n,m)
        f_value  = zeros(n,m)
        f_valueN = zeros(n,m)
        f_policy = zeros(n,m)
        f_vp= SharedArray{Tuple{Float64,Float64}}(n,m)
        E = zeros(n,m)
        output = (k_grid.^α)*z';
        #-----------------------------------------------------------------
        error = 10.0
        tol = 0.0000001
        iter = 0
        #-----------------------------------------------------------------
        while error > tol
            E = f_value*T';
            bellman_operator!(p ,Matrix(output),SharedArray(f_vp),Matrix(E))
            parsing(f_valueN, f_policy, f_vp)
            error  = maximum(abs.(f_valueN - f_value))
            f_value    = f_valueN
            f_valueN = zeros(n,m)
            iter = iter + 1
            if mod(iter,10)==0 || iter == 1
                println(" Iteration = ", iter, " Sup Diff = ", error)
            end 
        end
        #-----------------------------------------------------------------
        println(" Iteration = ", iter, " Sup Diff = ", error)
        println(" ")
        println(" My check = ", f_policy[1000,3])

end


