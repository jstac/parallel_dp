type BasicRBC
    aalpha::Float64
    bbeta::Float64
    vProductivity
    mTransition
    capitalSteadyState::Float64
    outputSteadyState::Float64
    consumptionSteadyState::Float64
    vGridCapital
    
end

function BasicRBC(;aalpha = 1/3,
                        bbeta = 0.95,
                        vProductivity = [0.9792 0.9896 1.0000 1.0106 1.0212],
                        mTransition = [0.9727 0.0273 0.0000 0.0000 0.0000;
                                       0.0041 0.9806 0.0153 0.0000 0.0000;
                                       0.0000 0.0082 0.9837 0.0082 0.0000;
                                       0.0000 0.0000 0.0153 0.9806 0.0041;
                                       0.0000 0.0000 0.0000 0.0273 0.9727])
    
    capitalSteadyState = (aalpha*bbeta)^(1/(1-aalpha))
    outputSteadyState = capitalSteadyState^aalpha
    consumptionSteadyState = outputSteadyState-capitalSteadyState
    println("Output = ",outputSteadyState," Capital = ",capitalSteadyState," Consumption = ",consumptionSteadyState)
    vGridCapital = collect(0.5*capitalSteadyState:0.00001:1.5*capitalSteadyState)
    BasicRBC(aalpha,bbeta,vProductivity,mTransition,capitalSteadyState,outputSteadyState,
                    consumptionSteadyState,vGridCapital)
   
end
    
l(x)=length(x)

function main()
    
       v=BasicRBC()
       aalpha,bbeta,vGridCapital,vProductivity,mTransition = v.aalpha,v.bbeta,v.vGridCapital,v.vProductivity,v.mTransition    
       nGridCapital,nGridProductivity=map(l,[vGridCapital,vProductivity])
          
        # Required matrices and vectors
        mOutput           = zeros(nGridCapital,nGridProductivity)
        mValueFunction    = zeros(nGridCapital,nGridProductivity)
        mValueFunctionNew = zeros(nGridCapital,nGridProductivity)
        mPolicyFunction   = zeros(nGridCapital,nGridProductivity)
        expectedValueFunction = zeros(nGridCapital,nGridProductivity)

        #We pre-build output for each point in the grid
        mOutput = (vGridCapital.^aalpha)*vProductivity;

        # Main iteration
        maxDifference = 10.0
        tolerance = 0.0000001
        iteration = 0

        while(maxDifference > tolerance)
            expectedValueFunction = mValueFunction*mTransition';

            for nProductivity = 1:nGridProductivity
        
            # We start from previous choice (monotonicity of policy function)
                gridCapitalNextPeriod = 1
        
                for nCapital = 1:nGridCapital
        
                    valueHighSoFar = -1000.0
                    capitalChoice  = vGridCapital[1]
            
                    for nCapitalNextPeriod = gridCapitalNextPeriod:nGridCapital

                        consumption = mOutput[nCapital,nProductivity]-vGridCapital[nCapitalNextPeriod]
                        valueProvisional = (1-bbeta)*log(consumption)+bbeta*expectedValueFunction[nCapitalNextPeriod,nProductivity]
               
                        if (valueProvisional>valueHighSoFar)
                        valueHighSoFar = valueProvisional
                        capitalChoice = vGridCapital[nCapitalNextPeriod]
                        gridCapitalNextPeriod = nCapitalNextPeriod
                        else
                        break # We break when we have achieved the max
                        end
                                 
                    end
            
                    mValueFunctionNew[nCapital,nProductivity] = valueHighSoFar
                    mPolicyFunction[nCapital,nProductivity] = capitalChoice
          
                end
        
            end

            maxDifference  = maximum(abs.(mValueFunctionNew-mValueFunction))
            mValueFunction    = mValueFunctionNew
            mValueFunctionNew = zeros(nGridCapital,nGridProductivity)

            iteration = iteration+1
            if mod(iteration,10)==0 || iteration == 1
                println(" Iteration = ", iteration, " Sup Diff = ", maxDifference)
            end
           
        end

        println(" Iteration = ", iteration, " Sup Diff = ", maxDifference)
        println(" ")
        println(" My check = ", mPolicyFunction[1000,3])

    end

