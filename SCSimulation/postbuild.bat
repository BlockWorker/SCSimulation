echo Copying header files to output...
setlocal
set outdir="%2headers"
rd /s /q %outdir% 2>nul
mkdir %outdir%
copy /b %1BasicCombinatorial.cuh %outdir% 1>nul
copy /b %1LFSR.cuh %outdir% 1>nul
copy /b %1Stanh.cuh %outdir% 1>nul
copy /b %1circuit_component_defines.cuh %outdir% 1>nul
copy /b %1CircuitComponent.cuh %outdir% 1>nul
copy /b %1CombinatorialComponent.cuh %outdir% 1>nul
copy /b %1cuda_base.cuh %outdir% 1>nul
copy /b %1curand_base.cuh %outdir% 1>nul
copy /b %1dll.h %outdir% 1>nul
copy /b %1SequentialComponent.cuh %outdir% 1>nul
copy /b %1StochasticCircuit.cuh %outdir% 1>nul
copy /b %1StochasticCircuitFactory.cuh %outdir% 1>nul
copy /b %1StochasticNumber.cuh %outdir% 1>nul
endlocal
echo Header files copied to output
