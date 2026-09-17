$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    & "$repoRoot/local-deps/emsdk/python/3.13.3_64bit/python.exe" "$repoRoot/local-deps/emsdk/upstream/emscripten/emcc.py" wasm/full-model.c -O3 -msimd128 -pthread -ffp-contract=off --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 '-sENVIRONMENT=web,worker,node' -sPTHREAD_POOL_SIZE=8 -sINITIAL_MEMORY=268435456 -sALLOW_MEMORY_GROWTH=0 '-sEXPORTED_FUNCTIONS=["_malloc","_free","_setup","_reset_model","_evaluate"]' '-sEXPORTED_RUNTIME_METHODS=["HEAPU8","HEAPF32","PThread"]' -o wasm/full-model.mjs
    if ($LASTEXITCODE -ne 0) { throw 'WASM build failed' }
} finally { Pop-Location }
