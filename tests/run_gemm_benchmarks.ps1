$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildDir = Join-Path $env:TEMP ("gemm-bench-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $buildDir | Out-Null

function Invoke-BenchmarkBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$ExeName,

        [string[]]$ExtraArgs = @()
    )

    Write-Host "==> $Source"
    $sourcePath = Join-Path $rootDir $Source
    $exePath = Join-Path $buildDir $ExeName
    & nvcc -std=c++17 -I $rootDir @ExtraArgs $sourcePath -o $exePath
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed for $Source"
    }
    & $exePath
    if ($LASTEXITCODE -ne 0) {
        throw "$ExeName failed"
    }
    Write-Host ""
}

Invoke-BenchmarkBuild -Source "bench_gemm.cu" -ExeName "bench_gemm.exe"
Invoke-BenchmarkBuild -Source "bench_optimized_gemm.cu" -ExeName "bench_optimized_gemm.exe"
Invoke-BenchmarkBuild -Source "bench_tensor_core_gemm.cu" -ExeName "bench_tensor_core_gemm.exe" -ExtraArgs @("-arch=sm_80")
Invoke-BenchmarkBuild -Source "bench_hyperoptimized_gemm.cu" -ExeName "bench_hyperoptimized_gemm.exe" -ExtraArgs @("-arch=sm_80", "-Xcompiler", "/Zc:preprocessor")

Write-Host "build artifacts: $buildDir"
