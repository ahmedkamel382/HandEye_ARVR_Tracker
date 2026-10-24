[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$HandImage = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $ProjectRoot
$BuildDir = Join-Path $ProjectRoot "build"
$DistDir = Join-Path $ProjectRoot "dist"

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python executable not found: $PythonExecutable"
}

$Cmake = (Get-Command cmake -ErrorAction SilentlyContinue | Select-Object -First 1).Source
$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$VsRoot = ""
if (Test-Path -LiteralPath $VsWhere) {
    $VsRoot = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
}
if (-not $Cmake -and $VsRoot) {
    $Candidate = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path -LiteralPath $Candidate) {
        $Cmake = $Candidate
    }
}
if (-not $Cmake) {
    throw "CMake was not found. Install Visual Studio C++ build tools or put cmake on PATH."
}

$Generator = $null
if ($VsRoot -match "\\18\\") {
    $Generator = "Visual Studio 18 2026"
} elseif ($VsRoot -match "\\17\\") {
    $Generator = "Visual Studio 17 2022"
} elseif ($VsRoot -match "\\16\\") {
    $Generator = "Visual Studio 16 2019"
}

$ConfigureArgs = @(
    "-S", $ProjectRoot,
    "-B", $BuildDir,
    "-A", "x64",
    "-DPython3_EXECUTABLE=$PythonExecutable"
)
if ($Generator) {
    $ConfigureArgs += @("-G", $Generator)
}

Write-Host "Configuring index tracker DLL..."
& $Cmake @ConfigureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed"
}

Write-Host "Building index tracker DLL ($Configuration)..."
& $Cmake --build $BuildDir --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed"
}

$BinDir = Join-Path $BuildDir "bin\$Configuration"
$LibDir = Join-Path $BuildDir "lib\$Configuration"
$DllPath = Join-Path $BinDir "hand_index_tracker.dll"
$LibPath = Join-Path $LibDir "hand_index_tracker.lib"
$SmokePath = Join-Path $BinDir "hand_index_smoke.exe"
$ExamplePath = Join-Path $BinDir "hand_index_example.exe"
$IntegrationPath = Join-Path $BinDir "hand_index_real_backend_integration.exe"

foreach ($Required in @($DllPath, $LibPath, $SmokePath, $ExamplePath, $IntegrationPath)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Expected build artifact missing: $Required"
    }
}

if ($VsRoot) {
    $Dumpbin = Get-ChildItem -Path (Join-Path $VsRoot "VC\Tools\MSVC") -Recurse -Filter dumpbin.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*Hostx64*x64*" } |
        Select-Object -First 1 -ExpandProperty FullName
    if ($Dumpbin) {
        $Exports = & $Dumpbin /exports $DllPath
        $ExportText = $Exports -join "`n"
        $ExpectedExports = @(
            "hand_index_version",
            "hand_index_abi_version",
            "hand_index_error_string",
            "hand_index_config_default",
            "hand_index_result_init",
            "hand_index_create",
            "hand_index_process_bgr",
            "hand_index_destroy"
        )
        $MissingExports = @($ExpectedExports | Where-Object { $ExportText -notmatch "\b$_\b" })
        if ($MissingExports.Count -gt 0) {
            throw "Missing DLL exports: $($MissingExports -join ', ')"
        }
        Write-Host "Verified all $($ExpectedExports.Count) expected DLL exports."
    }
}

if (-not $SkipTests) {
    $PythonBase = (& $PythonExecutable -c "import sys; print(sys.base_prefix)").Trim()
    $env:PATH = "$PythonBase;$env:PATH"

    Write-Host "Running mock interface test..."
    & $SmokePath --mock
    if ($LASTEXITCODE -ne 0) {
        throw "Mock DLL interface test failed"
    }

    Write-Host "Running real MediaPipe initialization/loss test..."
    & $SmokePath --real
    if ($LASTEXITCODE -ne 0) {
        throw "Real DLL interface test failed"
    }

    if (-not [string]::IsNullOrWhiteSpace($HandImage)) {
        if (-not (Test-Path -LiteralPath $HandImage)) {
            throw "Hand image not found: $HandImage"
        }
        Write-Host "Running real hand-image integration test..."
        & $IntegrationPath $HandImage
        if ($LASTEXITCODE -ne 0) {
            throw "Real hand-image integration test failed"
        }
    } else {
        Write-Warning "No -HandImage supplied; the real valid-hand integration test was not run."
    }
}

Write-Host "Staging distribution..."
$DistBin = Join-Path $DistDir "bin"
$DistLib = Join-Path $DistDir "lib"
$DistInclude = Join-Path $DistDir "include"
$DistPython = Join-Path $DistDir "python"
$DistModel = Join-Path $DistDir "model"
$DistExample = Join-Path $DistDir "examples"
foreach ($Directory in @($DistBin, $DistLib, $DistInclude, $DistPython, $DistModel, $DistExample)) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

Copy-Item -LiteralPath $DllPath -Destination $DistBin -Force
Copy-Item -LiteralPath $LibPath -Destination $DistLib -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "include\hand_index_tracker.h") -Destination $DistInclude -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "ai_pipeline\hand_kinematics\index_tracker.py") -Destination $DistPython -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "ai_pipeline\hand_kinematics\hand_features.py") -Destination $DistPython -Force
Copy-Item -LiteralPath (Join-Path $RepoRoot "assets\hand_landmarker.task") -Destination $DistModel -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "examples\index_tracker_example.cpp") -Destination $DistExample -Force
Copy-Item -LiteralPath $SmokePath -Destination $DistBin -Force
Copy-Item -LiteralPath $ExamplePath -Destination $DistBin -Force
Copy-Item -LiteralPath $IntegrationPath -Destination $DistBin -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "README.md") -Destination $DistDir -Force
Remove-Item -LiteralPath (Join-Path $DistPython "__pycache__") -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Build complete."
Write-Host "DLL:  $DllPath"
Write-Host "LIB:  $LibPath"
Write-Host "DIST: $DistDir"
