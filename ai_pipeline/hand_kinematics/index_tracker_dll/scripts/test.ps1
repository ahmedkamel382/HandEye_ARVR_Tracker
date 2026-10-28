[CmdletBinding()]
param(
    [string]$PythonExecutable = "",
    [string]$HandImage = "",
    [switch]$Camera,
    [ValidateRange(0, 16)]
    [int]$CameraIndex = 0,
    [ValidateRange(1, 120)]
    [int]$CameraSeconds = 10,
    [switch]$CameraPreview,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

$script:Passed = 0
$script:Failed = 0
$script:Skipped = 0
$script:StartedAt = Get-Date

function Write-Status {
    param(
        [string]$Status,
        [string]$Message
    )

    $color = switch ($Status) {
        "PASS" { "Green" }
        "FAIL" { "Red" }
        "SKIP" { "Yellow" }
        default { "Cyan" }
    }

    Write-Host ("[{0}] {1}" -f $Status, $Message) -ForegroundColor $color
    switch ($Status) {
        "PASS" { $script:Passed++ }
        "FAIL" { $script:Failed++ }
        "SKIP" { $script:Skipped++ }
    }
}

function Exit-Runner {
    param([int]$Code)

    $elapsed = (Get-Date) - $script:StartedAt
    Write-Host ""
    Write-Host ("Summary: {0} passed, {1} failed, {2} skipped, {3:N1}s" -f
        $script:Passed, $script:Failed, $script:Skipped, $elapsed.TotalSeconds)
    if ($Code -eq 0 -and $script:Failed -gt 0) {
        $Code = 1
    }
    exit $Code
}

function Invoke-TestExecutable {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Name
    )

    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $FilePath @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    } catch {
        Write-Status "FAIL" ("{0} - {1}" -f $Name, $_.Exception.Message)
        return $false
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }

    foreach ($line in $output) {
        Write-Host ("  {0}" -f $line)
    }
    if ($exitCode -ne 0) {
        Write-Status "FAIL" ("{0} (exit {1})" -f $Name, $exitCode)
        return $false
    }

    Write-Status "PASS" $Name
    return $true
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = $ProjectRoot
while (-not (Test-Path -LiteralPath (Join-Path $RepoRoot ".git"))) {
    $Parent = Split-Path -Parent $RepoRoot
    if ([string]::IsNullOrEmpty($Parent) -or $Parent -eq $RepoRoot) {
        Write-Status "FAIL" "Environment preflight - cannot locate repository root"
        Exit-Runner 10
    }
    $RepoRoot = $Parent
}

$BuildScript = Join-Path $ProjectRoot "scripts\build.ps1"
$BuildBin = Join-Path $ProjectRoot "build\bin\Release"
$BuildLib = Join-Path $ProjectRoot "build\lib\Release"
$DllPath = Join-Path $BuildBin "hand_index_tracker.dll"
$LibPath = Join-Path $BuildLib "hand_index_tracker.lib"
$SmokePath = Join-Path $BuildBin "hand_index_smoke.exe"
$IntegrationPath = Join-Path $BuildBin "hand_index_real_backend_integration.exe"
$CameraProbe = Join-Path $ProjectRoot "tests\camera_probe.py"
$DistPython = Join-Path $ProjectRoot "dist\python"
$DistModel = Join-Path $ProjectRoot "dist\model\hand_landmarker.task"
$DistIndexTracker = Join-Path $DistPython "index_tracker.py"
$DistHandFeatures = Join-Path $DistPython "hand_features.py"

Write-Host "hand_index_tracker.dll test runner"
Write-Host ("Project: {0}" -f $ProjectRoot)
Write-Host ""

if ($env:OS -ne "Windows_NT") {
    Write-Status "FAIL" "Environment preflight - Windows is required"
    Exit-Runner 10
}
if (-not [Environment]::Is64BitOperatingSystem) {
    Write-Status "FAIL" "Environment preflight - 64-bit Windows is required"
    Exit-Runner 10
}
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Status "FAIL" "Environment preflight - PowerShell 5.1 or newer is required"
    Exit-Runner 10
}
foreach ($RequiredPath in @($BuildScript, $CameraProbe, $DistIndexTracker, $DistHandFeatures, $DistModel)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        Write-Status "FAIL" ("Environment preflight - missing {0}" -f $RequiredPath)
        Exit-Runner 10
    }
}
if ((Get-Item -LiteralPath $DistModel).Length -le 0) {
    Write-Status "FAIL" ("Environment preflight - model is empty: {0}" -f $DistModel)
    Exit-Runner 10
}

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $RepoVenv = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $RepoVenv) {
        $PythonExecutable = $RepoVenv
    } else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($PythonCommand -and $PythonCommand.Source) {
            $PythonExecutable = $PythonCommand.Source
        }
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExecutable) -or
    -not (Test-Path -LiteralPath $PythonExecutable)) {
    Write-Status "FAIL" "Environment preflight - Python not found; pass -PythonExecutable or create .venv"
    Exit-Runner 10
}
$PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path

if (-not [string]::IsNullOrWhiteSpace($HandImage)) {
    if (-not (Test-Path -LiteralPath $HandImage)) {
        Write-Status "FAIL" ("Environment preflight - hand image not found: {0}" -f $HandImage)
        Exit-Runner 10
    }
    $HandImage = (Resolve-Path -LiteralPath $HandImage).Path
}

$pythonPreflight = @'
import importlib
import site
import sys

dist_python = sys.argv[1]
if sys.version_info < (3, 11):
    print(f'Python 3.11+ is required; found {sys.version.split()[0]}')
    raise SystemExit(1)
if sys.maxsize <= 2 ** 32:
    print('64-bit Python is required')
    raise SystemExit(1)
if sys.version_info[:2] != (3, 11):
    print(f'WARNING: validated with Python 3.11; found {sys.version.split()[0]}')

for package in ('numpy', 'cv2', 'mediapipe'):
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        print(f'missing Python package {package}: {exc}')
        raise SystemExit(1)
    version = getattr(module, '__version__', 'unknown')
    print(f'{package}={version}')

sys.path.insert(0, dist_python)
try:
    hand_features = importlib.import_module('hand_features')
    index_tracker = importlib.import_module('index_tracker')
except Exception as exc:
    print(f'cannot import DLL runtime modules from {dist_python}: {exc}')
    raise SystemExit(1)
if not hasattr(index_tracker, 'IndexFingerTracker'):
    print('index_tracker.IndexFingerTracker is missing')
    raise SystemExit(1)

site_packages = next(
    (path for path in site.getsitepackages() if path.replace(chr(92), '/').endswith('/site-packages')),
    '',
)
if not site_packages:
    print('cannot determine site-packages directory')
    raise SystemExit(1)

print(f'python={sys.version.split()[0]} ({sys.executable})')
print(f'python_base={sys.base_prefix}')
print(f'site_packages={site_packages}')
print('runtime modules loaded')
'@

$previousErrorAction = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $preflightOutput = & $PythonExecutable -c $pythonPreflight $DistPython 2>&1
    $preflightExit = $LASTEXITCODE
} catch {
    Write-Status "FAIL" ("Environment preflight - {0}" -f $_.Exception.Message)
    Exit-Runner 10
} finally {
    $ErrorActionPreference = $previousErrorAction
}
foreach ($line in $preflightOutput) {
    Write-Host ("  {0}" -f $line)
}
if ($preflightExit -ne 0) {
    Write-Status "FAIL" "Environment preflight - Python packages or runtime modules are unavailable"
    Exit-Runner 10
}

$PythonBaseLine = $preflightOutput | ForEach-Object { [string]$_ } |
    Where-Object { $_ -match "^python_base=" } | Select-Object -First 1
$SitePackagesLine = $preflightOutput | ForEach-Object { [string]$_ } |
    Where-Object { $_ -match "^site_packages=" } | Select-Object -First 1
$PythonBase = if ($PythonBaseLine) {
    ($PythonBaseLine -replace "^python_base=", "").Trim()
} else {
    ""
}
$SitePackages = if ($SitePackagesLine) {
    ($SitePackagesLine -replace "^site_packages=", "").Trim()
} else {
    ""
}
if ([string]::IsNullOrWhiteSpace($PythonBase) -or -not (Test-Path -LiteralPath $PythonBase)) {
    Write-Status "FAIL" "Environment preflight - cannot determine Python base directory"
    Exit-Runner 10
}
if ([string]::IsNullOrWhiteSpace($SitePackages) -or -not (Test-Path -LiteralPath $SitePackages)) {
    Write-Status "FAIL" "Environment preflight - cannot determine Python site-packages directory"
    Exit-Runner 10
}
$env:PATH = "$PythonBase;$env:PATH"

if (-not $SkipBuild) {
    $Cmake = (Get-Command cmake -ErrorAction SilentlyContinue | Select-Object -First 1).Source
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $VsRoot = ""
    if (Test-Path -LiteralPath $VsWhere) {
        $VsRoot = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
    }
    if (-not $VsRoot) {
        Write-Status "FAIL" "Environment preflight - Visual Studio C++ x64 build tools not found"
        Exit-Runner 10
    }
    if (-not $Cmake) {
        $VsCmake = Join-Path $VsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if (-not (Test-Path -LiteralPath $VsCmake)) {
            Write-Status "FAIL" "Environment preflight - CMake not found; install CMake or VS C++ tools"
            Exit-Runner 10
        }
    }
} else {
    Write-Status "SKIP" "C++ build tool check - -SkipBuild"
}

Write-Status "PASS" "Environment preflight"

if ($SkipBuild) {
    Write-Status "SKIP" "DLL build - -SkipBuild; using existing build\bin\Release"
} else {
    try {
        & $BuildScript -PythonExecutable $PythonExecutable -SkipTests -SkipStage
        Write-Status "PASS" "DLL build (Release)"
    } catch {
        Write-Status "FAIL" ("DLL build - {0}" -f $_.Exception.Message)
        Exit-Runner 20
    }
}

$MissingArtifacts = @()
foreach ($RequiredArtifact in @($DllPath, $LibPath, $SmokePath, $IntegrationPath)) {
    if (-not (Test-Path -LiteralPath $RequiredArtifact)) {
        $MissingArtifacts += $RequiredArtifact
    }
}
if ($MissingArtifacts.Count -gt 0) {
    Write-Status "FAIL" ("Build artifacts - missing {0}" -f ($MissingArtifacts -join ", "))
    Exit-Runner 30
}
Write-Status "PASS" "Build artifacts"
Write-Host ("  Testing DLL: {0}" -f $DllPath)

$runtimeReady = $false
$realBackendReady = $false
$failureCode = 0

Push-Location -LiteralPath $BuildBin
try {
    $runtimeReady = Invoke-TestExecutable -FilePath $SmokePath -Arguments @("--mock") `
        -Name "Exported API smoke tests (mock backend)"

    if (-not $runtimeReady) {
        $failureCode = 40
        Write-Status "SKIP" "MediaPipe initialization - DLL smoke tests failed"
    } else {
        $realBackendReady = Invoke-TestExecutable -FilePath $SmokePath -Arguments @("--real") `
            -Name "MediaPipe initialization and no-hand loss"
        if (-not $realBackendReady) {
            $failureCode = 50
        }
    }

    if (-not $realBackendReady) {
        Write-Status "SKIP" "Landmark tracking on sample image - real backend unavailable"
    } elseif ([string]::IsNullOrWhiteSpace($HandImage)) {
        Write-Status "SKIP" "Landmark tracking on sample image - no -HandImage supplied (use -Camera for live input)"
    } else {
        $imageOk = Invoke-TestExecutable -FilePath $IntegrationPath -Arguments @($HandImage) `
            -Name ("Landmark tracking / loss / reacquisition ({0})" -f (Split-Path -Leaf $HandImage))
        if (-not $imageOk) {
            $failureCode = 60
        }
    }

    $cameraRequested = $Camera.IsPresent -or $CameraPreview.IsPresent -or $PSBoundParameters.ContainsKey("CameraIndex")
    if (-not $cameraRequested) {
        Write-Status "SKIP" "Camera test - not requested"
    } elseif (-not $realBackendReady) {
        Write-Status "SKIP" "Camera test - real backend unavailable"
    } else {
        $cameraArguments = @(
            $CameraProbe,
            "--dll", $DllPath,
            "--python-dir", $DistPython,
            "--model", $DistModel,
            "--site-packages", $SitePackages,
            "--camera-index", [string]$CameraIndex,
            "--seconds", [string]$CameraSeconds
        )
        if ($CameraPreview) {
            $cameraArguments += "--preview"
        }
        $cameraStageName = if ($CameraPreview) {
            "Live camera preview (index {0})" -f $CameraIndex
        } else {
            "Live camera test (index {0})" -f $CameraIndex
        }
        $cameraOk = Invoke-TestExecutable -FilePath $PythonExecutable -Arguments $cameraArguments `
            -Name $cameraStageName
        if (-not $cameraOk) {
            $failureCode = 70
        }
    }
} finally {
    Pop-Location
}

Exit-Runner $failureCode
