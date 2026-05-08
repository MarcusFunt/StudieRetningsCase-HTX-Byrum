#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv",
    [string]$WifiSsid = "",
    [string]$WifiPassword = "",
    [switch]$RotateSecrets,
    [switch]$ForceGroundMarkers,
    [switch]$SkipBoard,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-ProjectPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }

    return Join-Path $ProjectRoot $Path
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

Write-Step "Checking Python"
Get-Command $Python -ErrorAction Stop | Out-Null
$VersionCheck = "import sys; print(sys.version_info.major, sys.version_info.minor, sys.version_info.micro, sep=chr(46)); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
& $Python -c $VersionCheck
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 or newer is required. Install it or pass -Python with the path to a newer python.exe."
}

$ResolvedVenvPath = Get-ProjectPath $VenvPath
$VenvPython = Join-Path $ResolvedVenvPath "Scripts\python.exe"
$RequirementsPath = Get-ProjectPath "requirements.txt"

Write-Step "Creating virtual environment"
if (-not (Test-Path $ResolvedVenvPath)) {
    Invoke-Checked $Python @("-m", "venv", $ResolvedVenvPath)
}
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment was created, but $VenvPython was not found."
}

Write-Step "Installing Python dependencies"
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", $RequirementsPath)

Write-Step "Preparing local Wi-Fi secrets"
$SecretArgs = @("scripts\generate_secrets.py")
if ($WifiSsid) {
    $SecretArgs += @("--ssid", $WifiSsid)
}
if ($WifiPassword) {
    $SecretArgs += @("--password", $WifiPassword)
}
if ($RotateSecrets) {
    $SecretArgs += "--rotate"
}
Invoke-Checked $VenvPython $SecretArgs

Write-Step "Preparing local data and output folders"
$Folders = @(
    "data\detections",
    "data\calibration_images\charuco",
    "data\calibration_images\ground",
    "data\calibration_images\checkerboard",
    "outputs",
    "outputs\analysis"
)
foreach ($Folder in $Folders) {
    New-Item -ItemType Directory -Force -Path (Get-ProjectPath $Folder) | Out-Null
}

$GroundMarkersTemplate = Get-ProjectPath "data\ground_markers_template.csv"
$GroundMarkersPath = Get-ProjectPath "data\ground_markers.csv"
if ((-not (Test-Path $GroundMarkersPath)) -or $ForceGroundMarkers) {
    Copy-Item -Path $GroundMarkersTemplate -Destination $GroundMarkersPath -Force
    Write-Host "Created data\ground_markers.csv from the template."
} else {
    Write-Host "Keeping existing data\ground_markers.csv."
}

if (-not $SkipBoard) {
    Write-Step "Generating ChArUco calibration board"
    Invoke-Checked $VenvPython @("scripts\generate_charuco_board.py", "--output-dir", "outputs\charuco_board")
}

if (-not $SkipTests) {
    Write-Step "Running test suite"
    Invoke-Checked $VenvPython @("-m", "pytest")
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
$DisplayVenvPath = if ([System.IO.Path]::IsPathRooted($VenvPath)) { $ResolvedVenvPath } else { ".\$VenvPath" }
$DisplayActivatePath = Join-Path $DisplayVenvPath "Scripts\Activate.ps1"
$DisplayPythonPath = Join-Path $DisplayVenvPath "Scripts\python.exe"
Write-Host "Activate the environment: $DisplayActivatePath"
Write-Host "Start GUI dashboard:    $DisplayPythonPath -m panel serve pedflow/gui.py --show --autoreload"
Write-Host "Capture serial CSV:     $DisplayPythonPath scripts\capture_serial.py --port COM5 --output data\detections\session.csv"
Write-Host "Wi-Fi credentials:      secrets\pedflow_wifi.txt"
Write-Host "Firmware sketch:        GroveAIV2_Box_AP\GroveAIV2_Box_AP.ino"
