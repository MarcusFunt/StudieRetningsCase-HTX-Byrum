<#
.SYNOPSIS
    Build, flash, and test the HX6538 vision module firmware.

.PARAMETER Port
    Serial port for flashing (e.g. COM3). Required unless -SkipFlash is set.

.PARAMETER ExternalDir
    Directory where the SDKs are cloned. Defaults to <repo-root>/external.

.PARAMETER SkipFlash
    Skip the image generation and flashing steps.

.PARAMETER SkipTest
    Skip the Python test suite.

.EXAMPLE
    # Build only (no device needed)
    .\firmware\hx6538\build.ps1 -SkipFlash

    # Full build + flash + test
    .\firmware\hx6538\build.ps1 -Port COM3
#>
param(
    [string]$Port,
    [string]$ExternalDir = (Join-Path $PSScriptRoot "..\..\external"),
    [switch]$SkipFlash,
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"

$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$externalDir = [System.IO.Path]::GetFullPath($ExternalDir)
$himaxDir    = Join-Path $externalDir "Seeed_Grove_Vision_AI_Module_V2"
$sscmaDir    = Join-Path $externalDir "SSCMA-Micro"
$overridesDir = Join-Path $PSScriptRoot "sscma-overrides"

# ── Step 1: Clone SDKs ────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== Step 1: SDKs ===" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path $externalDir | Out-Null

if (-not (Test-Path (Join-Path $himaxDir ".git"))) {
    Write-Host "Cloning Himax SDK..."
    git -c core.longpaths=true clone --recursive `
        https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2 `
        $himaxDir
} else {
    Write-Host "Himax SDK already present."
}

git -C $himaxDir config core.longpaths true
git -C $himaxDir restore --source=HEAD :/
git -C $himaxDir restore --staged :/
git -C $himaxDir submodule update --init --recursive

if (-not (Test-Path (Join-Path $sscmaDir ".git"))) {
    Write-Host "Cloning SSCMA-Micro..."
    git clone --branch 1.0.x --single-branch `
        https://github.com/Seeed-Studio/SSCMA-Micro `
        $sscmaDir
} else {
    Write-Host "SSCMA-Micro already present."
    git -C $sscmaDir checkout 1.0.x
}

# ── Step 2: Overlay custom sources ───────────────────────────────────────────

Write-Host ""
Write-Host "=== Step 2: Overlay custom sources ===" -ForegroundColor Cyan

$overrideFiles = Get-ChildItem -Path $overridesDir -Recurse -File
foreach ($src in $overrideFiles) {
    $rel  = $src.FullName.Substring($overridesDir.Length).TrimStart('\', '/')
    $dest = Join-Path $sscmaDir $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -Path $src.FullName -Destination $dest -Force
    Write-Host "  Copied: $rel"
}

# ── Step 3: Compile ───────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== Step 3: Compile ===" -ForegroundColor Cyan

$buildDir = Join-Path $himaxDir "EPII_CM55M_APP_S"
if (-not (Test-Path $buildDir)) {
    throw "Build directory not found: $buildDir"
}

Push-Location $buildDir
try {
    make clean
    if ($LASTEXITCODE -ne 0) { throw "make clean failed" }
    make
    if ($LASTEXITCODE -ne 0) { throw "make failed" }
} finally {
    Pop-Location
}

$elfPath = Join-Path $buildDir "obj_epii_evb_icv30_bdv10\gnu_epii_evb_WLCSP65\EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
if (-not (Test-Path $elfPath)) {
    throw "ELF not found after build: $elfPath"
}
Write-Host "Build succeeded: $elfPath"

if ($SkipFlash) {
    Write-Host ""
    Write-Host "Skipping flash (-SkipFlash)." -ForegroundColor Yellow
} else {

# ── Step 4: Generate boot image ───────────────────────────────────────────────

    Write-Host ""
    Write-Host "=== Step 4: Generate boot image ===" -ForegroundColor Cyan

    $imageGenDir  = Join-Path $himaxDir "we2_image_gen_local"
    $imageInput   = Join-Path $imageGenDir "input_case1_secboot"
    $imageConfig  = Join-Path $imageGenDir "project_case1_blp_wlcsp.json"
    $imageOutput  = Join-Path $imageGenDir "output_case1_sec_wlcsp\output.img"
    $imageGenExe  = Join-Path $imageGenDir "we2_local_image_gen.exe"

    New-Item -ItemType Directory -Force -Path $imageInput | Out-Null
    Copy-Item -Path $elfPath -Destination $imageInput -Force

    Push-Location $imageGenDir
    try {
        & $imageGenExe $imageConfig
        if ($LASTEXITCODE -ne 0) { throw "Image generation failed" }
    } finally {
        Pop-Location
    }

    if (-not (Test-Path $imageOutput)) {
        throw "Output image not found: $imageOutput"
    }
    Write-Host "Image ready: $imageOutput"

# ── Step 5: Flash ─────────────────────────────────────────────────────────────

    Write-Host ""
    Write-Host "=== Step 5: Flash ===" -ForegroundColor Cyan

    if (-not $Port) {
        Write-Host "No -Port specified; skipping flash." -ForegroundColor Yellow
    } else {
        $xmodemScript = Join-Path $himaxDir "xmodem\xmodem_send.py"
        python $xmodemScript `
            --port $Port `
            --baudrate 921600 `
            --protocol xmodem `
            --file $imageOutput
        if ($LASTEXITCODE -ne 0) { throw "Flash failed" }
        Write-Host "Flash complete."
    }
}

# ── Step 6: Test ─────────────────────────────────────────────────────────────

if (-not $SkipTest) {
    Write-Host ""
    Write-Host "=== Step 6: Tests ===" -ForegroundColor Cyan

    Push-Location $repoRoot
    try {
        python -m pytest tests/ -v
        if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
