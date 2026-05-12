param(
    [string]$SscmaPath = (Join-Path $PSScriptRoot "..\..\external\SSCMA-Micro"),
    [switch]$Reverse
)

$ErrorActionPreference = "Stop"

$patchPath = Join-Path $PSScriptRoot "patches\sscma-micro-calibsample-uart.patch"
$resolvedSscmaPath = [System.IO.Path]::GetFullPath($SscmaPath)

if (-not (Test-Path (Join-Path $resolvedSscmaPath ".git"))) {
    throw "SSCMA-Micro clone not found at $resolvedSscmaPath. Run firmware\hx6538\setup_sdks.ps1 first."
}

if (-not (Test-Path $patchPath)) {
    throw "Patch file not found at $patchPath"
}

$applyArgs = @("apply", "--check")
if ($Reverse) {
    $applyArgs += "--reverse"
}
$applyArgs += $patchPath

git -C $resolvedSscmaPath @applyArgs
if ($LASTEXITCODE -ne 0) {
    throw "Patch check failed for $patchPath"
}

$applyArgs = @("apply")
if ($Reverse) {
    $applyArgs += "--reverse"
}
$applyArgs += $patchPath

git -C $resolvedSscmaPath @applyArgs
if ($LASTEXITCODE -ne 0) {
    throw "Patch apply failed for $patchPath"
}

if ($Reverse) {
    Write-Host "Reverted SSCMA-Micro calibration patch."
}
else {
    Write-Host "Applied SSCMA-Micro calibration patch."
}
