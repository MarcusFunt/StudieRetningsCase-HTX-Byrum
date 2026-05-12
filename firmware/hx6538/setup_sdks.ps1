param(
    [string]$ExternalDir = (Join-Path $PSScriptRoot "..\..\external")
)

$ErrorActionPreference = "Stop"

$externalPath = [System.IO.Path]::GetFullPath($ExternalDir)
$himaxPath = Join-Path $externalPath "Seeed_Grove_Vision_AI_Module_V2"
$sscmaPath = Join-Path $externalPath "SSCMA-Micro"

New-Item -ItemType Directory -Force -Path $externalPath | Out-Null

if (-not (Test-Path (Join-Path $himaxPath ".git"))) {
    git -c core.longpaths=true clone --recursive `
        https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2 `
        $himaxPath
}
else {
    Write-Host "Himax SDK already exists at $himaxPath"
}

git -C $himaxPath config core.longpaths true
git -C $himaxPath restore --source=HEAD :/
git -C $himaxPath restore --staged :/
git -C $himaxPath submodule update --init --recursive

if (-not (Test-Path (Join-Path $sscmaPath ".git"))) {
    git clone --branch 1.0.x --single-branch `
        https://github.com/Seeed-Studio/SSCMA-Micro `
        $sscmaPath
}
else {
    Write-Host "SSCMA-Micro already exists at $sscmaPath"
}

git -C $sscmaPath checkout 1.0.x

Write-Host "SDKs are ready:"
Write-Host "  $himaxPath"
Write-Host "  $sscmaPath"
