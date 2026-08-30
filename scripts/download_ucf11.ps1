# Download / prepare UCF11 (YouTube Action Dataset)
# Official page: https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/download_ucf11.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/download_ucf11.ps1 -OutDir data

param(
    [string]$OutDir = "data",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Url = "https://www.crcv.ucf.edu/data/UCF11_updated_mpg.rar"
$RarPath = Join-Path $OutDir "UCF11_updated_mpg.rar"
$ExtractDir = Join-Path $OutDir "UCF11_updated_mpg"

Write-Host "UCF11 setup"
Write-Host "  target: $ExtractDir"
Write-Host "  source: $Url"

if (-not $SkipDownload) {
    if (-not (Test-Path $RarPath)) {
        Write-Host "Downloading (this can take a while, ~1GB)..."
        try {
            Invoke-WebRequest -Uri $Url -OutFile $RarPath
        } catch {
            Write-Warning "Automatic download failed: $_"
            Write-Host "Please download manually from:"
            Write-Host "  https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php"
            Write-Host "and place/extract under: $OutDir"
            exit 1
        }
    } else {
        Write-Host "Found existing archive: $RarPath"
    }
}

if (-not (Test-Path $ExtractDir)) {
    $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
    if ($null -eq $sevenZip) {
        Write-Warning "7z not found. Install 7-Zip, then extract:"
        Write-Host "  7z x `"$RarPath`" -o`"$OutDir`""
        exit 1
    }
    Write-Host "Extracting with 7z..."
    & 7z x $RarPath "-o$OutDir" -y
}

Write-Host "Done. Train with:"
Write-Host "  .\build\har_cnn.exe train --data $ExtractDir --epochs 5 --batch 4 --size 64"
