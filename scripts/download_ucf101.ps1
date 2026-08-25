# Download / prepare UCF101 (standard HAR recognition benchmark)
# Official page: https://www.crcv.ucf.edu/data/UCF101.php
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/download_ucf101.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/download_ucf101.ps1 -OutDir data

param(
    [string]$OutDir = "data",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $Root $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$VideoUrl = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
$SplitUrl = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
$RarPath = Join-Path $OutDir "UCF101.rar"
$SplitZip = Join-Path $OutDir "UCF101TrainTestSplits-RecognitionTask.zip"
$VideoDirCandidates = @(
    (Join-Path $OutDir "UCF-101"),
    (Join-Path $OutDir "UCF101")
)
$SplitDir = Join-Path $OutDir "ucfTrainTestlist"
$ExpectedRarBytes = [int64]6932971618

function Test-Ucf101Videos([string]$Dir) {
    if (-not (Test-Path $Dir)) { return $false }
    $n = @(Get-ChildItem -Directory $Dir -ErrorAction SilentlyContinue).Count
    return $n -ge 50
}

function Invoke-BitsResume([string]$Url, [string]$OutFile, [string]$Label, [int64]$Expected) {
    if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -ge $Expected)) {
        Write-Host "Found complete $Label ($((Get-Item $OutFile).Length) bytes)"
        return
    }
    Write-Host "Downloading $Label ..."
    Write-Host "  $Url"
    Import-Module BitsTransfer
    Get-BitsTransfer -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -eq $Label } |
        Remove-BitsTransfer -Confirm:$false
    $null = Start-BitsTransfer -Source $Url -Destination $OutFile -Asynchronous -DisplayName $Label
    while ($true) {
        Start-Sleep -Seconds 15
        $j = Get-BitsTransfer | Where-Object { $_.DisplayName -eq $Label } | Select-Object -First 1
        if ($null -eq $j) {
            if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -ge $Expected)) {
                return
            }
            throw "BITS job disappeared before $Label finished"
        }
        $pct = if ($j.BytesTotal -gt 0) { $j.BytesTransferred / $j.BytesTotal } else { 0 }
        Write-Host ("{0}  {1:N1} MB / {2:N1} MB ({3:P1})" -f $j.JobState, ($j.BytesTransferred/1MB), ($j.BytesTotal/1MB), $pct)
        if ($j.JobState -eq 'Transferred') {
            Complete-BitsTransfer -BitsJob $j
            break
        }
        if ($j.JobState -eq 'TransientError' -or $j.JobState -eq 'Error') {
            Write-Host "resume after $($j.JobState)"
            Resume-BitsTransfer -BitsJob $j
        }
    }
    if (-not (Test-Path $OutFile) -or ((Get-Item $OutFile).Length -lt $Expected)) {
        throw "Download failed for $Label"
    }
}

Write-Host "UCF101 setup"
Write-Host "  target: $OutDir"

if (-not (Test-Path (Join-Path $SplitDir "classInd.txt")) -and -not $SkipDownload) {
    if (-not (Test-Path $SplitZip)) {
        Write-Host "Downloading official splits..."
        curl.exe -L --fail -A "Mozilla/5.0" -o $SplitZip $SplitUrl
    }
    Write-Host "Extracting train/test splits..."
    tar -xf $SplitZip -C $OutDir
}

$haveVideos = $false
foreach ($cand in $VideoDirCandidates) {
    if (Test-Ucf101Videos $cand) {
        $haveVideos = $true
        Write-Host "Found videos at $cand"
        break
    }
}

if (-not $haveVideos) {
    if (-not $SkipDownload) {
        Invoke-BitsResume $VideoUrl $RarPath "UCF101" $ExpectedRarBytes
    }
    if (-not (Test-Path $RarPath)) {
        throw "UCF101.rar not found at $RarPath"
    }
    $sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
    if ($null -eq $sevenZip) {
        Write-Warning "7z not found. Install 7-Zip, then extract:"
        Write-Host "  7z x `"$RarPath`" -o`"$OutDir`""
        exit 1
    }
    Write-Host "Extracting videos with 7z..."
    & 7z x $RarPath "-o$OutDir" -y
}

$videoRoot = $null
foreach ($cand in $VideoDirCandidates) {
    if (Test-Ucf101Videos $cand) {
        $videoRoot = $cand
        break
    }
}
if ($null -eq $videoRoot) {
    throw "Could not find extracted UCF101 class folders under $OutDir"
}

Write-Host "Done."
Write-Host "  videos: $videoRoot"
Write-Host "  splits: $SplitDir"
Write-Host "Train with:"
Write-Host "  .\build\Release\har_cnn.exe train --dataset ucf101 --data `"$OutDir`""
