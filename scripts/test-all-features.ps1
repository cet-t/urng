#!/usr/bin/env pwsh
# Runs the test suite across every feature flag combination.
$ErrorActionPreference = "Stop"

$cargoTomlPath = Join-Path $PSScriptRoot "..\Cargo.toml"
$cargoToml = Get-Content -Path $cargoTomlPath -Raw

$sectionMatch = [regex]::Match($cargoToml, '(?ms)^\[features\]\s*\r?\n(.*?)(?=^\[|\z)')
if (-not $sectionMatch.Success) {
    Write-Host "No [features] section found in $cargoTomlPath" -ForegroundColor Red
    exit 1
}

$features = @()
foreach ($line in ($sectionMatch.Groups[1].Value -split "`r?`n")) {
    $line = $line.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { continue }
    $nameMatch = [regex]::Match($line, '^([A-Za-z0-9_-]+)\s*=')
    if ($nameMatch.Success) { $features += $nameMatch.Groups[1].Value }
}

if ($features.Count -eq 0) {
    Write-Host "No features parsed from $cargoTomlPath" -ForegroundColor Red
    exit 1
}

Write-Host "Discovered features: $($features -join ', ')" -ForegroundColor DarkGray

function Invoke-Test {
    param([string[]]$TestArgs)
    Write-Host "==> cargo test $($TestArgs -join ' ')" -ForegroundColor Cyan
    & cargo test @TestArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: cargo test $($TestArgs -join ' ')" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Per-feature passes only run lib + doc tests (fast); the slow integration
# suite (tests/stat_tests.rs) doesn't vary by feature, so it's exercised once
# below with every feature enabled instead of once per combination.
Invoke-Test @("--no-default-features", "--lib")
Invoke-Test @("--no-default-features", "--doc")

foreach ($f in $features) {
    Invoke-Test @("--no-default-features", "--features", $f, "--lib")
    Invoke-Test @("--no-default-features", "--features", $f, "--doc")
}

Invoke-Test @("--all-features")

Write-Host "All feature combinations passed." -ForegroundColor Green
