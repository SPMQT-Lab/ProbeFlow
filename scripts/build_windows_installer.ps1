$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = if ($env:PROBEFLOW_BUILD_ROOT) {
    $env:PROBEFLOW_BUILD_ROOT
} else {
    Join-Path $Root "build\windows"
}
$Python = Join-Path $BuildRoot "venv\Scripts\python.exe"
$App = Join-Path $BuildRoot "dist\ProbeFlow"
$Nsi = Join-Path $Root "packaging\windows\ProbeFlow.nsi"
$Icon = Join-Path $Root "packaging\windows\ProbeFlow.ico"
$License = Join-Path $Root "LICENSE"

if (-not $IsWindows) { throw "The Windows installer must be built on Windows." }
if (-not (Test-Path $App -PathType Container)) {
    throw "The packaged Windows app is missing: $App"
}
if (-not (Test-Path $Python -PathType Leaf)) {
    throw "The Windows build environment is missing: $Python"
}

$Version = (& $Python -c "import probeflow; print(probeflow.__version__)").Trim()
$ArtifactVersion = $Version -replace "rc", "-rc"
$ProductVersion = $Version -replace "rc", " RC "
$FileVersion = "1.0.0.1"
$InstallerName = "ProbeFlow-$ArtifactVersion-Windows-x64-Setup.exe"
$PortableName = "ProbeFlow-$ArtifactVersion-Windows-x64-portable.zip"
$Installer = Join-Path (Join-Path $BuildRoot "dist") $InstallerName
$Portable = Join-Path (Join-Path $BuildRoot "dist") $PortableName

$MakeNsisCandidates = @(
    @(
        $env:PROBEFLOW_MAKENSIS,
        (Join-Path ${env:ProgramFiles(x86)} "NSIS\makensis.exe"),
        (Join-Path $env:ProgramFiles "NSIS\makensis.exe")
    ) | Where-Object { $_ -and (Test-Path $_ -PathType Leaf) }
)
if (-not $MakeNsisCandidates) {
    throw "makensis.exe was not found; install NSIS 3.12.0 first."
}
$MakeNsis = $MakeNsisCandidates[0]

Remove-Item -Force -ErrorAction SilentlyContinue $Installer, "$Installer.sha256", $Portable, "$Portable.sha256"
Write-Host "Building $InstallerName with $MakeNsis"
& $MakeNsis `
    "/V3" `
    "/DSOURCE_DIR=$App" `
    "/DOUTPUT_FILE=$Installer" `
    "/DICON_FILE=$Icon" `
    "/DLICENSE_FILE=$License" `
    "/DPRODUCT_VERSION=$ProductVersion" `
    "/DFILE_VERSION=$FileVersion" `
    $Nsi
if ($LASTEXITCODE -ne 0) { throw "NSIS failed to build the ProbeFlow installer." }

Compress-Archive -Path $App -DestinationPath $Portable -CompressionLevel Optimal
foreach ($Artifact in @($Installer, $Portable)) {
    $Hash = (Get-FileHash -Algorithm SHA256 $Artifact).Hash.ToLowerInvariant()
    "$Hash  $(Split-Path $Artifact -Leaf)" | Set-Content -Encoding ascii "$Artifact.sha256"
}

$InstallRoot = Join-Path $BuildRoot "installer-smoke\ProbeFlow"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Split-Path $InstallRoot -Parent)
New-Item -ItemType Directory -Force -Path (Split-Path $InstallRoot -Parent) | Out-Null
$Install = Start-Process `
    -FilePath $Installer `
    -ArgumentList @("/S", "/D=$InstallRoot") `
    -Wait `
    -PassThru
if ($Install.ExitCode -ne 0) {
    throw "Silent installer smoke test failed with exit code $($Install.ExitCode)."
}

& $Python (Join-Path $Root "scripts\validate_windows_app.py") $InstallRoot
if ($LASTEXITCODE -ne 0) { throw "The installed ProbeFlow bundle failed validation." }
$Smoke = Start-Process `
    -FilePath (Join-Path $InstallRoot "ProbeFlow.exe") `
    -ArgumentList "--smoke-test" `
    -Wait `
    -PassThru
if ($Smoke.ExitCode -ne 0) {
    throw "The installed ProbeFlow smoke test failed with exit code $($Smoke.ExitCode)."
}

$Uninstall = Start-Process `
    -FilePath (Join-Path $InstallRoot "Uninstall ProbeFlow.exe") `
    -ArgumentList "/S" `
    -Wait `
    -PassThru
if ($Uninstall.ExitCode -ne 0) {
    throw "Silent uninstall test failed with exit code $($Uninstall.ExitCode)."
}
for ($Attempt = 0; $Attempt -lt 50 -and (Test-Path $InstallRoot); $Attempt++) {
    Start-Sleep -Milliseconds 200
}
if (Test-Path $InstallRoot) { throw "The installer smoke-test directory was not removed." }

Write-Host "Built and verified $Installer"
Write-Host "Built portable archive $Portable"
