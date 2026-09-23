# Sets up the KytyPS5 build environment in the current PowerShell session.
$ErrorActionPreference = 'Stop'

$VsPath   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
$VcVars   = "$VsPath\VC\Auxiliary\Build\vcvars64.bat"
$NinjaDir = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe"
$CMakeDir = 'C:\Program Files\CMake\bin'
$VulkanSDK= (Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
$QtDir    = 'C:\Qt\6.10.3\msvc2022_64'

# Import the MSVC x64 environment (INCLUDE/LIB/PATH) into this session.
cmd /c "`"$VcVars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

$env:PATH = "$NinjaDir;$CMakeDir;$VulkanSDK\Bin;$QtDir\bin;$env:PATH"
$env:VULKAN_SDK = $VulkanSDK
$env:Qt6_DIR    = $QtDir

Write-Host "KytyPS5 build environment ready:" -ForegroundColor Green
foreach ($t in 'cmake','ninja','clang-cl','glslangValidator') {
    $src = (Get-Command $t -EA SilentlyContinue).Source
    Write-Host ("  {0,-18} {1}" -f $t, $(if ($src) { $src } else { 'NOT FOUND' }))
}
Write-Host ("  {0,-18} {1}" -f 'Qt6_DIR', $env:Qt6_DIR)
