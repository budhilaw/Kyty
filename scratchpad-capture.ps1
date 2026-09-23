# Captures the kyty_emulator window (or the whole screen) to a PNG.
# Run with Windows PowerShell - PowerShell 7 lacks System.Drawing.
param(
    [string]$Out = "C:\Users\Ericsson\Dev\Personal\Other\KytyPS5\_Build\windows\install\_shot.png",
    [switch]$Screen
)

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class WinRect {
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

$proc = Get-Process kyty_emulator -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $proc) { "no emulator process"; exit 1 }

[void][WinRect]::SetForegroundWindow($proc.MainWindowHandle)
Start-Sleep -Milliseconds 800

if ($Screen) {
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $x = $b.X; $y = $b.Y; $w = $b.Width; $h = $b.Height
} else {
    $rect = New-Object WinRect+RECT
    [void][WinRect]::GetWindowRect($proc.MainWindowHandle, [ref]$rect)
    $x = $rect.Left; $y = $rect.Top
    $w = $rect.Right - $rect.Left
    $h = $rect.Bottom - $rect.Top
}
if ($w -le 0 -or $h -le 0) { "bad rect"; exit 1 }

$bmp = New-Object System.Drawing.Bitmap $w, $h
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($x, $y, 0, 0, (New-Object System.Drawing.Size $w, $h))
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$gfx.Dispose()
$bmp.Dispose()
"captured ${w}x${h} -> $Out"
