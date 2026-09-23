# Counts differing pixels between two PNGs. Windows PowerShell only.
param([string]$A, [string]$B)

Add-Type -AssemblyName System.Drawing

$ia = [System.Drawing.Bitmap]::FromFile($A)
$ib = [System.Drawing.Bitmap]::FromFile($B)
if ($ia.Width -ne $ib.Width -or $ia.Height -ne $ib.Height) { "size mismatch"; exit 1 }

$diff = 0
$total = 0
$maxDelta = 0
# Sample a grid to stay fast on 2560x1440.
for ($y = 0; $y -lt $ia.Height; $y += 4) {
    for ($x = 0; $x -lt $ia.Width; $x += 4) {
        $pa = $ia.GetPixel($x, $y)
        $pb = $ib.GetPixel($x, $y)
        $d = [Math]::Abs($pa.R - $pb.R) + [Math]::Abs($pa.G - $pb.G) + [Math]::Abs($pa.B - $pb.B)
        $total++
        if ($d -gt 8) { $diff++ }
        if ($d -gt $maxDelta) { $maxDelta = $d }
    }
}
$ia.Dispose(); $ib.Dispose()
"sampled=$total differing=$diff maxDelta=$maxDelta pct=$([Math]::Round(100.0*$diff/$total,2))"
