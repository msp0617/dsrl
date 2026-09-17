param(
  [string]$Logs = (Join-Path $PSScriptRoot '..\..\logs\bundle_20260917\logs')
)
# Post-hoc (NOT pre-registered) "mid-late window mean" W = mean of the 7 evaluations at
# env 72,080 .. 102,128 (online 40,064 .. 70,096; 100 episodes each = 700 episodes per seed),
# computed to see whether a window metric has enough power for the section-26 questions.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$WIN = @(72080, 77088, 82096, 87104, 92112, 97120, 102128)
$arms = @('baseline','tent12','tent12r','tent12_hq','tent12i','tent12i_cap03','tent12i_cfix03','tent12i_cap1','tent12i_cfix1','tent12i_g099','tent12i_rs02','fixalpha_03','mix_prefill','iql','td')
$val = @{}
foreach ($g in $arms) {
  Get-ChildItem $Logs -Directory | ForEach-Object {
    if ($_.Name -notmatch "^square_${g}_s(\d+)$") { return }
    $s = [int]$Matches[1]
    $rows = Import-Csv (Join-Path $_.FullName 'eval_log.csv') | Where-Object { $_.deterministic -eq '0' }
    $pts = @()
    foreach ($e in $WIN) { $r = $rows | Where-Object { [math]::Abs([int]$_.env_steps - $e) -le 32 } | Select-Object -Last 1; if ($r) { $pts += [double]$r.success_rate } }
    if ($pts.Count -eq 7) { $val["$g|$s"] = ($pts | Measure-Object -Average).Average }
  }
}
function Stat($v) { $v = @($v); $m = ($v | Measure-Object -Average).Average; if ($v.Count -lt 2) { return @($m, [double]::NaN, [double]::NaN) }; $ss = 0; foreach ($x in $v) { $ss += ($x - $m) * ($x - $m) }; $sd = [math]::Sqrt($ss / ($v.Count - 1)); $se = $sd / [math]::Sqrt($v.Count); return @($m, $sd, $se) }
Write-Output "=== post-hoc window mean W (env 72,080..102,128, 7 evals, 700 ep/seed) ==="
Write-Output ("{0,-16} {1,2} {2,7} {3,7} {4,7}  per-seed" -f 'arm', 'n', 'mean', 'SD', 'SE')
foreach ($g in $arms) {
  $ks = @($val.Keys | Where-Object { $_ -like "$g|*" } | Sort-Object { [int](($_ -split '\|')[1]) })
  if ($ks.Count -eq 0) { continue }
  $v = $ks | ForEach-Object { $val[$_] }; $st = Stat $v
  Write-Output ("{0,-16} {1,2} {2,7:F3} {3,7:F3} {4,7:F3}  {5}" -f $g, $ks.Count, $st[0], $st[1], $st[2], (($v | ForEach-Object { '{0:F3}' -f $_ }) -join ' '))
}
Write-Output ""
Write-Output "=== paired contrasts on W (mean +- SE, t, pos/n) ==="
function PairW($a, $b, $seeds) {
  $d = @()
  foreach ($s in $seeds) { $ka = "$a|$s"; $kb = "$b|$s"; if ($val.ContainsKey($ka) -and $val.ContainsKey($kb)) { $d += ($val[$ka] - $val[$kb]) } }
  if ($d.Count -lt 2) { return ("{0} - {1}: n/a" -f $a, $b) }
  $st = Stat $d; $pos = @($d | Where-Object { $_ -gt 0 }).Count
  return ('{0} - {1}: {2:+0.000;-0.000} +- {3:F3} (t {4:F2}; {5}/{6}) [{7}]' -f $a, $b, $st[0], $st[2], ($st[0] / $st[2]), $pos, $d.Count, (($d | ForEach-Object { '{0:+0.000;-0.000}' -f $_ }) -join ' '))
}
foreach ($p in @(@('tent12r','tent12',@(1,2,3)), @('tent12i','tent12r',@(1,2,3)), @('tent12i','tent12',@(1,2,3,4,5)), @('tent12i_cap03','tent12i',@(1,2,3,4,5)), @('tent12i_cfix03','tent12i',@(1,2,3)), @('tent12i_cfix03','tent12i_cap03',@(1,2,3)), @('tent12i_cap03','tent12_hq',@(1,2,3)), @('tent12i_cfix03','tent12_hq',@(1,2,3)), @('tent12_hq','tent12',@(1,2,3)), @('tent12i_cfix1','tent12i_cap1',@(1,2,3)), @('tent12i_cap1','tent12i',@(1,2,3)), @('tent12i_g099','tent12i',@(1,2,3)), @('tent12i_cap03','mix_prefill',@(1,2,3)), @('tent12i_cap03','iql',@(1,2,3)))) {
  Write-Output (PairW $p[0] $p[1] $p[2])
}
