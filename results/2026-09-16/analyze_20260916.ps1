$ErrorActionPreference = 'Stop'
$LOGS = Join-Path $PSScriptRoot '..\..\logs\bundle_20260916\logs'   # unzip csv_bundle.zip (2026-09-16 01:07 KST) there
$OUT = $PSScriptRoot
$ROLLOUT = @{ can = 24016; square = 32016 }
$UNTIL = @{ can = 75984; square = 67984 }
$END = @{ can = 129152; square = 127136 }
$PRE = @{ can = 104144; square = 102128 }
$GRID = 5008
$GROUPS = @{
  square = @('baseline','tent12','tent12_hq','fixalpha_03','mix_prefill','iql','tent6','tent12i','tent12i_cap03','tent12i_cap1','tent12i_rs02','fixa015_rs02')
  can = @('baseline','tent12i','tent12i_hq','tent12i_cap03','fixalpha_03','fixalpha_01','mix_prefill','prefill_t12i','prefill_t12i_a015','prefill_fixa03','prefill_fixa015','calql_prefill','hardq')
}

function SE($vals) { if ($vals.Count -lt 2) { return [double]::NaN }; $m = ($vals | Measure-Object -Average).Average; $ss = 0; foreach ($v in $vals) { $ss += ($v - $m) * ($v - $m) }; return [math]::Sqrt($ss / ($vals.Count - 1)) / [math]::Sqrt($vals.Count) }
function Mean($vals) { if ($vals.Count -eq 0) { return [double]::NaN }; return ($vals | Measure-Object -Average).Average }
function F3($x) { if ([double]::IsNaN($x)) { return '  —  ' }; return ('{0,6:F3}' -f $x) }

# ---------------------------------------------------------------- eval
$runs = @{}
Get-ChildItem $LOGS -Directory | ForEach-Object {
  if ($_.Name -notmatch '^(can|square)_(.+)_s(\d+)$') { return }
  $task = $Matches[1]; $group = $Matches[2]; $seed = [int]$Matches[3]
  if ($GROUPS[$task] -notcontains $group) { return }
  $f = Join-Path $_.FullName 'eval_log.csv'
  if (-not (Test-Path $f)) { return }
  $rows = Import-Csv $f | Where-Object { $_.deterministic -eq '0' -or $_.deterministic -eq 'False' }
  $byEnv = @{}
  foreach ($r in $rows) { $byEnv[[int]$r.env_steps] = [double]$r.success_rate }   # keep=last on resume
  $pts = @()
  foreach ($e in ($byEnv.Keys | Sort-Object)) { $on = if ($e -eq 0) { 0 } else { $e - $ROLLOUT[$task] }; $pts += [pscustomobject]@{ env = $e; online = $on; s = $byEnv[$e] } }
  $runs["$task|$group|$seed"] = [pscustomobject]@{ task = $task; group = $group; seed = $seed; pts = $pts }
}

function AtEnv($run, $env) { $p = $run.pts | Where-Object { [math]::Abs($_.env - $env) -le 32 } | Select-Object -First 1; if ($p) { return $p.s }; return [double]::NaN }
function AtOnline($run, $on) { $p = $run.pts | Where-Object { $_.online -eq $on } | Select-Object -First 1; if ($p) { return $p.s }; return [double]::NaN }
function AUC($run) {
  $until = $UNTIL[$run.task]
  $p = @($run.pts | Where-Object { $_.online -le $until } | Sort-Object online)
  if ($p.Count -lt 2) { return [double]::NaN }
  $a = 0.0
  for ($i = 1; $i -lt $p.Count; $i++) { $a += ($p[$i].online - $p[$i-1].online) * ($p[$i].s + $p[$i-1].s) / 2.0 }
  $a += ($until - $p[-1].online) * $p[-1].s
  return $a / $until
}
function T80($run) { $p = $run.pts | Where-Object { $_.online -gt 0 -and $_.s -ge 0.8 } | Sort-Object online | Select-Object -First 1; if ($p) { return $p.online }; return [double]::NaN }

$perSeed = @()
foreach ($k in $runs.Keys) {
  $r = $runs[$k]; $t = $r.task
  $o = [ordered]@{ task = $t; group = $r.group; seed = $r.seed; n_evals = $r.pts.Count; step0 = (AtEnv $r 0); auc = (AUC $r)
    at5k = (AtOnline $r 5008); at10k = (AtOnline $r 10016); at15k = (AtOnline $r 15024); at20k = (AtOnline $r 20032)
    pre = (AtEnv $r $PRE[$t]); end = (AtEnv $r $END[$t]); t80 = (T80 $r) }
  $o.late = ($o.pre + $o.end) / 2.0
  $perSeed += [pscustomobject]$o
}
$perSeed | Sort-Object task, group, seed | Export-Csv (Join-Path $OUT 'per_seed.csv') -NoTypeInformation

# group summary + min of the seed-mean curve on the 5k grid (k>=1, within the early window)
$summary = @()
foreach ($t in @('square','can')) {
  foreach ($g in $GROUPS[$t]) {
    $rs = @($runs.Values | Where-Object { $_.task -eq $t -and $_.group -eq $g } | Sort-Object seed)
    if ($rs.Count -eq 0) { continue }
    $ps = @($perSeed | Where-Object { $_.task -eq $t -and $_.group -eq $g })
    $minv = [double]::PositiveInfinity; $minAt = -1
    for ($k = 1; $k * $GRID -le $UNTIL[$t]; $k++) {
      $on = $k * $GRID; $vals = @(); foreach ($r in $rs) { $v = AtOnline $r $on; if (-not [double]::IsNaN($v)) { $vals += $v } }
      if ($vals.Count -eq $rs.Count) { $m = Mean $vals; if ($m -lt $minv) { $minv = $m; $minAt = $on } }
    }
    $s13 = @($ps | Where-Object { $_.seed -in 1,2,3 })
    $summary += [pscustomobject]@{ task = $t; group = $g; n = $rs.Count; seeds = (($rs | ForEach-Object { $_.seed }) -join ',')
      step0 = (Mean ($ps.step0)); at5k = (Mean ($ps.at5k)); at10k = (Mean ($ps.at10k)); at10k_se = (SE ($ps.at10k)); at15k = (Mean ($ps.at15k)); at20k = (Mean ($ps.at20k))
      auc = (Mean ($ps.auc)); auc_se = (SE ($ps.auc)); min_grid = $minv; min_at = $minAt
      pre = (Mean ($ps.pre)); end = (Mean ($ps.end)); end_se = (SE ($ps.end)); late = (Mean ($ps.late)); late_se = (SE ($ps.late))
      t80_n = (@($ps | Where-Object { -not [double]::IsNaN($_.t80) })).Count
      s13_at10k = (Mean ($s13.at10k)); s13_auc = (Mean ($s13.auc)); s13_end = (Mean ($s13.end)); s13_late = (Mean ($s13.late)); s13_at5k = (Mean ($s13.at5k)) }
  }
}
$summary | Export-Csv (Join-Path $OUT 'groups.csv') -NoTypeInformation

Write-Output "=== SQUARE (42k = env 42,032 = online 10,016; early AUC online 0-67,984; min = seed-mean on 5k grid; end 127,136; pre 102,128) ==="
Write-Output ("{0,-14} {1,2} {2,6} {3,6} {4,6} {5,12} {6,6} {7,6} {8,6} | s1-3: 42k {9} auc {10} end {11} late {12}" -f 'group','n','step0','42k','AUC','min@online','102k','127k','late','','','','')
foreach ($s in ($summary | Where-Object task -eq 'square')) {
  Write-Output ("{0,-14} {1,2} {2} {3} {4} {5}@{6,-6} {7} {8} {9} | {10} {11} {12} {13}" -f $s.group, $s.n, (F3 $s.step0), (F3 $s.at10k), (F3 $s.auc), (F3 $s.min_grid), $s.min_at, (F3 $s.pre), (F3 $s.end), (F3 $s.late), (F3 $s.s13_at10k), (F3 $s.s13_auc), (F3 $s.s13_end), (F3 $s.s13_late))
}
Write-Output ""
Write-Output "=== CAN (online 5k/10k/15k/20k; early AUC online 0-75,984; end 129,152; pre 104,144; T80 = seeds reaching 0.8) ==="
Write-Output ("{0,-18} {1,2} {2,6} {3,6} {4,6} {5,6} {6,6} {7,6} {8,12} {9,6} {10,6} {11,6} {12,4}" -f 'group','n','step0','5k','10k','15k','20k','AUC','min@online','104k','129k','late','T80')
foreach ($s in ($summary | Where-Object task -eq 'can')) {
  Write-Output ("{0,-18} {1,2} {2} {3} {4} {5} {6} {7} {8}@{9,-6} {10} {11} {12} {13}/{14}" -f $s.group, $s.n, (F3 $s.step0), (F3 $s.at5k), (F3 $s.at10k), (F3 $s.at15k), (F3 $s.at20k), (F3 $s.auc), (F3 $s.min_grid), $s.min_at, (F3 $s.pre), (F3 $s.end), (F3 $s.late), $s.t80_n, $s.n)
}
Write-Output ""
Write-Output "=== per-seed (new arms) ==="
foreach ($p in ($perSeed | Where-Object { $_.group -match 'tent12i$|cap03|cap1|rs02|fixa015' } | Sort-Object task, group, seed)) {
  Write-Output ("{0}_{1}_s{2}: step0 {3} 5k {4} 10k {5} 15k {6} 20k {7} auc {8} pre {9} end {10} t80 {11}" -f $p.task, $p.group, $p.seed, (F3 $p.step0), (F3 $p.at5k), (F3 $p.at10k), (F3 $p.at15k), (F3 $p.at20k), (F3 $p.auc), (F3 $p.pre), (F3 $p.end), $p.t80)
}

# seed-matched contrasts (s1-3)
function Contrast($t, $a, $b, $cols) {
  $out = @()
  foreach ($c in $cols) {
    $d = @()
    foreach ($s in 1,2,3) {
      $pa = $perSeed | Where-Object { $_.task -eq $t -and $_.group -eq $a -and $_.seed -eq $s }
      $pb = $perSeed | Where-Object { $_.task -eq $t -and $_.group -eq $b -and $_.seed -eq $s }
      if ($pa -and $pb -and -not [double]::IsNaN($pa.$c) -and -not [double]::IsNaN($pb.$c)) { $d += ($pa.$c - $pb.$c) }
    }
    if ($d.Count -eq 3) { $pos = @($d | Where-Object { $_ -gt 0 }).Count; $out += ('{0} {1:+0.000;-0.000} ± {2:F3} ({3}/3)' -f $c, (Mean $d), (SE $d), $pos) } else { $out += "$c n/a" }
  }
  return ("{0} − {1}: " -f $a, $b) + ($out -join ' | ')
}
Write-Output ""
Write-Output "=== seed-matched contrasts, Square (s1-3) ==="
foreach ($a in @('tent12i_cap03','tent12i_cap1','tent12i_rs02','fixa015_rs02','tent12i')) {
  foreach ($b in @('tent12','tent12_hq','baseline','tent12i')) { if ($a -ne $b) { Write-Output (Contrast 'square' $a $b @('at10k','auc','pre','end','late')) } }
}
Write-Output "tent12_hq − tent12 (check vs 9/8: end +0.093 ± 0.020, auc −0.050 ± 0.040): " + (Contrast 'square' 'tent12_hq' 'tent12' @('at10k','auc','end','late'))
Write-Output ""
Write-Output "=== seed-matched contrasts, Can (s1-3) ==="
foreach ($pair in @(@('tent12i_cap03','tent12i'), @('tent12i_cap03','tent12i_hq'), @('tent12i_cap03','fixalpha_03'), @('prefill_fixa015','prefill_fixa03'), @('prefill_fixa015','prefill_t12i_a015'), @('prefill_fixa015','fixalpha_01'), @('prefill_fixa015','fixalpha_03'), @('prefill_fixa015','prefill_t12i'))) {
  Write-Output (Contrast 'can' $pair[0] $pair[1] @('at5k','at10k','at15k','auc','end','late'))
}

# ---------------------------------------------------------------- diagnostics (train_log)
Write-Output ""
Write-Output "=== diagnostics: seed-mean per 5k online bin (bin = floor(online/5000)*5000): alpha | critic alpha | Q_W | logp | |mu| ==="
$diagGroups = @(@('square','tent12'), @('square','tent12_hq'), @('square','tent12i'), @('square','tent12i_cap03'), @('square','tent12i_cap1'), @('square','tent12i_rs02'), @('square','fixa015_rs02'), @('square','fixalpha_03'), @('can','tent12i'), @('can','tent12i_cap03'), @('can','prefill_fixa015'), @('can','prefill_fixa03'), @('can','prefill_t12i_a015'))
$showBins = @(0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 70000, 90000, 110000)
$diagRows = @()
foreach ($dg in $diagGroups) {
  $t = $dg[0]; $g = $dg[1]
  $dirs = Get-ChildItem $LOGS -Directory | Where-Object { $_.Name -match "^${t}_${g}_s(\d+)$" }
  if ($dirs.Count -eq 0) { continue }
  $bins = @{}   # bin -> list of per-seed means
  $onset = @(); $bindFrac = @(); $amax = @(); $qmax = @()
  foreach ($d in $dirs) {
    $f = Join-Path $d.FullName 'train_log.csv'; if (-not (Test-Path $f)) { continue }
    $rows = Import-Csv $f
    $seedBins = @{}
    $firstBind = [double]::NaN; $nb = 0; $nr = 0; $am = 0; $qm = [double]::NegativeInfinity
    foreach ($r in $rows) {
      $env = [int]$r.env_steps; $on = $env - $ROLLOUT[$t]; if ($on -lt 0) { continue }
      $b = [int]([math]::Floor($on / 5000) * 5000)
      if (-not $seedBins.ContainsKey($b)) { $seedBins[$b] = @{ a = @(); c = @(); q = @(); l = @(); m = @() } }
      $a = [double]$r.ent_coef; $seedBins[$b].a += $a
      $q = [double]$r.qw_mean; $seedBins[$b].q += $q
      $seedBins[$b].l += [double]$r.logp_mean; $seedBins[$b].m += [double]$r.mu_absmean
      if ($a -gt $am) { $am = $a }; if ($q -gt $qm) { $qm = $q }
      $nr++
      if ($r.PSObject.Properties['critic_ent_coef'] -and $r.critic_ent_coef -ne '' -and $r.critic_ent_coef -ne 'nan') {
        $c = [double]$r.critic_ent_coef; $seedBins[$b].c += $c
        if ($c -lt $a - 1e-6) { $nb++; if ([double]::IsNaN($firstBind)) { $firstBind = $on } }
      }
    }
    foreach ($b in $seedBins.Keys) {
      if (-not $bins.ContainsKey($b)) { $bins[$b] = @{ a = @(); c = @(); q = @(); l = @(); m = @() } }
      $bins[$b].a += (Mean $seedBins[$b].a); $bins[$b].q += (Mean $seedBins[$b].q); $bins[$b].l += (Mean $seedBins[$b].l); $bins[$b].m += (Mean $seedBins[$b].m)
      if ($seedBins[$b].c.Count -gt 0) { $bins[$b].c += (Mean $seedBins[$b].c) }
    }
    $onset += $firstBind; $bindFrac += ($nb / [math]::Max($nr, 1)); $amax += $am; $qmax += $qm
  }
  $line = "{0}_{1} (n={2}): " -f $t, $g, $dirs.Count
  foreach ($b in $showBins) {
    if ($bins.ContainsKey($b)) {
      $ca = if ($bins[$b].c.Count -gt 0) { '{0:F2}' -f (Mean $bins[$b].c) } else { '-' }
      $line += ("[{0}k a{1:F2}/c{2} Q{3:F0} H{4:F1} mu{5:F2}] " -f ($b/1000), (Mean $bins[$b].a), $ca, (Mean $bins[$b].q), (-(Mean $bins[$b].l)), (Mean $bins[$b].m))
      $diagRows += [pscustomobject]@{ task = $t; group = $g; bin = $b; alpha = (Mean $bins[$b].a); critic_alpha = $ca; qw = (Mean $bins[$b].q); H = (-(Mean $bins[$b].l)); mu = (Mean $bins[$b].m) }
    }
  }
  Write-Output $line
  Write-Output ("    alpha max per seed: {0} | Q_W max per seed: {1} | cap-binding onset (online): {2} | binding fraction of rows: {3}" -f (($amax | ForEach-Object { '{0:F2}' -f $_ }) -join '/'), (($qmax | ForEach-Object { '{0:F0}' -f $_ }) -join '/'), (($onset | ForEach-Object { if ([double]::IsNaN($_)) { 'never' } else { $_ } }) -join '/'), (($bindFrac | ForEach-Object { '{0:F2}' -f $_ }) -join '/'))
}
$diagRows | Export-Csv (Join-Path $OUT 'diag_bins.csv') -NoTypeInformation
Write-Output "written: per_seed.csv, groups.csv, diag_bins.csv in $OUT"
