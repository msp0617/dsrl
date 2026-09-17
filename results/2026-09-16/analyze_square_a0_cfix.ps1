param(
  # Folder that holds the run directories (<task>_<group>_s<seed>/eval_log.csv, train_log.csv).
  # Unzip the Drive csv_bundle.zip into logs/bundle_<date>/ and point here.
  [string]$Logs = (Join-Path $PSScriptRoot '..\..\logs\bundle_20260916\logs'),
  [string]$Out = $PSScriptRoot
)
# HANDOFF 26 (2026-09-16) pre-registered analysis for VM SA (vm_square_a0: square_tent12r s1-3,
# square_tent12i s4-5, square_tent12i_cap03 s4-5) and VM SB (vm_square_cfix: square_tent12i_cfix03,
# _cfix1, _g099 s1-3). Same metric definitions as analyze_20260916.ps1 (results/2026-09-08/README.md 2):
# online = env - rollout, initial evaluation at online 0; early AUC = trapezoid over online 0..67,984
# (Square) with the last value held to the cut-off, / window; min = min of the seed-mean curve on the
# 5,008 grid inside the early window; late = mean of env 102,128 and 127,136. +- = SD/sqrt(n).
# Decision rules (26): n=3 paired difference = |t| > 4.303 and 3/3; n=5 = |t| > 2.776 and >= 4/5 same
# sign; equivalence margins 0.05 (Q1a) and 0.075 (Q2b) always carry "(suggestive)".
$ErrorActionPreference = 'Stop'
$ROLLOUT = @{ can = 24016; square = 32016 }
$UNTIL = @{ can = 75984; square = 67984 }
$END = @{ can = 129152; square = 127136 }
$PRE = @{ can = 104144; square = 102128 }
$GRID = 5008
$GROUPS = @{
  square = @('baseline','tent12','tent12r','tent12_hq','fixalpha_03','mix_prefill','iql','td','tent12i','tent12i_cap03','tent12i_cap1','tent12i_cfix03','tent12i_cfix1','tent12i_g099','tent12i_rs02','fixa015_rs02')
  can = @('baseline','tent12i','tent12i_hq','tent12i_cap03','fixalpha_03','prefill_fixa015')
}

function SE($vals) { $v = @($vals); if ($v.Count -lt 2) { return [double]::NaN }; $m = ($v | Measure-Object -Average).Average; $ss = 0; foreach ($x in $v) { $ss += ($x - $m) * ($x - $m) }; return [math]::Sqrt($ss / ($v.Count - 1)) / [math]::Sqrt($v.Count) }
function Mean($vals) { $v = @($vals); if ($v.Count -eq 0) { return [double]::NaN }; return ($v | Measure-Object -Average).Average }
function F3($x) { if ($null -eq $x -or [double]::IsNaN($x)) { return '  -   ' }; return ('{0,6:F3}' -f $x) }

# ---------------------------------------------------------------- eval
$runs = @{}
Get-ChildItem $Logs -Directory | ForEach-Object {
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
  $runs["$task|$group|$seed"] = [pscustomobject]@{ task = $task; group = $group; seed = $seed; pts = $pts; dir = $_.FullName }
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

$perSeed = @()
foreach ($k in $runs.Keys) {
  $r = $runs[$k]; $t = $r.task
  $o = [ordered]@{ task = $t; group = $r.group; seed = $r.seed; n_evals = $r.pts.Count; last_env = ($r.pts | Measure-Object env -Maximum).Maximum
    step0 = (AtEnv $r 0); auc = (AUC $r); at5k = (AtOnline $r 5008); at10k = (AtOnline $r 10016); at15k = (AtOnline $r 15024); at20k = (AtOnline $r 20032)
    at62k = (AtEnv $r 97120); pre = (AtEnv $r $PRE[$t]); end = (AtEnv $r $END[$t]) }
  $o.late = ($o.pre + $o.end) / 2.0
  $perSeed += [pscustomobject]$o
}
$perSeed | Sort-Object task, group, seed | Export-Csv (Join-Path $Out 'per_seed.csv') -NoTypeInformation

function SeedRow($t, $g, $s) { return ($perSeed | Where-Object { $_.task -eq $t -and $_.group -eq $g -and $_.seed -eq $s } | Select-Object -First 1) }

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
    $summary += [pscustomobject]@{ task = $t; group = $g; n = $rs.Count; seeds = (($rs | ForEach-Object { $_.seed }) -join ',')
      step0 = (Mean ($ps.step0)); at10k = (Mean ($ps.at10k)); at10k_se = (SE ($ps.at10k)); auc = (Mean ($ps.auc)); auc_se = (SE ($ps.auc))
      min_grid = $minv; min_at = $minAt; at62k = (Mean ($ps.at62k)); pre = (Mean ($ps.pre)); end = (Mean ($ps.end)); end_se = (SE ($ps.end)); late = (Mean ($ps.late)); late_se = (SE ($ps.late)) }
  }
}
$summary | Export-Csv (Join-Path $Out 'groups.csv') -NoTypeInformation

Write-Output "=== groups (Square: 42k = env 42,032 = online 10,016; AUC online 0-67,984; min = seed-mean on 5k grid; 62k = env 97,120; end 127,136; late = mean(102k,127k)) ==="
Write-Output ("{0,-16} {1,2} {2,-10} {3,6} {4,6} {5,6} {6,13} {7,6} {8,6} {9,6} {10,6}" -f 'group','n','seeds','step0','42k','AUC','min@online','62k','102k','127k','late')
foreach ($s in $summary) {
  Write-Output ("{0,-16} {1,2} {2,-10} {3} {4} {5} {6}@{7,-6} {8} {9} {10} {11}" -f "$($s.task)_$($s.group)", $s.n, $s.seeds, (F3 $s.step0), (F3 $s.at10k), (F3 $s.auc), (F3 $s.min_grid), $s.min_at, (F3 $s.at62k), (F3 $s.pre), (F3 $s.end), (F3 $s.late))
}
Write-Output ""
Write-Output "=== per-seed, 26 arms ==="
foreach ($p in ($perSeed | Where-Object { $_.task -eq 'square' -and $_.group -in @('tent12','tent12r','tent12i','tent12i_cap03','tent12i_cap1','tent12i_cfix03','tent12i_cfix1','tent12i_g099') } | Sort-Object group, seed)) {
  Write-Output ("{0,-16} s{1}: evals {2,2} last_env {3,6} step0 {4} 42k {5} auc {6} 62k {7} 102k {8} 127k {9} late {10}" -f $p.group, $p.seed, $p.n_evals, $p.last_env, (F3 $p.step0), (F3 $p.at10k), (F3 $p.auc), (F3 $p.at62k), (F3 $p.pre), (F3 $p.end), (F3 $p.late))
}

# ---------------------------------------------------------------- paired contrasts with the 26 rules
function Paired($t, $a, $b, $col, $seeds) {
  $d = @(); $used = @()
  foreach ($s in $seeds) {
    $pa = SeedRow $t $a $s; $pb = SeedRow $t $b $s
    if ($pa -and $pb -and -not [double]::IsNaN($pa.$col) -and -not [double]::IsNaN($pb.$col)) { $d += ($pa.$col - $pb.$col); $used += $s }
  }
  if ($d.Count -lt 2) { return $null }
  $m = Mean $d; $se = SE $d; $tstat = if ($se -gt 0) { $m / $se } else { [double]::NaN }
  $pos = @($d | Where-Object { $_ -gt 0 }).Count; $neg = @($d | Where-Object { $_ -lt 0 }).Count
  $n = $d.Count; $crit = if ($n -eq 3) { 4.303 } elseif ($n -eq 5) { 2.776 } elseif ($n -eq 4) { 3.182 } elseif ($n -eq 2) { 12.706 } else { 2.0 }
  $need = if ($n -eq 3) { 3 } elseif ($n -eq 5) { 4 } else { $n }
  $pass = ([math]::Abs($tstat) -gt $crit) -and (($pos -ge $need) -or ($neg -ge $need))
  return [pscustomobject]@{ a = $a; b = $b; col = $col; n = $n; seeds = ($used -join ','); mean = $m; se = $se; t = $tstat; pos = $pos; neg = $neg; diffs = $d; pass = $pass; crit = $crit }
}
function Fmt($c) { if ($null -eq $c) { return 'n/a' }; $ds = (($c.diffs | ForEach-Object { '{0:+0.000;-0.000}' -f $_ }) -join ' '); return ('{0:+0.000;-0.000} ± {1:F3} (t {2:F2}; +{3}/-{4} of {5}; seeds {7}) [{6}]' -f $c.mean, $c.se, $c.t, $c.pos, $c.neg, $c.n, $ds, $c.seeds) }

$verdicts = [ordered]@{}
Write-Output ""
Write-Output "=== Q1a batch reproduction: tent12r - tent12 (s1-3), 127k and late; rule: both |mean| <= 0.05 and neither (3/3 & |t|>4.303) -> 'no batch effect (suggestive)'; either (3/3 & |t|>4.303) or |mean| > 0.10 -> 'batch effect'; else undecided ==="
$q1a_end = Paired 'square' 'tent12r' 'tent12' 'end' @(1,2,3); $q1a_late = Paired 'square' 'tent12r' 'tent12' 'late' @(1,2,3)
Write-Output ("127k: " + (Fmt $q1a_end)); Write-Output ("late: " + (Fmt $q1a_late))
if ($q1a_end -and $q1a_late) {
  $strong = $q1a_end.pass -or $q1a_late.pass
  $big = ([math]::Abs($q1a_end.mean) -gt 0.10) -or ([math]::Abs($q1a_late.mean) -gt 0.10)
  $small = ([math]::Abs($q1a_end.mean) -le 0.05) -and ([math]::Abs($q1a_late.mean) -le 0.05)
  $verdicts.Q1a = if ($strong -or $big) { 'batch effect' } elseif ($small -and -not $strong) { 'no batch effect (suggestive)' } else { 'undecided' }
} else { $verdicts.Q1a = 'not run' }
Write-Output ("Q1a -> " + $verdicts.Q1a)
Write-Output ("diagnostic reproduction (descriptive only): tent12r online 110k bin alpha >= 10 and Q_W >= 10k expected — see diagnostics block")

Write-Output ""
Write-Output "=== Q1b alpha-init effect n=5: tent12i - tent12 (s1-5) 127k and late; df=4 rule |t|>2.776 & >=4/5 same sign on 127k or late with the other same sign -> 'alpha-init effect confirmed' (only if Q1a = no batch effect); else 'suggestive' ==="
$q1b_end = Paired 'square' 'tent12i' 'tent12' 'end' @(1,2,3,4,5); $q1b_late = Paired 'square' 'tent12i' 'tent12' 'late' @(1,2,3,4,5)
Write-Output ("127k: " + (Fmt $q1b_end)); Write-Output ("late: " + (Fmt $q1b_late))
$q1b_same = ($q1b_end -and $q1b_late) -and (($q1b_end.mean -gt 0) -eq ($q1b_late.mean -gt 0))
$q1b_pass = $q1b_same -and (($q1b_end.pass) -or ($q1b_late.pass))
$verdicts.Q1b = if ($verdicts.Q1a -ne 'no batch effect (suggestive)') { "label 'confirmed' not allowed (Q1a = $($verdicts.Q1a)); report the same-night pair tent12i s1-3 - tent12r s1-3 and s4-5 descriptively" } elseif ($q1b_pass) { 'alpha-init effect confirmed (n=5)' } else { 'suggestive (as n=3)' }
Write-Output ("Q1b -> " + $verdicts.Q1b)
Write-Output ("same-night pair tent12i - tent12r (s1-3): 127k " + (Fmt (Paired 'square' 'tent12i' 'tent12r' 'end' @(1,2,3))) + " | late " + (Fmt (Paired 'square' 'tent12i' 'tent12r' 'late' @(1,2,3))))
Write-Output ("s4-5 pair tent12i - tent12 (9/6 s4-5): 127k " + (Fmt (Paired 'square' 'tent12i' 'tent12' 'end' @(4,5))) + " | late " + (Fmt (Paired 'square' 'tent12i' 'tent12' 'late' @(4,5))))

Write-Output ""
Write-Output "=== Q1c cap-specific effect n=5: cap03 - tent12i (s1-5) 127k, late, 102k; df=4 rule on 127k or late (other same sign) -> 'cap-specific effect'; else both 127k & late >=4/5 same direction & mean>0 -> 'suggestive'; else 'none' ==="
$q1c_end = Paired 'square' 'tent12i_cap03' 'tent12i' 'end' @(1,2,3,4,5); $q1c_late = Paired 'square' 'tent12i_cap03' 'tent12i' 'late' @(1,2,3,4,5); $q1c_pre = Paired 'square' 'tent12i_cap03' 'tent12i' 'pre' @(1,2,3,4,5)
Write-Output ("127k: " + (Fmt $q1c_end)); Write-Output ("late: " + (Fmt $q1c_late)); Write-Output ("102k: " + (Fmt $q1c_pre))
if ($q1c_end -and $q1c_late) {
  $same = (($q1c_end.mean -gt 0) -eq ($q1c_late.mean -gt 0))
  $verdicts.Q1c = if ($same -and ($q1c_end.pass -or $q1c_late.pass)) { 'cap-specific effect (n=5)' } elseif (($q1c_end.pos -ge ($q1c_end.n - [int]($q1c_end.n -eq 5))) -and ($q1c_late.pos -ge ($q1c_late.n - [int]($q1c_late.n -eq 5))) -and ($q1c_end.mean -gt 0) -and ($q1c_late.mean -gt 0)) { 'suggestive' } else { 'none' }
} else { $verdicts.Q1c = 'not run' }
Write-Output ("Q1c -> " + $verdicts.Q1c)

Write-Output ""
Write-Output "=== Q2 main: cfix1 - cap1 (s1-3) 127k; (a) mean >= +0.15 & 3/3 -> 'binding shock supported'; (b) mean <= +0.075 -> 'level effect (suggestive)'; (c) else undecided. late classified the same; cross (a/b) -> undecided. Q1a dependency: batch effect -> (a) threshold += |tent12r-tent12 127k diff|, (b) -> undecided; Q1a undecided -> (b) -> undecided ==="
$q2_end = Paired 'square' 'tent12i_cfix1' 'tent12i_cap1' 'end' @(1,2,3); $q2_late = Paired 'square' 'tent12i_cfix1' 'tent12i_cap1' 'late' @(1,2,3)
Write-Output ("127k: " + (Fmt $q2_end)); Write-Output ("late: " + (Fmt $q2_late))
function Q2Class($c, $thrA) { if ($null -eq $c) { return 'not run' }; if (($c.mean -ge $thrA) -and ($c.pos -eq 3)) { return 'a' }; if ($c.mean -le 0.075) { return 'b' }; return 'c' }
$thrA = 0.15
if ($verdicts.Q1a -eq 'batch effect' -and $q1a_end) { $thrA = 0.15 + [math]::Abs($q1a_end.mean) }
$cEnd = Q2Class $q2_end $thrA; $cLate = Q2Class $q2_late $thrA
$q2 = if ($cEnd -eq 'not run') { 'not run' } elseif (($cEnd -eq 'a' -and $cLate -eq 'b') -or ($cEnd -eq 'b' -and $cLate -eq 'a')) { 'undecided (127k/late cross)' } elseif ($cEnd -eq 'a') { 'binding shock supported' } elseif ($cEnd -eq 'b') { 'level effect (suggestive)' } else { 'undecided' }
if ($q2 -eq 'level effect (suggestive)' -and $verdicts.Q1a -ne 'no batch effect (suggestive)') { $q2 = "undecided (level-effect reading demoted: Q1a = $($verdicts.Q1a))" }
$verdicts.Q2 = "$q2 [127k class $cEnd, late class $cLate, (a) threshold $thrA]"
Write-Output ("Q2 -> " + $verdicts.Q2)
$cfix1_end = Mean (@($perSeed | Where-Object { $_.task -eq 'square' -and $_.group -eq 'tent12i_cfix1' -and $_.seed -le 3 }).end)
Write-Output ("aux (only if a): cfix1 127k mean {0} vs 0.472 (baseline s1-3) -> {1}" -f (F3 $cfix1_end), $(if ([double]::IsNaN($cfix1_end)) { 'n/a' } elseif ($cfix1_end -ge 0.472) { 'all of the cap1 loss is binding shock' } else { 'not all' }))
Write-Output ""
Write-Output "=== Q2 sanity: cfix03 - cap03 (s1-3) late (record only; expected |mean| <= 0.075; 3/3 & |t|>4.303 -> note that cap03's unbound first 10k changed the result) ==="
$q2s = Paired 'square' 'tent12i_cfix03' 'tent12i_cap03' 'late' @(1,2,3)
Write-Output ("late: " + (Fmt $q2s) + " | 127k: " + (Fmt (Paired 'square' 'tent12i_cfix03' 'tent12i_cap03' 'end' @(1,2,3))))
$verdicts.Q2sanity = if ($null -eq $q2s) { 'not run' } elseif ($q2s.pass) { 'DIFFERENT (3/3 & |t|>4.303): the unbound first 10k of cap03 matters' } elseif ([math]::Abs($q2s.mean) -le 0.075) { 'as expected (|mean| <= 0.075)' } else { 'larger than expected, not significant' }
Write-Output ("Q2 sanity -> " + $verdicts.Q2sanity)

# ---------------------------------------------------------------- diagnostics (train_log)
Write-Output ""
Write-Output "=== diagnostics: seed-mean per 5k online bin: alpha(actor) / critic alpha / Q_W / H=-logp; plus cap/fixed checks ==="
$diagGroups = @('tent12','tent12r','tent12i','tent12i_cap03','tent12i_cap1','tent12i_cfix03','tent12i_cfix1','tent12i_g099','tent12_hq')
$showBins = @(0, 5000, 10000, 20000, 30000, 50000, 70000, 90000, 110000)
$diagRows = @(); $q3 = @{}
foreach ($g in $diagGroups) {
  $dirs = @(Get-ChildItem $Logs -Directory | Where-Object { $_.Name -match "^square_${g}_s(\d+)$" })
  if ($dirs.Count -eq 0) { continue }
  $bins = @{}; $amax = @(); $qmax = @(); $constC = @(); $onset = @(); $a110 = @(); $q110 = @(); $seedsUsed = @()
  foreach ($d in $dirs) {
    $f = Join-Path $d.FullName 'train_log.csv'; if (-not (Test-Path $f)) { continue }
    $seedsUsed += [int]([regex]::Match($d.Name, '_s(\d+)$').Groups[1].Value)
    $rows = Import-Csv $f
    $seedBins = @{}; $am = 0; $qm = [double]::NegativeInfinity; $cvals = @(); $firstBind = [double]::NaN
    foreach ($r in $rows) {
      $env = [int]$r.env_steps; $on = $env - $ROLLOUT['square']; if ($on -lt 0) { continue }
      $b = [int]([math]::Floor($on / 5000) * 5000)
      if (-not $seedBins.ContainsKey($b)) { $seedBins[$b] = @{ a = @(); c = @(); q = @(); l = @() } }
      $a = [double]$r.ent_coef; $q = [double]$r.qw_mean
      $seedBins[$b].a += $a; $seedBins[$b].q += $q; $seedBins[$b].l += [double]$r.logp_mean
      if ($a -gt $am) { $am = $a }; if ($q -gt $qm) { $qm = $q }
      if ($r.PSObject.Properties['critic_ent_coef'] -and $r.critic_ent_coef -ne '' -and $r.critic_ent_coef -ne 'nan') {
        $c = [double]$r.critic_ent_coef; $seedBins[$b].c += $c; $cvals += $c
        if (($c -lt $a - 1e-6) -and [double]::IsNaN($firstBind)) { $firstBind = $on }
      }
    }
    foreach ($b in $seedBins.Keys) {
      if (-not $bins.ContainsKey($b)) { $bins[$b] = @{ a = @(); c = @(); q = @(); l = @() } }
      $bins[$b].a += (Mean $seedBins[$b].a); $bins[$b].q += (Mean $seedBins[$b].q); $bins[$b].l += (Mean $seedBins[$b].l)
      if ($seedBins[$b].c.Count -gt 0) { $bins[$b].c += (Mean $seedBins[$b].c) }
    }
    $amax += $am; $qmax += $qm; $onset += $firstBind
    if ($cvals.Count -gt 0) { $constC += ('{0:F3}..{1:F3}' -f ($cvals | Measure-Object -Minimum).Minimum, ($cvals | Measure-Object -Maximum).Maximum) } else { $constC += '-' }
    if ($seedBins.ContainsKey(110000)) { $a110 += (Mean $seedBins[110000].a); $q110 += (Mean $seedBins[110000].q) }
  }
  $line = "square_{0} (n={1}, seeds {2}): " -f $g, $dirs.Count, ($seedsUsed -join ',')
  foreach ($b in $showBins) {
    if ($bins.ContainsKey($b)) {
      $ca = if ($bins[$b].c.Count -gt 0) { '{0:F2}' -f (Mean $bins[$b].c) } else { '-' }
      $line += ("[{0}k a{1:F2}/c{2} Q{3:F0} H{4:F1}] " -f ($b/1000), (Mean $bins[$b].a), $ca, (Mean $bins[$b].q), (-(Mean $bins[$b].l)))
      $diagRows += [pscustomobject]@{ group = $g; bin = $b; alpha = (Mean $bins[$b].a); critic_alpha = $ca; qw = (Mean $bins[$b].q); H = (-(Mean $bins[$b].l)) }
    }
  }
  Write-Output $line
  Write-Output ("    alpha max/seed: {0} | Q_W max/seed: {1} | critic_ent_coef range/seed: {2} | first bind (online): {3} | 110k bin: alpha {4} Q_W {5}" -f (($amax | ForEach-Object { '{0:F2}' -f $_ }) -join '/'), (($qmax | ForEach-Object { '{0:F0}' -f $_ }) -join '/'), ($constC -join ' '), (($onset | ForEach-Object { if ([double]::IsNaN($_)) { 'never' } else { $_ } }) -join '/'), (($a110 | ForEach-Object { '{0:F2}' -f $_ }) -join '/'), (($q110 | ForEach-Object { '{0:F0}' -f $_ }) -join '/'))
  $q3[$g] = @{ a110 = (Mean $a110); q110 = (Mean $q110) }
}
$diagRows | Export-Csv (Join-Path $Out 'diag_bins.csv') -NoTypeInformation

Write-Output ""
Write-Output "=== Q3 gamma diagnostic: g099 online 110k bin actor alpha seed mean; < 3 -> 'gamma is the source (supported)'; >= 10 -> refuted; else undecided. Reference 9/16: tent12i 18.2, cap03 1.71, cap1 3.58; Q_W predicted ~1/10 of tent12i (29.5k) ==="
if ($q3.ContainsKey('tent12i_g099')) {
  $a = $q3['tent12i_g099'].a110; $q = $q3['tent12i_g099'].q110
  $verdicts.Q3 = if ($a -lt 3) { 'gamma is the source (supported)' } elseif ($a -ge 10) { 'refuted' } else { 'undecided' }
  Write-Output ("g099 110k bin: alpha {0:F2}, Q_W {1:F0} -> {2}" -f $a, $q, $verdicts.Q3)
} else { $verdicts.Q3 = 'not run'; Write-Output 'g099 not present' }

Write-Output ""
Write-Output "=== verdict summary (26) ==="
foreach ($k in $verdicts.Keys) { Write-Output ("{0,-9} {1}" -f $k, $verdicts[$k]) }
Write-Output "written: per_seed.csv, groups.csv, diag_bins.csv in $Out"
