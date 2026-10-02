# Windows CPU arms for the #2563 review. Base (merge base 5bf8f0fe4) first, then the PR head, per configuration.
# Each arm: purge the standby list (RAMMap -Es), copy the GGUF unbuffered to a fresh directory (a file the cache has
# never seen), run llama-cli, and record the standby-cache growth, the peak working set, the wall time, the exit code,
# the loader lines, and a hash of stdout (prompt plus 32 tokens at temp 0).
$ErrorActionPreference = 'Continue'
try { Set-MpPreference -DisableRealtimeMonitoring $true } catch { "Defender toggle failed: $_" }
Invoke-WebRequest https://live.sysinternals.com/RAMMap64.exe -OutFile RAMMap64.exe
[System.Environment]::OSVersion.VersionString
$exes = [ordered]@{
  base = (Get-ChildItem -Recurse ctl\build -Filter llama-cli.exe | Select-Object -First 1).FullName
  pr   = (Get-ChildItem -Recurse build -Filter llama-cli.exe | Select-Object -First 1).FullName
}
$exes | Format-Table | Out-String | Write-Host
& $exes.pr --help 2>&1 | Select-String 'defer-ple'

function Standby {
  try { ((Get-Counter '\Memory\Standby Cache Core Bytes','\Memory\Standby Cache Normal Priority Bytes','\Memory\Standby Cache Reserve Bytes').CounterSamples | Measure-Object -Sum CookedValue).Sum }
  catch { "Get-Counter failed: $_" | Write-Host; 0 }
}

$prompts = @{ gemma = $env:PROMPT_GEMMA; ralphqwen = $env:PROMPT_RALPHQWEN; ralphseek = $env:PROMPT_RALPHSEEK }
$configs = @(
  @{ m = 'gemma';     name = 'noflag';         flags = '' },
  @{ m = 'gemma';     name = 'defer';          flags = '--defer-ple' },
  @{ m = 'gemma';     name = 'nommap-defer';   flags = '--no-mmap --defer-ple' },
  @{ m = 'gemma';     name = 'rtr-defer';      flags = '-rtr --defer-ple' },
  @{ m = 'ralphqwen'; name = 'noflag';         flags = '' },
  @{ m = 'ralphqwen'; name = 'defer';          flags = '--defer-ple' },
  @{ m = 'ralphqwen'; name = 'muge-defer';     flags = '-muge --defer-ple' },
  @{ m = 'ralphqwen'; name = 'rtr-muge-defer'; flags = '-rtr -muge --defer-ple' },
  @{ m = 'ralphseek'; name = 'noflag';         flags = '' },
  @{ m = 'ralphseek'; name = 'defer';          flags = '--defer-ple' },
  @{ m = 'ralphseek'; name = 'muge-defer';     flags = '-muge --defer-ple' },
  @{ m = 'ralphseek'; name = 'nommap-defer';   flags = '--no-mmap --defer-ple' },
  @{ m = 'ralphseek'; name = 'rtr-muge-defer'; flags = '-rtr -muge --defer-ple' }
)
$pattern = 'deferring|per-layer|engram|Repacked|mmap is disabled|keeping file mappings|had no effect|prefetch|buffer size|GGML_ASSERT|error|eval time'

foreach ($c in $configs) {
  $hashes = @{}
  foreach ($side in $exes.Keys) {
    $tag = "$($c.m)-$($c.name)-$side"
    $s0 = Standby
    $r = Start-Process .\RAMMap64.exe -ArgumentList '-accepteula','-Es' -PassThru
    if (-not $r.WaitForExit(120000)) { $r.Kill(); "RAMMap -Es timed out" }
    Start-Sleep 5
    $s1 = Standby
    $dir = "arm-$tag"
    robocopy . $dir "$($c.m).gguf" /J /NJH /NJS /NFL /NDL /NP | Out-Null
    Start-Sleep 5
    $before = Standby
    # Start-Process joins arguments without quoting them, so build one quoted line
    $argline = "-m $dir\$($c.m).gguf -p `"$($prompts[$c.m])`" $env:ARGS $($c.flags)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process $exes[$side] -ArgumentList $argline -PassThru -NoNewWindow -RedirectStandardOutput "$tag.out" -RedirectStandardError "$tag.err"
    $peak = 0
    while (-not $p.HasExited) { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 }; Start-Sleep -Milliseconds 200 }
    $wall = $sw.Elapsed.TotalSeconds
    Start-Sleep 5
    $after = Standby
    $h = (Get-FileHash "$tag.out" -Algorithm SHA256).Hash.Substring(0, 12)
    $hashes[$side] = $h
    $line = "{0}: purge_delta_MiB={1:N1} copy_delta_MiB={2:N1} run_delta_MiB={3:N1} peak_ws_MiB={4:N1} wall_s={5:N2} exit={6} out={7}" -f $tag, (($s1 - $s0) / 1MB), (($before - $s1) / 1MB), (($after - $before) / 1MB), ($peak / 1MB), $wall, $p.ExitCode, $h
    $line | Tee-Object -FilePath summary.txt -Append
    Select-String -Path "$tag.err" -Pattern $pattern | ForEach-Object { "    " + $_.Line } | Tee-Object -FilePath summary.txt -Append
    Remove-Item -Recurse -Force $dir
  }
  $verdict = if ($hashes.base -eq $hashes.pr) { 'IDENTICAL' } else { 'DIFFERS' }
  "== $($c.m)-$($c.name): output $verdict (base $($hashes.base), pr $($hashes.pr))" | Tee-Object -FilePath summary.txt -Append
}
exit 0
