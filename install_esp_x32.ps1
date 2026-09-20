$src = "E:\xtensa-esp-elf"
$dst = Join-Path $env:LOCALAPPDATA "Arduino15\packages\esp32\tools\esp-x32\2601-cn"
if (-not (Test-Path (Join-Path $src "bin"))) { Write-Host "SOURCE INVALID: bin not found in $src"; exit 1 }
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
Write-Host "Copied to: $dst"
Write-Host ""
Write-Host "== verify =="
$gcc = Get-ChildItem (Join-Path $dst "bin") -Filter "*gcc*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Host ("  bin dir: " + $(Test-Path (Join-Path $dst "bin")))
Write-Host ("  compiler: " + $(if ($gcc) { $gcc.Name } else { "MISSING" }))
Write-Host ("  lib dir: " + $(Test-Path (Join-Path $dst "lib")))
Write-Host "DONE"