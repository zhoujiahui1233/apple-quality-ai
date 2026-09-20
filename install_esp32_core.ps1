$src = "E:\esp32-core-3.3.11"
$dst = Join-Path $env:LOCALAPPDATA "Arduino15\packages\esp32\hardware\esp32\3.3.11"
if (-not (Test-Path (Join-Path $src "platform.txt"))) { Write-Host "SOURCE INVALID: platform.txt not found in $src"; exit 1 }
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
Write-Host "Copied to: $dst"
Write-Host ""
Write-Host "== verify =="
foreach ($f in @("platform.txt","boards.txt","cores","libraries","variants")) {
  Write-Host ("  " + $f + " : " + $(Test-Path (Join-Path $dst $f)))
}
Write-Host "DONE"