$src = "E:\esp32-libs-3.3.11\esp32-libs"
$dst = Join-Path $env:LOCALAPPDATA "Arduino15\packages\esp32\tools\esp32-libs\3.3.11"
if (-not (Test-Path $src)) { Write-Host "SOURCE NOT FOUND: $src"; exit 1 }
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
Write-Host "Copied to: $dst"
Write-Host ""
Write-Host "== verify =="
$lib = Join-Path $dst "lib"
if (Test-Path $lib) {
  $aCount = (Get-ChildItem $lib -Filter *.a -File).Count
  Write-Host ("lib dir OK, .a files: " + $aCount)
  foreach ($need in @("libnet80211.a","librtc.a","libphy.a","libesp_phy.a","libbtdm_app.a","libpp.a","libsmartconfig.a")) {
    $p = Join-Path $lib $need
    Write-Host ("  " + $need + " : " + $(Test-Path $p))
  }
} else {
  Write-Host "LIB DIR MISSING!"
}
Write-Host "DONE"