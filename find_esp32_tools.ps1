$jsonPath = "$env:LOCALAPPDATA\Arduino15\package_esp32_index.json"
if (-not (Test-Path $jsonPath)) { Write-Host "NOT FOUND: $jsonPath"; exit 1 }
$json = Get-Content $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
$pkg = $json.packages | Where-Object { $_.name -eq "esp32" }
if (-not $pkg) { Write-Host "esp32 packager not found in index"; exit 1 }

Write-Host "== platform versions available =="
$pkg.platforms | ForEach-Object { Write-Host ("  " + $_.version) }

Write-Host ""
Write-Host "== 3.3.11 required tools (toolsDependencies) =="
$plat = $pkg.platforms | Where-Object { $_.version -eq "3.3.11" }
if ($plat) {
  foreach ($d in $plat.toolsDependencies) {
    Write-Host ("  " + $d.packager + ":" + $d.name + "@" + $d.version)
  }
} else {
  Write-Host "  3.3.11 not in index"
}

Write-Host ""
Write-Host "== download URLs for windows =="
foreach ($t in $pkg.tools) {
  if ($t.name -match "esp-x32|esp32-libs|esptool") {
    foreach ($s in $t.systems) {
      if ($s.host -like "*mingw32*") {
        Write-Host ("TOOL: " + $t.name + "@" + $t.version + "  HOST: " + $s.host)
        Write-Host ("URL: " + $s.url)
        Write-Host ""
      }
    }
  }
}
Write-Host "DONE"