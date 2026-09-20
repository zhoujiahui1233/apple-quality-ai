$ErrorActionPreference = "Continue"
$libRoot = Join-Path $env:USERPROFILE "Documents\Arduino\libraries"
New-Item -ItemType Directory -Force -Path $libRoot | Out-Null
Write-Host ("Library dir: " + $libRoot)

$libs = @(
  "Adafruit_ADS1X15",
  "Adafruit_BME680_Library",
  "Adafruit_SGP30_Sensor",
  "Adafruit_SSD1306",
  "Adafruit_GFX",
  "Adafruit_BusIO",
  "Adafruit_Unified_Sensor"
)

$mirrors = @(
  "https://ghfast.top/https://github.com/adafruit/{0}/archive/refs/heads/master.zip",
  "https://gh-proxy.com/https://github.com/adafruit/{0}/archive/refs/heads/master.zip",
  "https://ghproxy.net/https://github.com/adafruit/{0}/archive/refs/heads/master.zip"
)

foreach ($lib in $libs) {
  $ok = $false
  foreach ($m in $mirrors) {
    $url = [string]::Format($m, $lib)
    $zip = Join-Path $env:TEMP ($lib + ".zip")
    Write-Host ("Downloading " + $lib + " ...")
    try {
      Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 90
      $ok = $true
      break
    } catch {
      Write-Host ("  mirror failed: " + $_.Exception.Message)
    }
  }
  if (-not $ok) { Write-Host ("SKIP " + $lib); continue }
  $dest = Join-Path $libRoot $lib
  if (Test-Path $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
  Expand-Archive -LiteralPath $zip -DestinationPath $libRoot -Force
  $extracted = Get-ChildItem -LiteralPath $libRoot -Directory | Where-Object { $_.Name -like ($lib + "-*") } | Select-Object -First 1
  if ($extracted) {
    if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Recurse -Force }
    Rename-Item -LiteralPath $extracted.FullName -NewName $lib
  }
  Remove-Item -LiteralPath $zip -Force
  Write-Host ("OK -> " + $dest)
}
Write-Host "ALL DONE - restart Arduino IDE and compile again"