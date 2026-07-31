# Localize CDN assets (Font Awesome + Inter variable font) for offline use
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ErrorActionPreference = "Stop"
$root = "E:\apple"
$tmp = Join-Path $env:TEMP "apple_vendor"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

Write-Host "[1/3] Downloading Font Awesome 6.5.0 ..."
$faZip = Join-Path $tmp "fa.zip"
Invoke-WebRequest -Uri "https://use.fontawesome.com/releases/v6.5.0/fontawesome-free-6.5.0-web.zip" -OutFile $faZip -UseBasicParsing
Expand-Archive -Path $faZip -DestinationPath (Join-Path $tmp "fa") -Force
$faRoot = Join-Path $tmp "fa\fontawesome-free-6.5.0-web"
New-Item -ItemType Directory -Force -Path "$root\static\vendor\fontawesome\css", "$root\static\vendor\fontawesome\webfonts" | Out-Null
Copy-Item (Join-Path $faRoot "css\all.min.css") "$root\static\vendor\fontawesome\css\all.min.css" -Force
Copy-Item (Join-Path $faRoot "webfonts\*") "$root\static\vendor\fontawesome\webfonts\" -Recurse -Force
Write-Host "     all.min.css: $((Get-Item "$root\static\vendor\fontawesome\css\all.min.css").Length) bytes"
Write-Host "     webfonts: $((Get-ChildItem "$root\static\vendor\fontawesome\webfonts" -File).Count) files"

Write-Host "[2/3] Downloading Inter variable font ..."
$itZip = Join-Path $tmp "inter.zip"
Invoke-WebRequest -Uri "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip" -OutFile $itZip -UseBasicParsing
Expand-Archive -Path $itZip -DestinationPath (Join-Path $tmp "inter") -Force
$varFont = Get-ChildItem (Join-Path $tmp "inter") -Recurse -Filter "InterVariable.woff2" | Select-Object -First 1
if (-not $varFont) { throw "InterVariable.woff2 not found in the downloaded archive" }
New-Item -ItemType Directory -Force -Path "$root\static\vendor\fonts" | Out-Null
Copy-Item $varFont.FullName "$root\static\vendor\fonts\InterVariable.woff2" -Force
Write-Host "     InterVariable.woff2: $((Get-Item "$root\static\vendor\fonts\InterVariable.woff2").Length) bytes"

Write-Host "[3/3] Updating base.html to local paths ..."
$base = "$root\templates\base.html"
$t = [System.IO.File]::ReadAllText($base)
$t = $t.Replace("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css", "/static/vendor/fontawesome/css/all.min.css")
$google = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">'
$localFont = "<style>@font-face{font-family:'Inter';src:url('/static/vendor/fonts/InterVariable.woff2') format('woff2');font-weight:100 900;font-display:swap;}</style>"
if ($t.Contains($google)) {
    $t = $t.Replace($google, $localFont)
} else {
    Write-Host "     (Google Fonts link already replaced, skip)"
}
[System.IO.File]::WriteAllText($base, $t, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "base.html updated. Done!"
Remove-Item $tmp -Recurse -Force
