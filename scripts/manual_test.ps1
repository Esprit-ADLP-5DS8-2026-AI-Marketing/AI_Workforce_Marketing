param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$ApiKey = $env:SERVICE_API_KEY,
    [string]$Image = "products\product.jpg",
    [string]$ProductFile = "product_info.json",
    [string]$OutDir = "manual_test_output"
)

if (-not $ApiKey) { throw "Set `$env:SERVICE_API_KEY or pass -ApiKey." }
$headers = @{ "X-API-Key" = $ApiKey }

Write-Host "Health:" -ForegroundColor Cyan
Invoke-RestMethod "$BaseUrl/health" | ConvertTo-Json

$product = Get-Content $ProductFile -Raw -Encoding UTF8 | ConvertFrom-Json
$payload = @{
    product  = $product
    settings = @{ number_of_concepts = 2; variations_per_concept = 1 }
} | ConvertTo-Json -Depth 6 -Compress

$payloadFile = New-TemporaryFile
[IO.File]::WriteAllText($payloadFile.FullName, $payload, (New-Object Text.UTF8Encoding $false))

Write-Host "Submitting job..." -ForegroundColor Cyan
$raw = curl.exe -s -X POST "$BaseUrl/generate-creatives" `
    -H "X-API-Key: $ApiKey" `
    -F "payload=<$($payloadFile.FullName)" `
    -F "image=@$Image"
Remove-Item $payloadFile.FullName -Force
$accepted = $raw | ConvertFrom-Json
if ($accepted.status -ne "queued") { Write-Host $raw; throw "Job was not accepted." }
Write-Host "Job: $($accepted.job_id)"

do {
    Start-Sleep -Seconds 10
    $job = Invoke-RestMethod "$BaseUrl/jobs/$($accepted.job_id)" -Headers $headers
    Write-Host ("[{0}] {1} ({2}/{3})" -f $job.status, $job.message, $job.progress.completed, $job.progress.total)
} while ($job.status -in @("queued", "running"))

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
foreach ($img in $job.generated_images) {
    $dest = Join-Path $OutDir $img.filename
    Invoke-WebRequest $img.url -Headers $headers -OutFile $dest
    Write-Host "Downloaded $dest"
}
if ($job.errors.Count -gt 0) { Write-Host "Errors:" -ForegroundColor Yellow; $job.errors }