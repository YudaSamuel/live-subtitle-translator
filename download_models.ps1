# Downloads the models from Hugging Face (with a progress bar; resumes if interrupted)
$ErrorActionPreference = "Stop"
$models = Join-Path $PSScriptRoot "models"

$repos = @(
    @{ repo = "mobiuslabsgmbh/faster-whisper-large-v3-turbo";
       files = "config.json","preprocessor_config.json","tokenizer.json","vocabulary.json","model.bin" },
    @{ repo = "JustFrederik/nllb-200-distilled-1.3B-ct2-int8";
       files = "config.json","shared_vocabulary.txt","tokenizer.json","model.bin" }
)

foreach ($r in $repos) {
    $dir = Join-Path $models ($r.repo.Split("/")[1])
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($f in $r.files) {
        Write-Host "`n=== $($r.repo) / $f" -ForegroundColor Cyan
        $url = "https://huggingface.co/$($r.repo)/resolve/main/$f"
        for ($try = 1; $try -le 30; $try++) {
            curl.exe -L -C - --retry 5 --retry-delay 3 --retry-all-errors -o (Join-Path $dir $f) $url
            if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 33) { break }
            Write-Host "Connection dropped (curl $LASTEXITCODE), resuming in 3 s ($try/30)..." -ForegroundColor Yellow
            Start-Sleep 3
        }
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 33) { throw "Failed to download $f (curl $LASTEXITCODE)" }
    }
}
Write-Host "`nALL MODELS DOWNLOADED" -ForegroundColor Green
