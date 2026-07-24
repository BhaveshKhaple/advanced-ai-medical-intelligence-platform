$ProjectDir = "C:\Users\yadne\Desktop\medical-ai-platform"
Set-Location $ProjectDir

$envContent = Get-Content ".env"
$apiKey = ($envContent | Where-Object { $_ -match "^GEMINI_API_KEY=" }) -replace "^GEMINI_API_KEY=", ""
$env:GEMINI_API_KEY = $apiKey.Trim()
Write-Host "Loaded API key"

Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
Write-Host "Ports cleared"

$k = $env:GEMINI_API_KEY

Start-Process powershell -ArgumentList @("-NoExit", "-Command", "cd '$ProjectDir'; `$env:GEMINI_API_KEY='$k'; python -m uvicorn app.main:app --reload --port 8000") -WindowStyle Normal
Write-Host "FastAPI starting..."
Start-Sleep -Seconds 5

Start-Process powershell -ArgumentList @("-NoExit", "-Command", "cd '$ProjectDir'; `$env:API_URL='http://localhost:8000'; python -m streamlit run streamlit_app.py --server.port 8501") -WindowStyle Normal
Write-Host "Streamlit starting..."
Start-Sleep -Seconds 5

Start-Process "http://localhost:8501"
Write-Host "Done! Open http://localhost:8501"
