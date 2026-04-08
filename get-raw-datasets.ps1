# ==========================================
# Kaggle Auto Downloader PRO (Parallel)
# ==========================================

$ErrorActionPreference = "Stop"

# ---------- CONFIG ----------

$BasePath = Join-Path (Get-Location) "raw-sets"
$TokenPath = Join-Path (Get-Location) ".kaggle_token"
$MaxParallel = 3   # adjust based on bandwidth (2–5 recommended)

# ---------- CHECK TOKEN ----------

if (!(Test-Path $TokenPath)) {
    Write-Host "[ERROR] Missing .kaggle_token file!" -ForegroundColor Red
    exit
}

# ---------- SETUP KAGGLE ----------

$KaggleDir = Join-Path $env:USERPROFILE ".kaggle"
if (!(Test-Path $KaggleDir)) {
    New-Item -ItemType Directory -Force -Path $KaggleDir | Out-Null
}
$KaggleJson = Join-Path $KaggleDir "kaggle.json"

# 2026 Resilience: Auto-heal malformed or raw tokens
$TokenRaw = Get-Content $TokenPath -Raw
if ($TokenRaw -notmatch '\{.*"username".*\}') {
    Write-Host "[INFO] Raw Kaggle token detected. Wrapping into valid JSON..."
    
    # 2026: Permission Reset Guard
    # If a previous run locked the file to Read-Only (NTFS), we must reset permissions to regain write access.
    if (Test-Path $KaggleJson) {
        icacls $KaggleJson /reset | Out-Null
        attrib -r $KaggleJson
        Remove-Item $KaggleJson -Force
    }

    $TokenKey = $TokenRaw.Trim()
    # 2026 Resilience: Strip prefixes like 'KGAT_' which are not part of the hex key
    if ($TokenKey -match "KGAT_(.*)") {
        $TokenKey = $Matches[1]
    }
    
    $User = "lemtreursi" # Verified username from user confirmation
    $KaggleConfig = @{
        username = $User
        key = $TokenKey
    } | ConvertTo-Json -Compress
    $KaggleConfig | Set-Content $KaggleJson -Encoding Ascii
} else {
    # Even if it is JSON, ensure we can copy over it
    if (Test-Path $KaggleJson) { attrib -r $KaggleJson }
    Copy-Item $TokenPath $KaggleJson -Force
}

# Set permissions (Windows)
try {
    icacls $KaggleJson /inheritance:r | Out-Null
    icacls $KaggleJson /grant:r "$($env:USERNAME):(M)" | Out-Null
} catch {
    Write-Host "[WARNING] Could not set strict permissions on kaggle.json"
}

if (!(Get-Command kaggle -ErrorAction SilentlyContinue)) {
    pip install kaggle
}

# ---------- DATASETS ----------

$Datasets = @{
    "nima_aesthetic" = @("jessevent/all-kaggle-datasets","romainbeaumont/laion-aesthetic-6plus")
    "nima_technical" = @("anjanatiha/koniq-10k","taehoonlee/spaq","kanchana1990/tid2013")
    "codeformer"     = @("jessicali9530/celeba-dataset","kashyapkvh/mm-celeba-hq-dataset","arnaud58/flickrfaceshq-dataset-ffhq")
    "parsenet"       = @("ashishjangra27/celeba-mask-hq","sanjanchaudhari/lapa-dataset")
    "retinaface"     = @("gauravduttakiit/wider-face-dataset","aryashah2k/crowdhuman-dataset")
    "yolov8n"        = @("awsaf49/coco-2017-dataset","amsiddiqui/objects365","landlord/open-images-dataset")
    "ultrazoom"      = @("joe1995/div2k-dataset","joe1995/flickr2k","nikhilpandey360/hq-50k","doobiusp/huge-dataset-of-images-super-resolution")
    "ffanet"         = @("balraj98/reside-dataset")
    "mprnet"         = @("rahulbhalley/rain-dataset","jeongbinpark/rain-dataset")
    "mirnet"         = @("nareshbhat/low-light-image-enhancement","rajat95gupta/smartphone-image-denoising-dataset","joeljang/mit-adobe-fivek","ursulachang/dped-dataset")
    "nafnet"         = @("rahulbhalley/gopro-deblur","khushikhushikhushi/hide-dataset","akshatbhatnagar/darmstadt-noise-dataset")
}

# ---------- DOWNLOAD JOB ----------

$ScriptBlock = {
    param($dataset, $targetPath)

    # Re-define utility inside the job scope
    function Test-ZipValidLocal($zipPath) {
        try {
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
            $zip.Dispose()
            return $true
        } catch {
            return $false
        }
    }

    $datasetName = $dataset.Split("/")[1]
    $datasetFolder = Join-Path $targetPath $datasetName
    $zipPath = Join-Path $targetPath "$datasetName.zip"

    if (Test-Path $datasetFolder) {
        Write-Host "[SKIP] [$datasetName] already extracted"
        return
    }

    if (Test-Path $zipPath) {
        Write-Host "[INFO] [$datasetName] found existing zip, validating..."
        if (!(Test-ZipValidLocal $zipPath)) {
            Write-Host "[ERROR] [$datasetName] corrupted zip -> deleting"
            Remove-Item $zipPath -Force
        }
    }

    if (!(Test-Path $zipPath)) {
        Write-Host "[DOWN] [$datasetName] downloading..."
        kaggle datasets download -d $dataset -p $targetPath
    }

    if (Test-Path $zipPath) {
        if (Test-ZipValidLocal $zipPath) {
            Write-Host "[INFO] [$datasetName] extracting..."
            Expand-Archive -Path $zipPath -DestinationPath $datasetFolder -Force
        } else {
            Write-Host "[ERROR] [$datasetName] zip invalid after download"
            return
        }
    }
}

# ---------- PARALLEL EXECUTION ----------

$Jobs = @()

foreach ($model in $Datasets.Keys) {
    $modelPath = Join-Path $BasePath $model
    if (!(Test-Path $modelPath)) {
        New-Item -ItemType Directory -Force -Path $modelPath | Out-Null
    }

    foreach ($ds in $Datasets[$model]) {
        while (($Jobs | Where-Object { $_.State -eq "Running" }).Count -ge $MaxParallel) {
            Start-Sleep -Seconds 2
        }

        $job = Start-Job -ScriptBlock $ScriptBlock -ArgumentList $ds, $modelPath
        $Jobs += $job
    }
}

# Wait for all
$Jobs | Wait-Job | Receive-Job
$Jobs | Remove-Job

# ---------- DATASET INDEX ----------

$Index = @()

if (Test-Path $BasePath) {
    foreach ($model in Get-ChildItem $BasePath -Directory) {
        foreach ($ds in Get-ChildItem $model.FullName -Directory) {
            $files = Get-ChildItem $ds.FullName -Recurse -File -ErrorAction SilentlyContinue
            $size = ($files | Measure-Object -Property Length -Sum).Sum

            $Index += [PSCustomObject]@{
                model = $model.Name
                dataset = $ds.Name
                path = $ds.FullName
                size_mb = [math]::Round($size / 1MB, 2)
            }
        }
    }
}

$IndexPath = Join-Path $BasePath "dataset_index.json"
$Index | ConvertTo-Json -Depth 4 | Out-File $IndexPath -Encoding utf8

Write-Host "Dataset index saved to $IndexPath"
Write-Host "COMPLETED"
