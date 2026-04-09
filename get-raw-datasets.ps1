# ==========================================
# Kaggle Auto Downloader PRO (Parallel Dashboard)
# ==========================================

$ErrorActionPreference = "Stop"

# ---------- CONFIG ----------

$BasePath = Join-Path (Get-Location) "raw-sets"
$TokenPath = Join-Path (Get-Location) ".kaggle_token"
$MaxParallel = 3   # adjust based on bandwidth (2-5 recommended)

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
    if (Test-Path $KaggleJson) {
        icacls $KaggleJson /reset | Out-Null
        attrib -r $KaggleJson
        Remove-Item $KaggleJson -Force
    }

    $TokenKey = $TokenRaw.Trim()
    if ($TokenKey -match "KGAT_(.*)") { $TokenKey = $Matches[1] }
    
    $User = "lemtreursi" # Verified username from user confirmation
    $KaggleConfig = @{
        username = $User
        key = $TokenKey
    } | ConvertTo-Json -Compress
    $KaggleConfig | Set-Content $KaggleJson -Encoding Ascii
} else {
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

# Flatten for dashboard
$TaskList = @()
foreach ($model in $Datasets.Keys) {
    foreach ($ds in $Datasets[$model]) {
        $TaskList += [PSCustomObject]@{
            Dataset = $ds
            Category = $model
            Status = "Pending"
            Progress = 0
            JobId = $null
        }
    }
}

# ---------- DOWNLOAD JOB ----------

$ScriptBlock = {
    param($dataset, $targetPath)

    function Test-ZipValidLocal($zipPath) {
        try {
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
            $zip.Dispose()
            return $true
        } catch { return $false }
    }

    $datasetName = $dataset.Split("/")[1]
    $datasetFolder = Join-Path $targetPath $datasetName
    $zipPath = Join-Path $targetPath "$datasetName.zip"

    if (Test-Path $datasetFolder) {
        Write-Output "RESULT:SKIPPED"
        return
    }

    if (Test-Path $zipPath) {
        if (!(Test-ZipValidLocal $zipPath)) {
            Remove-Item $zipPath -Force
        }
    }

    if (!(Test-Path $zipPath)) {
        Write-Output "STATUS:DOWNLOADING"
        # --quiet stops the redundant progress bar clutter in jobs
        kaggle datasets download -d $dataset -p $targetPath --quiet
    }

    if (Test-Path $zipPath) {
        if (Test-ZipValidLocal $zipPath) {
            Write-Output "STATUS:QUEUED_FOR_UNZIP"
            # 2026: Cross-process serialization via Named Mutex
            $Mutex = New-Object System.Threading.Mutex($false, "Global\LemGendaryExtractionLock")
            $null = $Mutex.WaitOne()
            try {
                Write-Output "STATUS:EXTRACTING"
                Expand-Archive -Path $zipPath -DestinationPath $datasetFolder -Force
                # 2026 CLEANUP: Delete zip after successful extraction
                Remove-Item $zipPath -Force
                Write-Output "RESULT:COMPLETED"
            } finally {
                $Mutex.ReleaseMutex()
                $Mutex.Dispose()
            }
        } else {
            Write-Output "RESULT:ERROR_ZIP_INVALID"
        }
    }
}

# ---------- DASHBOARD LOOP ----------

Write-Host "`nInitializing LemGendary Acquisition Dashboard..." -ForegroundColor Cyan

$TotalDone = 0
$TotalTasks = $TaskList.Count

while ($TotalDone -lt $TotalTasks) {
    # Check for free slots and launch pending
    $RunningJobs = $TaskList | Where-Object { $_.JobId -ne $null -and (Get-Job -Id $_.JobId).State -eq "Running" }
    
    if ($RunningJobs.Count -lt $MaxParallel) {
        $NextTask = $TaskList | Where-Object { $_.Status -eq "Pending" } | Select-Object -First 1
        if ($NextTask) {
            $modelPath = Join-Path $BasePath $NextTask.Category
            if (!(Test-Path $modelPath)) { New-Item -ItemType Directory -Force -Path $modelPath | Out-Null }
            
            $NextTask.JobId = (Start-Job -ScriptBlock $ScriptBlock -ArgumentList $NextTask.Dataset, $modelPath).Id
            $NextTask.Status = "Starting"
        }
    }

    # Update states from Job output
    foreach ($task in $TaskList | Where-Object { $_.JobId -ne $null }) {
        $job = Get-Job -Id $task.JobId
        $output = $job | Receive-Job
        foreach ($line in $output) {
            if ($line -match "STATUS:(.*)") { $task.Status = $Matches[1] }
            if ($line -match "RESULT:(.*)") { $task.Status = $Matches[1] }
        }

        if ($job.State -eq "Completed" -and $task.Status -notmatch "COMPLETED|SKIPPED|ERROR") {
            $task.Status = "Completed"
        }
        if ($job.State -eq "Failed") { $task.Status = "Job Failed" }
    }

    # Draw Progress Bar Dashboard
    $TotalDone = ($TaskList | Where-Object { $_.Status -match "COMPLETED|SKIPPED|ERROR" }).Count
    $Percent = [math]::Round(($TotalDone / $TotalTasks) * 100)
    
    $CurrentAction = ($TaskList | Where-Object { $_.Status -match "DOWNLOADING|EXTRACTING" } | Select-Object -ExpandProperty Dataset -First 1)
    if (!$CurrentAction) { $CurrentAction = "Waiting for slots..." }

    Write-Progress -Activity "LemGendary Dataset Acquisition" `
                   -Status "Processed: $TotalDone / $TotalTasks ($Percent%) | Working on: $CurrentAction" `
                   -PercentComplete $Percent

    # Optional: Detailed text dashboard in console
    # Clear-Host # Caution: Clearing host rapidly might cause flicker
    
    Start-Sleep -Seconds 2
}

Write-Progress -Activity "LemGendary Dataset Acquisition" -Completed

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

Write-Host "`n[SUCCESS] Dataset index saved to $IndexPath"
Write-Host "All downloads and extractions completed. Zips purged."
