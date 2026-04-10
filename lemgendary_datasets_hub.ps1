# ==========================================
# LemGendary Dataset Hub (SOTA 2026 Dashboard)
# Centralized Dataset Orchestrator
# ==========================================

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "LemGendary Dataset Hub v2.5.0"

# ---------- COLOR SCHEME ----------
$C_Cyan = "`e[36m"
$C_Gold = "`e[33m"
$C_Green = "`e[32m"
$C_Red = "`e[31m"
$C_Gray = "`e[90m"
$C_Reset = "`e[0m"

# ---------- CONFIG ----------
$BasePath = Join-Path (Get-Location) "raw-sets"
$OutputPath = Join-Path (Get-Location) "compiled-datasets"
$TokenPath = Join-Path (Get-Location) ".kaggle_token"
$IndexPath = Join-Path $OutputPath "index.json"
$MaxParallel = 3

# ---------- BRANDING ----------
function Show-Branding {
    Clear-Host
    Write-Host "$C_Gold"
    Write-Host "  _      ______ __  __ _____ ______ _   _ _____          _____  __     __ "
    Write-Host " | |    |  ____|  \/  / ____|  ____| \ | |  __ \   /\   |  __ \ \ \   / / "
    Write-Host " | |    | |__  | \  / | |  __| |__  |  \| | |  | | /  \  | |__) | \ \_/ /  "
    Write-Host " | |    |  __| | |\/| | | |_ |  __| | . ` | |  | |/ /\ \ |  _  /   \   /   "
    Write-Host " | |____| |____| |  | | |__| | |____| |\  | |__| / ____ \| | \ \    | |    "
    Write-Host " |______|______|_|  |_|\_____|______|_| \_|_____/_/    \_\_|  \_\   |_|    "
    Write-Host "                                                                           "
    Write-Host "         [ SOTA 2026 DATASET ORCHESTRATION HUB - v2.5.0 ]                  "
    Write-Host "$C_Reset"
}

# ---------- DOWNLOADER LOGIC ----------
function Setup-Kaggle {
    if (!(Test-Path $TokenPath)) {
        Write-Host "  ⚠️  [ERROR] Missing .kaggle_token file!" -ForegroundColor Red
        return $false
    }

    $KaggleDir = Join-Path $env:USERPROFILE ".kaggle"
    if (!(Test-Path $KaggleDir)) { New-Item -ItemType Directory -Force -Path $KaggleDir | Out-Null }
    $KaggleJson = Join-Path $KaggleDir "kaggle.json"

    $TokenRaw = Get-Content $TokenPath -Raw
    if ($TokenRaw -notmatch '\{.*"username".*\}') {
        $TokenKey = $TokenRaw.Trim()
        if ($TokenKey -match "KGAT_(.*)") { $TokenKey = $Matches[1] }
        $User = "lemtreursi" # Standardized for 2026
        $KaggleConfig = @{ username = $User; key = $TokenKey } | ConvertTo-Json -Compress
        $KaggleConfig | Set-Content $KaggleJson -Encoding Ascii
    } else {
        Copy-Item $TokenPath $KaggleJson -Force
    }
    
    icacls $KaggleJson /inheritance:r | Out-Null
    icacls $KaggleJson /grant:r "$($env:USERNAME):(M)" | Out-Null
    
    if (!(Get-Command kaggle -ErrorAction SilentlyContinue)) { pip install kaggle }
    return $true
}

function Run-Acquisition {
    if (!(Setup-Kaggle)) { return }

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

    $TaskList = @()
    foreach ($model in $Datasets.Keys) {
        foreach ($ds in $Datasets[$model]) {
            $TaskList += [PSCustomObject]@{ Dataset = $ds; Category = $model; Status = "Pending"; JobId = $null }
        }
    }

    $ScriptBlock = {
        param($dataset, $targetPath)
        function Test-ZipValidLocal($zipPath) {
            try {
                Add-Type -AssemblyName System.IO.Compression.FileSystem
                $zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath); $zip.Dispose(); return $true
            } catch { return $false }
        }
        $datasetName = $dataset.Split("/")[1]
        $datasetFolder = Join-Path $targetPath $datasetName
        $zipPath = Join-Path $targetPath "$datasetName.zip"

        if (Test-Path $datasetFolder) { Write-Output "RESULT:SKIPPED"; return }
        
        Write-Output "STATUS:DOWNLOADING"
        kaggle datasets download -d $dataset -p $targetPath --quiet
        
        if (Test-Path $zipPath) {
            Write-Output "STATUS:EXTRACTING"
            $Mutex = New-Object System.Threading.Mutex($false, "Global\LemGendaryExtractionLock")
            $null = $Mutex.WaitOne()
            try {
                Expand-Archive -Path $zipPath -DestinationPath $datasetFolder -Force
                Remove-Item $zipPath -Force
                Write-Output "RESULT:COMPLETED"
            } finally { $Mutex.ReleaseMutex(); $Mutex.Dispose() }
        }
    }

    Write-Host "`n  📡 [ACQUISITION] Processing $C_Cyan$($TaskList.Count)$C_Reset datasets in parallel..."
    $TotalDone = 0; $TotalTasks = $TaskList.Count
    while ($TotalDone -lt $TotalTasks) {
        $RunningJobs = $TaskList | Where-Object { $_.JobId -ne $null -and (Get-Job -Id $_.JobId).State -eq "Running" }
        if ($RunningJobs.Count -lt $MaxParallel) {
            $NextTask = $TaskList | Where-Object { $_.Status -eq "Pending" } | Select-Object -First 1
            if ($NextTask) {
                $dir = Join-Path $BasePath $NextTask.Category
                if (!(Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
                $NextTask.JobId = (Start-Job -ScriptBlock $ScriptBlock -ArgumentList $NextTask.Dataset, $dir).Id
                $NextTask.Status = "Starting"
            }
        }
        foreach ($task in $TaskList | Where-Object { $_.JobId -ne $null }) {
            $job = Get-Job -Id $task.JobId; $output = $job | Receive-Job
            foreach ($line in $output) {
                if ($line -match "STATUS:(.*)") { $task.Status = $Matches[1] }
                if ($line -match "RESULT:(.*)") { $task.Status = $Matches[1] }
            }
            if ($job.State -eq "Completed" -and $task.Status -notmatch "COMPLETED|SKIPPED|ERROR") { $task.Status = "Completed" }
        }
        $TotalDone = ($TaskList | Where-Object { $_.Status -match "COMPLETED|SKIPPED|ERROR" }).Count
        $Percent = [math]::Round(($TotalDone / $TotalTasks) * 100)
        Write-Progress -Activity "LemGendary Acquisition" -Status "Progress: $TotalDone / $TotalTasks ($Percent%)" -PercentComplete $Percent
        Start-Sleep -Seconds 2
    }
    Write-Progress -Activity "LemGendary Acquisition" -Completed
    Write-Host "  ✅ ALL DATASETS ACQUIRED." -ForegroundColor Green
    Start-Sleep -Seconds 2
}

# ---------- PIPELINE ORCHESTRATOR ----------
function Run-Pipeline {
    Write-Host "`n  📦 Enter Dataset Identifier (e.g. SOTA_Detection_v1)" -ForegroundColor Gold
    $DatasetName = Read-Host "  Name [default: sota_synthesis]"
    if ([string]::IsNullOrWhiteSpace($DatasetName)) { $DatasetName = "sota_synthesis" }

    Write-Host "`n  ⚙️  Initiating Synthesis Pipeline for [$DatasetName]..." -ForegroundColor Cyan
    python compiler-pipeline.py --name $DatasetName
    Write-Host "`n  ✅ PIPELINE COMPLETED. REFRESHING INDEX..." -ForegroundColor Green
    Start-Sleep -Seconds 2
}

# ---------- STATS DASHBOARD ----------
function Show-Stats {
    $Subsets = Get-ChildItem -Path $OutputPath -Directory -ErrorAction SilentlyContinue
    if ($Subsets) {
        $Latest = $Subsets | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $SubIndexPath = Join-Path $Latest.FullName "index.json"
        if (Test-Path $SubIndexPath) {
            try {
                $data = Get-Content $SubIndexPath | ConvertFrom-Json
                $total = $data.Count
                $autolabeled = ($data | Where-Object { $_.is_autolabeled -eq $true }).Count
                Write-Host "  📊 [HUB STATS] Latest Dataset: $C_Cyan$($Latest.Name)$C_Reset | Total: $C_Green$total$C_Reset | Auto-Labeled: $C_Gold$autolabeled$C_Reset"
            } catch { Write-Host "  📊 [HUB STATS] Found datasets but index parsing failed." -ForegroundColor Gray }
        } else { Write-Host "  📊 [HUB STATS] Latest dataset [$($Latest.Name)] has no index yet." -ForegroundColor Gray }
    } else { Write-Host "  📊 [HUB STATS] No compiled datasets found. Run option 2." -ForegroundColor Gray }
}

# ---------- MAIN MENU ----------
while ($true) {
    Show-Branding
    Show-Stats
    Write-Host "`n  1. $C_Cyan[ACQUIRE]$C_Reset   Download & Extract Raw Datasets (Kaggle)"
    Write-Host "  2. $C_Green[COMPILE]$C_Reset   Run Synthesis Pipeline (Standardize & Vet)"
    Write-Host "  3. $C_Gold[METADATA]$C_Reset  Rebuild Search Index"
    Write-Host "  4. $C_Gray[VIEW]$C_Reset      Explore Compiled Output Folder"
    Write-Host "  Q. $C_Red[QUIT]$C_Reset      Exit Hub"
    
    $Choice = Read-Host "`n  Select operation"
    switch ($Choice) {
        "1" { Run-Acquisition }
        "2" { Run-Pipeline }
        "3" { python metadata_builder.py }
        "4" { ii $OutputPath }
        "q" { break } "Q" { break }
        default { Write-Host "Invalid selection." -ForegroundColor Red; Start-Sleep -Seconds 1 }
    }
}

Write-Host "`nExiting LemGendary Hub. Keep Convergence High.`n" -ForegroundColor Cyan
