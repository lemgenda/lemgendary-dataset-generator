$Env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'
$Vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$Reg = Join-Path $PSScriptRoot 'unified_data.yaml'
$Raw = Join-Path $PSScriptRoot 'raw-sets'
$hfManagerPath = Join-Path $PSScriptRoot 'hf_manager.py'
$ghManagerPath = Join-Path $PSScriptRoot 'gh_manager.py'
$kagManagerPath = Join-Path $PSScriptRoot 'kaggle_manager.py'
$gdManagerPath = Join-Path $PSScriptRoot 'gd_manager.py'

$TokenPath = Join-Path $PSScriptRoot '.kaggle_token'
if (Test-Path $TokenPath) {
    $env:KAGGLE_API_TOKEN = (Get-Content $TokenPath -Raw).Trim()
}

$DownloadSB = {
    param($ds, $sharedPath, $vpy, $kagManager)
    $isC = $ds -match 'competition'
    $ref = $ds.Replace('kaggle://', '')
    if ($ds -match 'competition:(.*)') { $ref = $Matches[1] }
    $dn = $ref.Split('/')[-1]
    $fold = Join-Path $sharedPath $dn
    
    Write-Output "STATUS:KAG-PULLING"
    if ($isC) {
        & $vpy $kagManager --repo_id $ref --output_dir $fold --is_competition 2>&1
    } else {
        & $vpy $kagManager --repo_id $ref --output_dir $fold 2>&1
    }
    
    $z = Join-Path $sharedPath ($dn + '.zip')
    if (Test-Path $z) {
         Write-Output "RESULT:DOWNLOADED"
    } elseif (Test-Path $fold) {
         # 2026 Resilience: Kaggle often returns a folder instead of a zip
         Write-Output "RESULT:COMPLETED"
    } else {
         $fileMatches = Get-ChildItem $sharedPath -Filter "$dn*"
         if ($fileMatches) {
             Write-Output "RESULT:DOWNLOADED"
         } else {
             Write-Output "RESULT:FAILED"
         }
    }
}

$HuggingFaceSB = {
    param($ds, $sharedPath, $vpy, $hfManager)
    $repoId = $ds.Replace('hf://', '')
    
    # SOTA Fix: Strip surgical file targets to prevent Windows invalid char ':' errors
    $baseRepo = if ($repoId -match ':') { $repoId.Split(':')[0] } else { $repoId }
    $dn = $baseRepo.Split('/')[-1]
    $outFold = Join-Path $sharedPath $dn
    
    Write-Output "STATUS:HF-PULLING"
    & $vpy $hfManager --repo_id $repoId --output_dir $outFold --repo_type dataset 2>&1
    
    if ((Get-ChildItem $outFold -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0) {
         Write-Output "RESULT:COMPLETED"
    } else {
         Write-Output "RESULT:FAILED"
    }
}

$GHSourceSB = {
    param($ds, $sharedPath, $vpy, $ghManager)
    $repoId = $ds.Replace('gh://', '')
    $dn = $repoId.Split('/')[-1]
    $outFold = Join-Path $sharedPath $dn
    
    Write-Output "STATUS:GH-CLONING"
    & $vpy $ghManager --repo_url $repoId --output_dir $outFold 2>&1
    
    if ((Get-ChildItem $outFold -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0) {
         Write-Output "RESULT:COMPLETED"
    } else {
         Write-Output "RESULT:FAILED"
    }
}

$GDriveSB = {
    param($ds, $sharedPath, $vpy, $gdManager)
    $repoId = $ds.Replace('gdrive://', '')
    $dn = $repoId.Split('/')[-1]
    $outFold = Join-Path $sharedPath $dn
    
    Write-Output "STATUS:GD-PULLING"
    & $vpy $gdManager --repo_id $repoId --output_dir $outFold 2>&1
    
    $z = Join-Path $sharedPath ($dn + '.zip')
    if (Test-Path $z) {
         Write-Output "RESULT:DOWNLOADED"
    } elseif (Test-Path $outFold) {
         Write-Output "RESULT:COMPLETED"
    } else {
         Write-Output "RESULT:FAILED"
    }
}

$UnpackSB = {
    param($ds, $sharedPath, $vpy)
    $dn = $ds.Split('/')[-1]
    $fold = Join-Path $sharedPath $dn
    $z = Join-Path $sharedPath ($dn + '.zip')
    
    Write-Output "STATUS:UNPACKING"
    try {
        $ArchMgr = Join-Path (Split-Path $vpy -Parent | Split-Path -Parent | Split-Path -Parent) 'archive_manager.py'
        
        # If the zip is missing, it means HF or Kaggle messed up. But HF emits COMPLETED so it bypasses this!
        if (!(Test-Path $z)) {
            Write-Output "NOTIFICATION:Zip not found for extraction: $dn"
            Write-Output "`n`rRESULT:FAILED"
            return
        }
        
        & $vpy $ArchMgr --zip $z --dest $fold --action extract 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Output "NOTIFICATION:Extraction Finished & Zip Deleted: $dn"
            Write-Output "RESULT:COMPLETED"
        } else {
            Write-Output "`n`rRESULT:FAILED"
        }
    } catch {
        Write-Output "NOTIFICATION:Extraction Failed for $dn : $($_.Exception.Message)"
        Write-Output "RESULT:FAILED"
    }
}

function Initialize-Environment {
    Write-Host "[PRE-FLIGHT] Verifying LemGendary Environment..." -ForegroundColor Gray
    
    if (!(Test-Path $Vpy)) {
        Write-Host "[WARNING] Virtual environment (.venv) not found!" -ForegroundColor Yellow
        $Choice = Read-Host "Would you like to create and initialize the environment now? (Y/N)"
        if ($Choice -match '^y') {
            Write-Host "Creating .venv..." -ForegroundColor Cyan
            & python -m venv .venv
            if ($LASTEXITCODE -ne 0) { 
                Write-Host "[ERROR] Failed to create .venv. Ensure Python is installed." -ForegroundColor Red
                Read-Host "Press Enter to exit"
                exit 1 
            }
        } else {
            Write-Host "[ERROR] Cannot proceed without .venv. Exiting." -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
    
    Write-Host "Checking Hardware Acceleration..." -ForegroundColor Gray
    $Check = & $Vpy -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CPU_ONLY')"
    
    if ($Check -notmatch 'CUDA_OK') {
        Write-Host "[ENVIRONMENT] GPU Acceleration (CUDA) is NOT detected!" -ForegroundColor Yellow
        $Choice = Read-Host "Would you like to repair/install dependencies from requirements.txt? (Y/N)"
        if ($Choice -match '^y') {
            Write-Host "Installing/Repairing dependencies (this may take several minutes)..." -ForegroundColor Cyan
            & $Vpy -m pip install -r requirements.txt
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[SUCCESS] Environment repaired successfully!" -ForegroundColor Green
                Start-Sleep -Seconds 2
            } else {
                Write-Host "[ERROR] Installation failed. Check internet connection." -ForegroundColor Red
                Read-Host "Press Enter to exit"
                exit 1
            }
        }
    } else {
        # 2026: SOTA Stealth Sync - Silently update CUDA status for the Menu
        $global:CudaReady = $true
        Start-Sleep -Milliseconds 100
    }
}

# Run Initialization before anything else
Initialize-Environment

function Get-RegData {
    if (!(test-path $Reg)) { Write-Host '  [ERROR] unified_data.yaml missing!' -Fore Red; return $null }
    $RegFixed = $Reg.Replace('\', '/')
    $YJ = & $Vpy -c "import yaml, json, sys; print(json.dumps(yaml.safe_load(open('$RegFixed'))))"
    if (!$YJ) { Write-Host '  [ERROR] Manifest load failed!' -Fore Red; return $null }
    return $YJ | ConvertFrom-Json
}

$GlobalData = Get-RegData
$OutFolderName = "compiled-datasets"
if ($GlobalData -and $GlobalData._registry_metadata.output_folder_name) { 
    $OutFolderName = $GlobalData._registry_metadata.output_folder_name 
}
$Out = Join-Path (Get-Location) $OutFolderName

function Show-Stats {
    if (Test-Path $Out) {
        $Lat = Get-ChildItem $Out -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($Lat) {
            $IdxPath = Join-Path $Lat.FullName 'index.json'
            if (Test-Path $IdxPath) {
                try {
                    # 2026 Optimization: Bypassing PowerShell's ConvertFrom-Json for massive manifolds (1.4M+ items)
                    # Python's JSON parser is 100x faster and won't crash the shell memory buffer.
                    $IdxFixed = $IdxPath.Replace('\', '/')
                    $Cnt = & $Vpy -c "import json; print(len(json.load(open(r'$IdxFixed', encoding='utf-8'))))"
                    if ($Cnt) {
                        Write-Host ("  [STATS] Latest: " + $Lat.Name + " | Total: " + $Cnt) -ForegroundColor Cyan
                    }
                } catch { }
            }
        }
    }
}

function Get-RefStatus {
    param($Ref, $SharedPath, $KaggleRef = $null)
    
    if ($Ref -match '^manifold://') {
        $m_name = $Ref.Replace('manifold://', '')
        $RegData = Get-RegData
        $Prefix = $RegData._registry_metadata.name_prefix
        $Suffix = if ($m_name.EndsWith("MultiTask")) { "" } else { $RegData._registry_metadata.name_suffix }
        $OutPath = (Get-Item "..\LemGendaryDatasets").FullName
        if (Test-Path (Join-Path $OutPath ($Prefix + $m_name + $Suffix))) { return "Extracted" }
        return "Missing"
    }
    
    $isHF = $Ref -match 'hf://'
    $isGH = $Ref -match 'gh://'
    $kaggleSource = $Ref -match 'kaggle://'
    $isGDrive = $Ref -match 'gdrive://'
    
    # 2026 Protocol Expansion: Check for primary Kaggle mirror first
    if ($KaggleRef) {
        $kagRepoId = $KaggleRef.Replace('kaggle://', '')
        $kagDn = $kagRepoId.Split('/')[-1]
        $kagFold = Join-Path $SharedPath $kagDn
        if (Test-Path $kagFold) { return "Extracted" }
    }
    $repoId = $Ref.Replace('hf://datasets/', 'hf://').Replace('hf://', '').Replace('gh://', '').Replace('kaggle://', '').Replace('gdrive://', '').Replace('local://raw-sets/', '').Replace('local://', '')
    if ($Ref -match 'competition:(.*)') { $repoId = $Matches[1] }
    $repoId = $repoId.Trim('/')
    
    $targetFile = $null
    if ($repoId -match ':' -and -not ($repoId -match '^[A-Za-z]:')) {
        $parts = $repoId.Split(':')
        $repoId = $parts[0]
        $targetFile = $parts[1]
        $dn = $repoId.Split('/')[-1]
    } else {
        # 2026: SOTA Path Normalization - extract the core slug from potentially deep hf/kaggle paths
        $dn = $repoId.Split('/')[-1]
    }
    
    $fold = Join-Path $SharedPath $dn
    $z = Join-Path $SharedPath ($dn + '.zip')
    
    # 2026: SOTA Fuzzy Matching - Search for normalized folder names
    $fuzzyFolders = Get-ChildItem $SharedPath -Directory | Where-Object { $_.Name -match "^$dn" -or $dn -match "^$($_.Name)" }
    $foundFold = if ($fuzzyFolders) { $fuzzyFolders[0].FullName } else { $fold }

    $fCount = 0
    if (Test-Path $foundFold) {
        # 2026: SOTA Fast-Check - Don't count millions of files, just check folder existence + sentinel
        $fCount = 1
        # Check if the download was interrupted (HuggingFace incomplete blobs)
        if (Get-ChildItem -Path $foundFold -Recurse -Force -Filter "*incomplete*" -ErrorAction SilentlyContinue | Select-Object -First 1) {
            $fCount = 0
        }
        # Secondary HuggingFace check removed: hf_manager.py writes directly to local_dir without snapshots
        # Check for metadata if it's a structural repo (HF/GH)
        if ($fCount -eq 0 -and (Test-Path (Join-Path $foundFold ".git"))) { $fCount = 1 }
    }
    
    if ($isHF -or $isGH -or $kaggleSource -or $isGDrive) {
        if ($fCount -gt 0) { return "Extracted" }
        if (Test-Path $z) { return "ZipOnly" }
        return "Missing"
    } else {
        if ($fCount -gt 0) { return "Extracted" }
        if (Test-Path $z) { return "ZipOnly" }
        return "Missing"
    }
} # end function Get-RefStatus

function Test-MissingDatasets {
    param([string[]]$TargetModels = $null, [switch]$SkipIfCompiled)
    $RegData = Get-RegData
    if (!$RegData) {
        return @()
    }
    $Missing = @()
    $ModelsToCheck = $TargetModels
    if ($null -eq $TargetModels) {
        $ModelsToCheck = @($RegData.datasets.PSObject.Properties.Name)
    }
    $Prefix = $RegData._registry_metadata.name_prefix
    $Suffix = $RegData._registry_metadata.name_suffix
    foreach ($C in $ModelsToCheck) {
        $ds_info = $RegData.datasets.$C
        $Slug = $ds_info.name
        $ManifoldName = $Prefix + $Slug + $Suffix
        $ManifoldPath = Join-Path $Out $ManifoldName
        $InfoPath = Join-Path $ManifoldPath "dataset_info.yaml"
        
        # 2026: Check if the FULL manifold exists locally or on Kaggle mirror
        if ($SkipIfCompiled -and (Test-Path $InfoPath)) {
            Write-Host "  [OK] $Slug manifold verified locally." -ForegroundColor Green
            continue
        }

        $kagRef = $ds_info.kaggle_ref
        if ($kagRef) {
            $Stat = Get-RefStatus -Ref $kagRef -SharedPath $Raw -KaggleRef $kagRef
            if ($Stat -eq "Extracted") {
                Write-Host "  [OK] $Slug manifold verified via Kaggle Mirror." -ForegroundColor Green
                continue
            }
        }

        foreach ($E in $ds_info.refs) {
            $Stat = Get-RefStatus -Ref $E.ref -SharedPath $Raw
            if ($Stat -match 'Missing|ZipOnly|HF_Partial') {
                if ($Missing -notcontains $E.ref) {
                    $Missing += $E.ref
                }
            }
        }
    }
    return $Missing
}

function Start-Acquisition {
    param([string[]]$ForcedRefs = $null)
    
    $Cred = @{ username = 'lemgenda'; key = 'd28f8f8b8eef9a8f688e8b8c7c9e8e8' }
    $KPath = Join-Path $env:USERPROFILE '.kaggle'
    if (!(Test-Path $KPath)) { [void](New-Item -ItemType Directory -Path $KPath -Force) }
    [System.IO.File]::WriteAllText((Join-Path $KPath 'kaggle.json'), ($Cred | ConvertTo-Json))
    
    $TokenPath = Join-Path $PSScriptRoot '.kaggle_token'
    if (Test-Path $TokenPath) {
        Copy-Item -Path $TokenPath -Destination (Join-Path $KPath 'access_token') -Force
    }

    $RegData = Get-RegData
    if (!$RegData) { return }

    $global:LogBuffer = @()
    $DatasetNames = @($RegData.datasets.PSObject.Properties.Name)
    $DoExtract = $true
    $ProcessList = @()
    
    if ($null -eq $ForcedRefs) {
        Write-Host "`n--- SELECT DATASET TO ACQUIRE ---" -ForegroundColor Cyan
        for ($i=0; $i -lt $DatasetNames.Count; $i++) {
            Write-Host "$($i+1). $($DatasetNames[$i])"
        }
        Write-Host "a. All Datasets"
        $Sel = Read-Host "Selection"
        
        $TargetDatasets = @()
        if ($Sel -eq 'a') {
            $TargetDatasets = $DatasetNames
        } else {
            $Idx = [int]$Sel - 1
            if ($Idx -ge 0 -and $Idx -lt $DatasetNames.Count) {
                $TargetDatasets += $DatasetNames[$Idx]
            } else {
                Write-Host "Invalid selection." -Fore Red
                return
            }
        }
        
        Write-Host "`nOptions:"
        Write-Host "1. Download only"
        Write-Host "2. Download and extract"
        $Opt = Read-Host "Selection"
        $DoExtract = ($Opt -eq '2')
        
        Write-Host "`n--- SOURCE STATUS CHECK ---" -ForegroundColor Cyan
        foreach ($td in $TargetDatasets) {
            Write-Host "[$td]" -ForegroundColor Yellow
            foreach ($E in $RegData.datasets.$td.refs) {
                # 2026: Consolidated Status Logic (v5.4)
                $Stat = Get-RefStatus -Ref $E.ref -SharedPath $Raw
                $Slug = $E.ref.Split('/')[-1].Split(':')[0].Replace('.tgz','').Replace('.zip','')
                
                $AlreadyQueued = $ProcessList | Where-Object { $_.Ref -eq $E.ref }
                if (!$AlreadyQueued) {
                    switch ($Stat) {
                        "Extracted" { Write-Host "  [OK] $Slug (Verified)" -ForegroundColor Green }
                        "HF_Partial" { 
                            Write-Host "  [SYNC] $Slug (Resuming HF)" -ForegroundColor Yellow
                            $ProcessList += @{ Ref = $E.ref; Action = 'Download' }
                        }
                        "ZipOnly" {
                            if ($DoExtract) {
                                Write-Host "  [UNPACK] $Slug (Ready)" -ForegroundColor Magenta
                                $ProcessList += @{ Ref = $E.ref; Action = 'UnpackOnly' }
                            } else {
                                Write-Host "  [SKIP] $Slug (Zip exists)" -ForegroundColor DarkGray
                            }
                        }
                        Default {
                            Write-Host "  [DL QUEUED] $Slug (Missing)" -ForegroundColor Red
                            $ProcessList += @{ Ref = $E.ref; Action = 'Download' }
                        }
                    }
                }
            }
        }
    } else {
        # Auto mode (Pre-flight check)
        $DoExtract = $true
        foreach ($r in $ForcedRefs) {
            $Stat = Get-RefStatus -Ref $r -SharedPath $Raw
            if ($Stat -eq "ZipOnly") {
                $ProcessList += @{ Ref = $r; Action = 'UnpackOnly' }
            } elseif ($Stat -match 'Missing|HF_Partial') {
                $ProcessList += @{ Ref = $r; Action = 'Download' }
            }
        }
    }

    if ($ProcessList.Count -eq 0) {
        Write-Host "`n  [OK] All required sources are already acquired/extracted!" -ForegroundColor Green
        if ($null -eq $ForcedRefs) { Read-Host "Press Enter to return" }
        return
    }

    Write-Host "`n--- ACQUISITION MANIFEST ---" -ForegroundColor Yellow

    if (!(Test-Path $Raw)) { [void](New-Item -ItemType Directory -Path $Raw -Force) }

    $UniqueDatasets = @{}
    foreach ($Item in $ProcessList) {
        $Slug = $Item.Ref.Replace('hf://', '').Split('/')[-1]
        $InitStatus = 'Queued'
        if ($Item.Action -eq 'UnpackOnly') { $InitStatus = 'DOWNLOADED' }
        $UniqueDatasets[$Item.Ref] = @{
            Ref = $Item.Ref
            Slug = $Slug
            Status = $InitStatus
            JobId = $null
            ProgressId = 0
        }
    }

    $UniqueList = @($UniqueDatasets.Values | Sort-Object Ref)
    $MaxJobs = 3
    $BaseId = 100
    
    $D = 0; $T = @($UniqueList).Count
    while ($D -lt $T) {
        $AnyUnpacking = @($UniqueList | Where-Object { $_.Status -eq 'UNPACKING' })
        
        for ($i=0; $i -lt $MaxJobs; $i++) {
            $SlotOwnedBy = @($UniqueList | Where-Object { $_.ProgressId -eq ($BaseId + $i) }) | Select-Object -First 1
            if (!$SlotOwnedBy) {
                $NextDl = @($UniqueList | Where-Object { $_.Status -eq 'Queued' }) | Select-Object -First 1
                if ($NextDl) {
                    $NextDl.ProgressId = $BaseId + $i
                    $NextDl.Status = 'Starting DL'
                    if ($NextDl.Ref -match 'hf://') {
                        $NextDl.JobId = (Start-Job -ScriptBlock $HuggingFaceSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $hfManagerPath).Id
                    } elseif ($NextDl.Ref -match 'gh://') {
                        $NextDl.JobId = (Start-Job -ScriptBlock $GHSourceSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $ghManagerPath).Id
                    } elseif ($NextDl.Ref -match 'kaggle://') {
                        $NextDl.JobId = (Start-Job -ScriptBlock $DownloadSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $kagManagerPath).Id
                    } elseif ($NextDl.Ref -match 'gdrive://') {
                        $NextDl.JobId = (Start-Job -ScriptBlock $GDriveSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $gdManagerPath).Id
                    } else {
                        # Legacy fallback
                        $NextDl.JobId = (Start-Job -ScriptBlock $DownloadSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $kagManagerPath).Id
                    }
                }
            }
        }
        
        if (!$AnyUnpacking -and $DoExtract) {
            $NextUnpack = @($UniqueList | Where-Object { $_.Status -eq 'DOWNLOADED' }) | Select-Object -First 1
            if ($NextUnpack) {
                $UnpackSlot = @($UniqueList | Where-Object { $_.ProgressId -gt 0 }).Count
                if ($UnpackSlot -lt ($MaxJobs + 1)) {
                    $NextUnpack.ProgressId = $BaseId + $MaxJobs 
                    $NextUnpack.Status = 'Starting UP'
                    $NextUnpack.JobId = (Start-Job -ScriptBlock $UnpackSB -ArgumentList $NextUnpack.Ref, $Raw, $Vpy).Id
                }
            }
        } elseif (!$DoExtract) {
            foreach ($ti in @($UniqueList | Where-Object { $_.Status -eq 'DOWNLOADED' })) {
                $ti.Status = 'COMPLETED'
                $ti.ProgressId = 0
            }
        }

        foreach ($ti in @($UniqueList | Where-Object { $null -ne $_.JobId })) {
            $jr = Get-Job -Id $ti.JobId
            if ($null -ne $jr) {
                # Capture everything (tqdm writes to stderr/stdout depending on config)
                $Outputs = @($jr | Receive-Job)
                foreach ($ls in $Outputs) {
                    if ($ls -match 'STATUS:(.*)') { $ti.Status = $Matches[1] }
                    elseif ($ls -match 'NOTIFICATION:(.*)') {
                        if ($global:LogBuffer.Count -ge 5) { $global:LogBuffer = $global:LogBuffer[1..4] }
                        $global:LogBuffer += "  [!] $($Matches[1])"
                    }
                    elseif ($ls -match 'RESULT:(.*)') { 
                        $ti.Status = $Matches[1]
                        if ($ti.Status -eq 'DOWNLOADED') {
                             $ti.ProgressId = 0
                             $ti.JobId = $null
                        }
                        if ($ti.Status -match 'COMPLETED|FOUND') { 
                            $ti.ProgressId = 0
                        }
                        if ($ti.Status -eq 'FAILED') { 
                            $ti.ProgressId = 0 
                            # 2026: SOTA Failure Guard - Ensure a clean line for the error debrief
                            if ($global:LogBuffer.Count -ge 5) { $global:LogBuffer = $global:LogBuffer[1..4] }
                            $global:LogBuffer += "  [!] Job FAILED: $($ti.Slug) (Source Offline or Restricted)"
                            Start-Sleep -Seconds 1 # Force UI to settle
                        }
                    } else {
                        if (-not [string]::IsNullOrWhiteSpace($ls)) {
                            $LineStr = ([string]$ls) -replace "`r", ""
                            
                            # 2026: SOTA Buffered Logger - Store notifications to prevent console overlap
                            if ($LineStr -match '^\[.*\]' -and $LineStr -notmatch '\[DL\]|%') {
                                if ($global:LogBuffer.Count -ge 5) { $global:LogBuffer = $global:LogBuffer[1..4] }
                                $global:LogBuffer += "    $($ti.Slug)> $LineStr"
                            } elseif ($LineStr -match '\[.*\]' -or $LineStr -match '%') {
                                $ti.Status = $LineStr.Trim()
                            }
                        }
                    }
                }
                if ($jr.State -eq 'Completed' -and $ti.Status -notmatch 'COMPLETED|FOUND|FAILED|MISSING|DOWNLOADED|UNPACKING|SUCCESS|processed' ) { 
                    $ti.Status = 'Done'; $ti.ProgressId = 0 
                }
            }
        }

        # --- AESTHETIC PROGRESS RENDERER ---
        $D = @($UniqueList | Where-Object { $_.Status -match 'COMPLETED|Done|FOUND|FAILED|MISSING|SUCCESS|processed' }).Count
        $P = [math]::Round(($D / $T) * 100)
        
        # Build the dynamic status display
        $DisplayLines = @()
        if ($global:LogBuffer) {
            $DisplayLines += "--- EVENT LOG ---"
            foreach ($log in $global:LogBuffer) { $DisplayLines += $log }
            $DisplayLines += "-----------------"
        }
        $DisplayLines += "[OVERALL] $D/$T ($P%)"
        for ($i=0; $i -lt $MaxJobs; $i++) {
            $Active = @($UniqueList | Where-Object { $_.ProgressId -eq ($BaseId + $i) }) | Select-Object -First 1
            if ($Active) {
                $SlugPad = $Active.Slug.PadRight(20).Substring(0, 20)
                $DisplayLines += "   Job $($i+1): $SlugPad | $($Active.Status)"
            }
        }

        # Use Absolute-Coordinate management with Buffer Safety
        $StartTop = [System.Console]::CursorTop
        $BufferHeight = [System.Console]::BufferHeight
        $WindowWidth = [System.Console]::WindowWidth - 1
        
        # Pre-scroll check: Ensure we have enough room in the buffer
        if (($StartTop + $DisplayLines.Count) -ge $BufferHeight) {
            # Scroll the terminal up by printing newlines
            $Needed = ($StartTop + $DisplayLines.Count) - $BufferHeight + 1
            for ($k=0; $k -le $Needed; $k++) { Write-Host "" }
            $StartTop = [System.Console]::CursorTop - $DisplayLines.Count - 1
            if ($StartTop -lt 0) { $StartTop = 0 }
        }

        for ($i=0; $i -lt $DisplayLines.Count; $i++) {
            $Line = $DisplayLines[$i]
            if ($Line.Length -gt $WindowWidth) { $Line = $Line.Substring(0, $WindowWidth) }
            
            $TargetTop = $StartTop + $i
            if ($TargetTop -lt $BufferHeight) {
                [System.Console]::SetCursorPosition(0, $TargetTop)
                # Dynamic Color SOTA Logic
                $Color = 'Gray'
                if ($Line -match 'OVERALL') { $Color = 'Cyan' }
                elseif ($Line -match '\[DL\]') { $Color = 'Yellow' }
                elseif ($Line -match '\[UNPACK\]|Extracting|processed|SUCCESS|Done') { $Color = 'Green' }
                
                Write-Host $Line.PadRight($WindowWidth) -ForegroundColor $Color -NoNewline
            }
        }
        
        if ($D -ge $T) { break }
        Start-Sleep -Milliseconds 500
        
        # Reset cursor back to the top of the block
        if ($StartTop -lt $BufferHeight) {
            [System.Console]::SetCursorPosition(0, $StartTop)
        }
    } # end while


    Write-Host "`n================================================================================" -ForegroundColor Cyan
    Write-Host " [MISSION DEBRIEF] ACQUISITION SUMMARY" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    
    $Successes = @($UniqueList | Where-Object { $_.Status -match 'COMPLETED|Done|FOUND|SUCCESS|processed' })
    $Failures = @($UniqueList | Where-Object { $_.Status -eq 'FAILED' })

    if ($Successes.Count -gt 0) {
        Write-Host "`n [SUCCESSES]" -ForegroundColor Green
        $Successes | ForEach-Object {
            $LocalDir = Join-Path $Raw $_.Slug
            Write-Host "  - $($_.Slug.PadRight(30)) | Path: $LocalDir"
        }
    }

    if ($Failures.Count -gt 0) {
        Write-Host "`n [FAILURES]" -ForegroundColor Red
        $Failures | ForEach-Object {
            Write-Host "  - $($_.Slug.PadRight(30)) | Ref: $($_.Ref)"
        }
    }
    Write-Host "`n================================================================================" -ForegroundColor Cyan

    Write-Host "`n[OK] ACQUISITION MISSION ENDED." -ForegroundColor Green
    if ($null -eq $ForcedRefs) { Read-Host "Press Enter to return to menu" }
} # end function Start-Acquisition

while ($true) {
    # 2026: Resilient Header - Safer than Clear-Host in background/remote shells
    if ($Host.Name -eq 'ConsoleHost') { Clear-Host } else { Write-Host "`n`n`n" }
    Write-Host '--- LEMGENDARY DATASETS HUB v5.2 ---' -ForegroundColor Yellow
    
    # 2026 Resilience: Use cached CUDA status to prevent 5-second menu delays
    if ($null -eq $global:CudaReady) {
        $CudaStatus = & $Vpy -c "import torch; print('OK' if torch.cuda.is_available() else 'OFF')"
        $global:CudaReady = ($CudaStatus -match "OK")
    }
    
    if (!$global:CudaReady) {
        Write-Host "[SYSTEM] NO CUDA DETECTED! AI tasks will run on CPU (SLOW)." -ForegroundColor Red
    } else {
        Write-Host "[SYSTEM] CUDA READY (GPU Accelerated)" -ForegroundColor Green
    }

    Show-Stats
    Write-Host '1. [COMPILE] Build new SOTA manifold' -ForegroundColor Gray
    Write-Host '2. [REDUCE]  Create downsampled variant' -ForegroundColor Gray
    Write-Host '3. [SYNC]    Push compiled manifold to Kaggle' -ForegroundColor Gray
    Write-Host 'Q. [QUIT]    Exit Dashboard' -ForegroundColor Gray
    $I = Read-Host 'Selection'
    if ($I -eq '1') {
        $RegData = Get-RegData
        $DatasetNames = @($RegData.datasets.PSObject.Properties.Name)
        
        Write-Host "`n--- SELECT DATASET TO COMPILE ---" -ForegroundColor Cyan
        $Prefix = $RegData._registry_metadata.name_prefix
        $Suffix = $RegData._registry_metadata.name_suffix
        for ($i=0; $i -lt $DatasetNames.Count; $i++) {
            $dsName = $DatasetNames[$i]
            $slug = $RegData.datasets.$dsName.name
            $ManifoldPath = Join-Path $Out ($Prefix + $slug + $Suffix)
            if (Test-Path (Join-Path $ManifoldPath "dataset_info.yaml")) {
                Write-Host "$($i+1). $dsName [COMPILED]" -ForegroundColor Green
            } else {
                Write-Host "$($i+1). $dsName" -ForegroundColor Gray
            }
        }
        Write-Host "a. All Datasets"
        $Sel = Read-Host "Selection"
        
        $TargetModels = @()
        if ($Sel -eq 'a') {
            $TargetModels = $DatasetNames
        } else {
            $Idx = -1
            try { $Idx = [int]$Sel - 1 } catch {}
            if ($Idx -ge 0 -and $Idx -lt $DatasetNames.Count) {
                $TargetModels += $DatasetNames[$Idx]
            } else {
                Write-Host "Invalid selection. Please enter a valid number or 'a'." -Fore Red
                Start-Sleep -Seconds 2
                continue
            }
        }
        
        if ($TargetModels -contains 'forex_predictor') {
            $RawForex = Join-Path $Raw 'forex'
            $TsForex = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) 'lemgendary-training-suite') 'data\forex'
            if (!(Test-Path $RawForex) -and !(Test-Path $TsForex)) {
                Write-Host "`n================================================================================" -ForegroundColor Yellow
                Write-Host " 📈 METATRADER 5 (MT5) SETUP INSTRUCTIONS FOR FOREX MANIFOLD DATASET" -ForegroundColor Cyan
                Write-Host "================================================================================" -ForegroundColor Yellow
                Write-Host " 1. Download MetaTrader 5: https://www.metatrader5.com/en/download" -ForegroundColor Green
                Write-Host " 2. Install & Open MetaTrader 5." -ForegroundColor White
                Write-Host " 3. Register a free Demo Account: File -> Open an Account -> 'MetaQuotes-Demo'." -ForegroundColor White
                Write-Host " 4. Note down your demo account credentials (Login Number, Password, Server)." -ForegroundColor White
                Write-Host " 5. Run live data download command in terminal:" -ForegroundColor White
                Write-Host "    python data/mt5_pipeline.py --mode download --login <ACCOUNT> --password <PASS> --server MetaQuotes-Demo" -ForegroundColor Yellow
                Write-Host "================================================================================`n" -ForegroundColor Yellow
            }
        }
        
        # Check missing sets (with Manifold-Aware skip)
        $Missing = Test-MissingDatasets -TargetModels $TargetModels
        
        if ($Missing.Count -gt 0) {
            Write-Host "`n[WARNING] Some raw datasets are missing or empty for this compilation:" -ForegroundColor Yellow
            $Missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
            $Choice = Read-Host "`nWould you like to acquire missing sets before compiling? (Y/N)"
            if ($Choice -match '^y') {
                Start-Acquisition -ForcedRefs $Missing
            } else {
                Write-Host "Proceeding with missing data... might fail if manifold isn't sufficient." -ForegroundColor Red
                Start-Sleep -Seconds 2
            }
        }
        
        # Ask for overrides
        $MaxSize = Read-Host "Enter Max Size GB [Default: $($RegData._registry_metadata.global_constraints.max_size_gb)]"
        if ([string]::IsNullOrWhiteSpace($MaxSize)) { $MaxSize = $RegData._registry_metadata.global_constraints.max_size_gb }
        
        $Suffix = Read-Host "Enter Suffix [Default: $($RegData._registry_metadata.name_suffix)]"
        if ([string]::IsNullOrWhiteSpace($Suffix)) { $Suffix = $RegData._registry_metadata.name_suffix }

        $Workers = Read-Host "Enter Number of Workers [Default: Auto]"
        $WorkerArg = @()
        if (![string]::IsNullOrWhiteSpace($Workers)) { $WorkerArg = @("--workers", $Workers) }
        Start-Sleep -Milliseconds 100
        Start-Sleep -Milliseconds 100
        
        foreach ($tm in $TargetModels) {
            Write-Host "`n[SYSTEM] Compiling dataset model: $tm" -ForegroundColor Cyan
            & $Vpy compiler-pipeline.py --model $tm --max_gb $MaxSize --suffix $Suffix @WorkerArg
            
            # Post-Compile Verification
            $OutFolder = Join-Path (Get-Location) $OutFolderName
            if ($RegData._registry_metadata.version) {
                $OutFolder = Join-Path $OutFolder "v_$($RegData._registry_metadata.version)"
            }
            
            if (Test-Path (Join-Path $OutFolder "README.md")) {
                Write-Host "  [OK] Dataset compiled successfully!" -ForegroundColor Green
            }
        }
        # 2026: Finalized Global Cleanup (Moved outside loop to prevent I/O collisions)
        Write-Host "`n[JANITOR] Purging compilation temp files..." -ForegroundColor Gray
        & $Vpy compiler-pipeline.py --cleanup
    }
    elseif ($I -eq '2') {
        & $Vpy compiler-pipeline.py --reduce
    }
    elseif ($I -eq '3') {
        $RegData = Get-RegData
        $DatasetNames = @($RegData.datasets.PSObject.Properties.Name)
        
        Write-Host "`n--- SELECT MANIFOLD TO SYNC TO KAGGLE ---" -ForegroundColor Cyan
        for ($i=0; $i -lt $DatasetNames.Count; $i++) {
            Write-Host "$($i+1). $($DatasetNames[$i])"
        }
        $Sel = Read-Host "Selection"
        $Idx = -1
        try { $Idx = [int]$Sel - 1 } catch {}
        if ($Idx -ge 0 -and $Idx -lt $DatasetNames.Count) {
            $tm = $DatasetNames[$Idx]
            $ds_info = $RegData.datasets.$tm
            $Slug = $ds_info.name
            $Prefix = $RegData._registry_metadata.name_prefix
            $Suffix = $RegData._registry_metadata.name_suffix
            $ManifoldName = $Prefix + $Slug + $Suffix
            $ManifoldPath = Join-Path $Out $ManifoldName
            
            if (!(Test-Path $ManifoldPath)) {
                Write-Host "  [ERROR] Compiled manifold not found at $ManifoldPath" -Fore Red
                Read-Host "Press Enter to return"
                continue
            }
            
            $KagHandle = $ds_info.kaggle_ref
            if (!$KagHandle) {
                $KagHandle = Read-Host "Enter Kaggle Dataset Handle (e.g. username/dataset-name)"
            } else {
                $KagHandle = $KagHandle.Replace("kaggle://", "")
                Write-Host "  [INFO] Target Kaggle Handle: $KagHandle" -ForegroundColor Gray
            }
            
            if ($KagHandle) {
                Write-Host "`n[SYNC] Initiating Hybrid Sync for $ManifoldName..." -ForegroundColor Cyan
                & $Vpy $kagManagerPath --action upload --repo_id $KagHandle --output_dir $ManifoldPath 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "[OK] [SYNC] Manifold successfully synchronized!" -ForegroundColor Green
                } else {
                    Write-Host "[FAILED] [SYNC] Synchronization failed." -ForegroundColor Red
                }
            }
            Read-Host "Press Enter to return"
        }
    }
    elseif ($I -match '^q') { break }
    else {
        Write-Host "Command finished or unrecognized input." -ForegroundColor DarkGray
    }
    
    # 2026: SOTA Debugging Pause - Prevent screen clear if there was an error
    Write-Host "`nPress Enter to return to Dashboard..." -ForegroundColor Yellow
    Read-Host

}
