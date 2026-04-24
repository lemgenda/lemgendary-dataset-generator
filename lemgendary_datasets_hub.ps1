# ==========================================
# LemGendary Dataset Hub (SOTA 2026 Dashboard)
# ==========================================

$ErrorActionPreference = 'Stop'
$Vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$Reg = Join-Path $PSScriptRoot 'unified_data.yaml'
$Raw = Join-Path $PSScriptRoot 'raw-sets'
$hfManagerPath = Join-Path $PSScriptRoot 'hf_manager.py'

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
                    $Txt = [System.IO.File]::ReadAllText($IdxPath)
                    $Djson = $Txt | ConvertFrom-Json
                    $Cnt = $Djson.Count
                    Write-Host ('  [STATS] Latest: ' + $Lat.Name + ' | Total: ' + $Cnt) -ForegroundColor Cyan
                } catch { }
            }
        }
    }
}

function Test-MissingDatasets {
    param([string[]]$TargetModels = $null)
    $RegData = Get-RegData
    if (!$RegData) { return @() }
    
    $Missing = @()
    
    $ModelsToCheck = $TargetModels
    if ($null -eq $TargetModels) {
        $ModelsToCheck = @($RegData.datasets.PSObject.Properties.Name)
    }
    
    foreach ($C in $ModelsToCheck) {
        foreach ($E in $RegData.datasets.$C.refs) {
            $Slug = $E.ref.Replace('hf://', '').Split('/')[-1]
            $P = Join-Path $Raw $Slug
            if (!(Test-Path $P) -or (Get-ChildItem $P -File -Recurse).Count -eq 0) {
                if ($Missing -notcontains $E.ref) { $Missing += $E.ref }
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

    $RegData = Get-RegData
    if (!$RegData) { return }

    $DatasetNames = @($RegData.datasets.PSObject.Properties.Name)
    $DoExtract = $true
    
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
        
        $ForcedRefs = @()
        foreach ($td in $TargetDatasets) {
            foreach ($E in $RegData.datasets.$td.refs) {
                if ($ForcedRefs -notcontains $E.ref) { $ForcedRefs += $E.ref }
            }
        }
    } else {
        # Auto mode
        $DoExtract = $true
    }

    Write-Host "`n--- ACQUISITION MANIFEST ---" -ForegroundColor Yellow

    if (!(Test-Path $Raw)) { [void](New-Item -ItemType Directory -Path $Raw -Force) }

    $UniqueDatasets = @{}
    foreach ($Ref in $ForcedRefs) {
        $Slug = $Ref.Replace('hf://', '').Split('/')[-1]
        if (!$UniqueDatasets.ContainsKey($Ref)) {
            $UniqueDatasets[$Ref] = @{
                Ref = $Ref
                Slug = $Slug
                Status = 'Queued'
                JobId = $null
                ProgressId = 0
            }
        }
    }

    if ($UniqueDatasets.Count -eq 0) {
        Write-Host "  [OK] No missing datasets to acquire." -ForegroundColor Green
        return
    }

    $UniqueList = $UniqueDatasets.Values | Sort-Object Ref
    $MaxJobs = 3
    $BaseId = 100
    
    $DownloadSB = {
        param($ds, $sharedPath, $vpy)
        $isC = $ds -match 'competition'
        $ref = $ds; if ($ds -match 'competition:(.*)') { $ref = $Matches[1] }
        $dn = $ref.Split('/')[-1]
        $z = Join-Path $sharedPath ($dn + '.zip')
        
        $ArchMgr = Join-Path (Split-Path $vpy -Parent | Split-Path -Parent | Split-Path -Parent) 'archive_manager.py'
        
        if (Test-Path $z) {
            Write-Output "STATUS:VERIFYING"
            & $vpy $ArchMgr --zip $z --dest "." --action verify 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Output "NOTIFICATION:Zip already exists and is valid: $dn.zip"
                Write-Output "RESULT:DOWNLOADED"
                return
            } else {
                Write-Output "NOTIFICATION:Zip corrupted, re-downloading: $dn.zip"
                Remove-Item $z -Force -ErrorAction SilentlyContinue
            }
        }
        
        Write-Output "STATUS:DOWNLOADING"
        if ($isC) { kaggle competitions download -c $ref -p $sharedPath --quiet 2>&1 } else { kaggle datasets download -d $ref -p $sharedPath --quiet 2>&1 }
        
        if (Test-Path $z) {
             Write-Output "RESULT:DOWNLOADED"
        } else {
             Write-Output "RESULT:FAILED"
        }
    }

    $HuggingFaceSB = {
        param($ds, $sharedPath, $vpy, $hfManager)
        $repoId = $ds.Replace('hf://', '')
        $dn = $repoId.Split('/')[-1]
        $outFold = Join-Path $sharedPath $dn
        
        Write-Output "STATUS:HF-PULLING"
        & $vpy $hfManager --repo_id $repoId --output_dir $outFold --repo_type dataset 2>&1
        
        if ((Get-ChildItem $outFold -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0) {
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
            & $vpy $ArchMgr --zip $z --dest $fold --action extract 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Output "NOTIFICATION:Extraction Finished & Zip Deleted: $dn"
                Write-Output "RESULT:COMPLETED"
            } else {
                Write-Output "RESULT:FAILED"
            }
        } catch {
            Write-Output "NOTIFICATION:Extraction Failed for $dn : $($_.Exception.Message)"
            Write-Output "RESULT:FAILED"
        }
    }

    $D = 0; $T = $UniqueList.Count
    while ($D -lt $T) {
        $AnyUnpacking = $UniqueList | Where-Object { $_.Status -eq 'UNPACKING' }
        
        for ($i=0; $i -lt $MaxJobs; $i++) {
            $SlotOwnedBy = $UniqueList | Where-Object { $_.ProgressId -eq ($BaseId + $i) } | Select-Object -First 1
            if (!$SlotOwnedBy) {
                $NextDl = $UniqueList | Where-Object { $_.Status -eq 'Queued' } | Select-Object -First 1
                if ($NextDl) {
                    $NextDl.ProgressId = $BaseId + $i
                    $NextDl.Status = 'Starting DL'
                    if ($NextDl.Ref -match 'hf://') {
                        $NextDl.JobId = (Start-Job -ScriptBlock $HuggingFaceSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $hfManagerPath).Id
                    } else {
                        $NextDl.JobId = (Start-Job -ScriptBlock $DownloadSB -ArgumentList $NextDl.Ref, $Raw, $Vpy).Id
                    }
                }
            }
        }
        
        if (!$AnyUnpacking -and $DoExtract) {
            $NextUnpack = $UniqueList | Where-Object { $_.Status -eq 'DOWNLOADED' } | Select-Object -First 1
            if ($NextUnpack) {
                $UnpackSlot = ($UniqueList | Where-Object { $_.ProgressId -gt 0 }).Count
                if ($UnpackSlot -lt ($MaxJobs + 1)) {
                    $NextUnpack.ProgressId = $BaseId + $MaxJobs 
                    $NextUnpack.Status = 'Starting UP'
                    $NextUnpack.JobId = (Start-Job -ScriptBlock $UnpackSB -ArgumentList $NextUnpack.Ref, $Raw, $Vpy).Id
                }
            }
        } elseif (!$DoExtract) {
            foreach ($ti in ($UniqueList | Where-Object { $_.Status -eq 'DOWNLOADED' })) {
                $ti.Status = 'COMPLETED'
                $ti.ProgressId = 0
            }
        }

        foreach ($ti in ($UniqueList | Where-Object { $null -ne $_.JobId })) {
            $jr = Get-Job -Id $ti.JobId
            if ($null -ne $jr) {
                foreach ($ls in ($jr | Receive-Job)) {
                    if ($ls -match 'STATUS:(.*)') { $ti.Status = $Matches[1] }
                    if ($ls -match 'NOTIFICATION:(.*)') { Write-Host "  [!] $($Matches[1])" -ForegroundColor Cyan }
                    if ($ls -match 'RESULT:(.*)') { 
                        $ti.Status = $Matches[1]
                        if ($ti.Status -eq 'DOWNLOADED') {
                             $ti.ProgressId = 0
                             $ti.JobId = $null
                        }
                        if ($ti.Status -match 'COMPLETED|FOUND') { 
                            $ti.ProgressId = 0
                        }
                        if ($ti.Status -eq 'FAILED') { $ti.ProgressId = 0 }
                    }
                }
                if ($jr.State -eq 'Completed' -and $ti.Status -notmatch 'COMPLETED|FOUND|FAILED|MISSING|DOWNLOADED|UNPACKING' ) { 
                    $ti.Status = 'Done'; $ti.ProgressId = 0 
                }
            }
        }

        $D = ($UniqueList | Where-Object { $_.Status -match 'COMPLETED|Done|FOUND|FAILED|MISSING' }).Count
        $P = [math]::Round(($D / $T) * 100)
        if ($P -gt 0) { Write-Progress -Activity "OVERALL ACQUISITION" -Status "$D/$T Complete ($P%)" -PercentComplete $P -Id 1 }
        
        for ($i=0; $i -lt $MaxJobs; $i++) {
            $Active = $UniqueList | Where-Object { $_.ProgressId -eq ($BaseId + $i) } | Select-Object -First 1
            if ($Active) {
                $StatStr = [string]$Active.Status
                Write-Progress -Id ($BaseId + $i) -ParentId 1 -Activity "Job $($i+1): $($Active.Slug)" -Status $StatStr -PercentComplete -1
            } else {
                Write-Progress -Id ($BaseId + $i) -ParentId 1 -Activity "Job $($i+1): Idle" -Status "Waiting..." -Completed
            }
        }

        if ($D -ge $T) { break }
        Start-Sleep -Seconds 1
    }

    Write-Progress -Id 1 -Activity "OVERALL ACQUISITION" -Completed
    Write-Host '  [OK] ACQUISITION MISSION ENDED.' -ForegroundColor Green
    Read-Host "Press Enter to return to menu"
}

while ($true) {
    Clear-Host
    Write-Host '--- LEMGENDARY DATASETS HUB v5.2 ---' -ForegroundColor Yellow
    Show-Stats
    Write-Host '1. Acquire  2. Compile  Q. Quit'
    $I = Read-Host 'Selection'
    if ($I -eq '1') { Start-Acquisition }
    elseif ($I -eq '2') {
        $RegData = Get-RegData
        $DatasetNames = @($RegData.datasets.PSObject.Properties.Name)
        
        Write-Host "`n--- SELECT DATASET TO COMPILE ---" -ForegroundColor Cyan
        for ($i=0; $i -lt $DatasetNames.Count; $i++) {
            Write-Host "$($i+1). $($DatasetNames[$i])"
        }
        Write-Host "a. All Datasets"
        $Sel = Read-Host "Selection"
        
        $TargetModels = @()
        if ($Sel -eq 'a') {
            $TargetModels = $DatasetNames
        } else {
            $Idx = [int]$Sel - 1
            if ($Idx -ge 0 -and $Idx -lt $DatasetNames.Count) {
                $TargetModels += $DatasetNames[$Idx]
            } else {
                Write-Host "Invalid selection." -Fore Red
                continue
            }
        }
        
        # Check missing sets
        $Missing = Test-MissingDatasets -TargetModels $TargetModels
        
        if ($Missing.Count -gt 0) {
            Write-Host "`n[WARNING] Some raw datasets are missing or empty for this compilation:" -ForegroundColor Yellow
            $Missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
            $Choice = Read-Host "`nWould you like to acquire missing sets before compiling? (Y/N)"
            if ($Choice -match '^y') {
                Start-Acquisition -ForcedRefs $Missing
            } else {
                Write-Host "Proceeding with missing data... might fail." -ForegroundColor Red
                Start-Sleep -Seconds 2
            }
        }
        
        # Ask for overrides
        $MaxSize = Read-Host "Enter Max Size GB [Default: $($RegData._registry_metadata.global_constraints.max_size_gb)]"
        if ([string]::IsNullOrWhiteSpace($MaxSize)) { $MaxSize = $RegData._registry_metadata.global_constraints.max_size_gb }
        
        $Suffix = Read-Host "Enter Suffix [Default: $($RegData._registry_metadata.name_suffix)]"
        if ([string]::IsNullOrWhiteSpace($Suffix)) { $Suffix = $RegData._registry_metadata.name_suffix }
        
        foreach ($tm in $TargetModels) {
            Write-Host "`n[SYSTEM] Compiling dataset model: $tm" -ForegroundColor Cyan
            & $Vpy compiler-pipeline.py --model $tm --max_gb $MaxSize --suffix $Suffix
            
            # Post-Compile Verification
            $OutFolder = Join-Path (Get-Location) $OutFolderName
            if ($RegData._registry_metadata.version) {
                $OutFolder = Join-Path $OutFolder "v_$($RegData._registry_metadata.version)"
            }
            
            if (Test-Path (Join-Path $OutFolder "README.md")) {
                Write-Host "  [OK] Dataset compiled successfully!" -ForegroundColor Green
                Get-Content (Join-Path $OutFolder "README.md") -TotalCount 10
            }
            
            $DelChoice = Read-Host "`nWould you like to delete the raw source data for $tm to save space? (Y/N)"
            if ($DelChoice -match '^y') {
                foreach ($E in $RegData.datasets.$tm.refs) {
                    $Slug = $E.ref.Replace('hf://', '').Split('/')[-1]
                    $P = Join-Path $Raw $Slug
                    if (Test-Path $P) {
                        Remove-Item $P -Recurse -Force -ErrorAction SilentlyContinue
                        Write-Host "  Deleted raw source: $Slug" -ForegroundColor DarkGray
                    }
                }
            }
        }
        Read-Host "Press Enter to return to menu"
    }
    elseif ($I -match '^q') { break }
}
