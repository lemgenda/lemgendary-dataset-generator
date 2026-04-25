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

function Get-RefStatus {
    param($Ref, $SharedPath)
    $isHF = $Ref -match 'hf://'
    $repoId = $Ref.Replace('hf://', '')
    if ($Ref -match 'competition:(.*)') { $repoId = $Matches[1] }
    
    $targetFile = $null
    if ($repoId -match ':') {
        $parts = $repoId.Split(':')
        $repoId = $parts[0]
        $targetFile = $parts[1]
        # For surgical files, the folder is the slug (filename without ext)
        $dn = $targetFile.Replace('.tgz', '').Replace('.tar.gz', '').Replace('.zip', '')
    } else {
        $dn = $repoId.Split('/')[-1]
    }
    
    $fold = Join-Path $SharedPath $dn
    $z = Join-Path $SharedPath ($dn + '.zip')
    
    $fCount = 0
    if (Test-Path $fold) {
        $fCount = (Get-ChildItem $fold -File -Recurse -ErrorAction SilentlyContinue).Count
    }
    
    if ($isHF) {
        if ($fCount -gt 0) { return "Extracted" } # OK
        return "Missing"
    } else {
        if (!(Test-Path $z) -and $fCount -gt 0) { return "Extracted" }
        if (Test-Path $z) { return "ZipOnly" }
        return "Missing"
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
            $Stat = Get-RefStatus -Ref $E.ref -SharedPath $Raw
            if ($Stat -match 'Missing|ZipOnly|HF_Partial') {
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
                $Stat = Get-RefStatus -Ref $E.ref -SharedPath $Raw
                $Slug = $E.ref.Replace('hf://', '').Split('/')[-1]
                
                # Check if it's already in the process list to avoid duplicates
                $AlreadyQueued = $ProcessList | Where-Object { $_.Ref -eq $E.ref }
                if (!$AlreadyQueued) {
                    if ($Stat -eq "Extracted") {
                        Write-Host "  [OK] $Slug (Already Extracted)" -ForegroundColor Green
                    } elseif ($Stat -eq "HF_Partial") {
                        Write-Host "  [HF SYNC] $Slug (Running HF Manager to ensure completion)" -ForegroundColor Yellow
                        $ProcessList += @{ Ref = $E.ref; Action = 'Download' }
                    } elseif ($Stat -eq "ZipOnly") {
                        if ($DoExtract) {
                            Write-Host "  [UNPACK QUEUED] $Slug (Zip exists, needs extraction)" -ForegroundColor Magenta
                            $ProcessList += @{ Ref = $E.ref; Action = 'UnpackOnly' }
                        } else {
                            Write-Host "  [SKIP] $Slug (Zip exists, extraction not requested)" -ForegroundColor DarkGray
                        }
                    } else {
                        Write-Host "  [DL QUEUED] $Slug (Missing)" -ForegroundColor Red
                        $ProcessList += @{ Ref = $E.ref; Action = 'Download' }
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
        $InitStatus = if ($Item.Action -eq 'UnpackOnly') { 'DOWNLOADED' } else { 'Queued' }
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
    
    $DownloadSB = {
        param($ds, $sharedPath, $vpy, $hfManager)
        $isC = $ds -match 'competition'
        $ref = $ds; if ($ds -match 'competition:(.*)') { $ref = $Matches[1] }
        $dn = $ref.Split('/')[-1]
        $z = Join-Path $sharedPath ($dn + '.zip')
        $fold = Join-Path $sharedPath $dn
        
        Write-Output "STATUS:DOWNLOADING"
        if ($isC) { kaggle competitions download -c $ref -p $sharedPath --quiet 2>&1 } else { kaggle datasets download -d $ref -p $sharedPath --quiet 2>&1 }
        
        if (Test-Path $z) {
             Write-Output "RESULT:DOWNLOADED"
        } else {
             Write-Output "NOTIFICATION:Kaggle API failed/empty for $dn. Falling back to HF Manager..."
             Write-Output "STATUS:HF-PULLING"
             & $vpy $hfManager --repo_id $ref --output_dir $fold --repo_type dataset 2>&1
             if ((Get-ChildItem $fold -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0) {
                  Write-Output "NOTIFICATION:Fallback to HF Successful!"
                  Write-Output "RESULT:COMPLETED"
             } else {
                  Write-Output "RESULT:FAILED"
             }
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
            
            # If the zip is missing, it means HF or Kaggle messed up. But HF emits COMPLETED so it bypasses this!
            if (!(Test-Path $z)) {
                Write-Output "NOTIFICATION:Zip not found for extraction: $dn"
                Write-Output "RESULT:FAILED"
                return
            }
            
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
                    } else {
                        $NextDl.JobId = (Start-Job -ScriptBlock $DownloadSB -ArgumentList $NextDl.Ref, $Raw, $Vpy, $hfManagerPath).Id
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
                    elseif ($ls -match 'NOTIFICATION:(.*)') { Write-Host "  [!] $($Matches[1])" -ForegroundColor Cyan }
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
                            Write-Host "`n  [!] Job FAILED: $($ti.Slug)" -ForegroundColor Red
                        }
                    } else {
                        if (-not [string]::IsNullOrWhiteSpace($ls)) {
                            # Ensure we are working with a string (prevents ErrorRecord crashes)
                            $LineStr = [string]$ls
                            # If it looks like a tqdm bar, we use it as status
                            if ($LineStr -match '\[.*\]' -or $LineStr -match '%') {
                                $ti.Status = $LineStr.Trim()
                            } else {
                                Write-Host "    $($ti.Slug)> $LineStr" -ForegroundColor DarkGray
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
    }

    Write-Host "`n[OK] ACQUISITION MISSION ENDED." -ForegroundColor Green
    if ($null -eq $ForcedRefs) { Read-Host "Press Enter to return to menu" }
}

while ($true) {
    Clear-Host
    Write-Host '--- LEMGENDARY DATASETS HUB v5.2 ---' -ForegroundColor Yellow
    Show-Stats
    Write-Host '1. [ACQUIRE] Pull remote datasets' -ForegroundColor Gray
    Write-Host '2. [COMPILE] Build new SOTA manifold' -ForegroundColor Gray
    Write-Host '3. [REDUCE]  Create downsampled variant' -ForegroundColor Gray
    Write-Host '4. [CLEANUP] Purge redundant sources' -ForegroundColor Gray
    Write-Host 'Q. [QUIT]    Exit Dashboard' -ForegroundColor Gray
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
            }
            
            # Trigger Smart Cleanup via Python
            & $Vpy compiler-pipeline.py --cleanup
        }
        Read-Host "Press Enter to return to menu"
    }
    elseif ($I -eq '3') {
        & $Vpy compiler-pipeline.py --reduce
        Read-Host "Press Enter to return to menu"
    }
    elseif ($I -eq '4') {
        & $Vpy compiler-pipeline.py --cleanup
        Read-Host "Press Enter to return to menu"
    }
    elseif ($I -match '^q') { break }
}
