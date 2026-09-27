# Waits until the running pipeline reaches the (resumable) feature step, stops it, and
# relaunches `auto` with the current code. Launched detached so it survives the session.
$log = "D:\Amzon ML\Dataset\student_resource\cache\run_full2.txt"
$note = "D:\Amzon ML\Dataset\student_resource\cache\swap_note.txt"
"watcher started $(Get-Date)" | Out-File $note -Append
while (-not (Select-String -Path $log -Pattern "==== step feats_train" -Quiet)) {
    if (Select-String -Path $log -Pattern "exit code|Traceback" -Quiet) { "pipeline ended before feats_train $(Get-Date)" | Out-File $note -Append; break }
    Start-Sleep 30
}
Start-Sleep 60
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_pipeline.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -like '*run_full.cmd*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 5
"stopped old pipeline $(Get-Date); relaunching auto" | Out-File $note -Append
$cmd = 'cmd.exe /c ""D:\Amzon ML\Dataset\student_resource\code\business_entity_resolution\run_auto.cmd""'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = "D:\Amzon ML\Dataset\student_resource\code\business_entity_resolution" }
"relaunched pid $($r.ProcessId) rv $($r.ReturnValue) $(Get-Date)" | Out-File $note -Append
