# Start a pipeline run detached from any interactive session, in a HIDDEN window.
#   powershell -File launch.ps1 DEV_FAST            (auto: resumes from the last finished step)
#   powershell -File launch.ps1 FINAL auto
param([Parameter(Mandatory = $true)][string]$Mode, [string[]]$Steps = @("auto"))
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cmd = "cmd.exe /c `"`"$here\run_mode.cmd`" $Mode $($Steps -join ' ')`""
$si = New-CimInstance -ClassName Win32_ProcessStartup -Property @{ ShowWindow = [uint16]0 } -ClientOnly
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $cmd; CurrentDirectory = $here; ProcessStartupInformation = $si }
"started MODE=$Mode steps=$($Steps -join ' ') pid=$($r.ProcessId) (log: cache\runs\$Mode.log)"
