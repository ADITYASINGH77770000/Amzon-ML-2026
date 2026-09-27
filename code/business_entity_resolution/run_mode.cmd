@echo off
rem usage: run_mode.cmd <DEV_FAST|DEV_MEDIUM|VALIDATION|FINAL> [steps...]   (default step: auto)
rem Every mode runs the same code; the mode only selects the data subset and cache folder.
rem Launch it detached with a HIDDEN window (closing a console window stops the run).
set HERE=%~dp0
cd /d "%HERE%src"
set PYTHONIOENCODING=utf-8
set ER_JOBS=4
set ER_ROOT=
set ER_MODE=%1
shift
set STEPS=%1 %2 %3 %4 %5 %6 %7 %8 %9
if "%1"=="" set STEPS=auto
if not exist "%HERE%..\..\cache\runs" mkdir "%HERE%..\..\cache\runs"
"C:\Python313\python.exe" run_pipeline.py %STEPS% >> "%HERE%..\..\cache\runs\%ER_MODE%.log" 2>&1
echo exit code %ERRORLEVEL% >> "%HERE%..\..\cache\runs\%ER_MODE%.log"
