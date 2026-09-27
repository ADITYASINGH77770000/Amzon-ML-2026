@echo off
rem Complete-data pipeline, resumed after the collect pass was interrupted.
rem Launched as a standalone process so it does not depend on any interactive session.
cd /d "%~dp0src"
set PYTHONIOENCODING=utf-8
set ER_JOBS=4
set ER_ROOT=
"C:\Python313\python.exe" run_pipeline.py %* > "%~dp0..\..\cache\run_full2.txt" 2>&1
echo exit code %ERRORLEVEL% >> "%~dp0..\..\cache\run_full2.txt"
