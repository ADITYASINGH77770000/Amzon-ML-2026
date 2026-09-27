@echo off
rem Resumable complete-data pipeline: every finished step leaves a marker; re-running
rem continues from the last finished step (and the last saved chunk inside it).
cd /d "%~dp0src"
set PYTHONIOENCODING=utf-8
set ER_JOBS=4
set ER_ROOT=
"C:\Python313\python.exe" run_pipeline.py auto >> "%~dp0..\..\cache\run_auto.txt" 2>&1
echo exit code %ERRORLEVEL% >> "%~dp0..\..\cache\run_auto.txt"
