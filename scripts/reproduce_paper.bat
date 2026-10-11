@echo off
rem Reproduce the paper results into outputs\ and compare them with paper_assets\.
rem
rem Usage (any directory):  scripts\reproduce_paper.bat
rem Requires the pinned dependencies:  pip install -r requirements.txt
setlocal
cd /d "%~dp0.."

python run_experiments.py --runs 20 --resolution 8 --seed 20240603
if errorlevel 1 exit /b %errorlevel%

python scripts\compare_to_paper.py
exit /b %errorlevel%
