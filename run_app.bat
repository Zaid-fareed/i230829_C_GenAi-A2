@echo off
rem Starts the browser application at http://127.0.0.1:8000 (the browser opens by itself).
rem Picks the first Python on this machine that has the required packages (torch, transformers, fastapi).
cd /d "%~dp0"
set "CHECK=import torch, transformers, fastapi, uvicorn"
set "PY="

for %%P in ("%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python311\python.exe") do (
    if not defined PY if exist %%P ( %%P -c "%CHECK%" >nul 2>&1 && set "PY=%%~P" )
)
if not defined PY ( py -3.13 -c "%CHECK%" >nul 2>&1 && set "PY=py -3.13" )
if not defined PY ( py -3 -c "%CHECK%" >nul 2>&1 && set "PY=py -3" )
if not defined PY ( python -c "%CHECK%" >nul 2>&1 && set "PY=python" )

if not defined PY (
    echo No Python with the required packages was found.
    echo Install them with:  python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Using Python: %PY%
echo Loading models - the browser opens automatically; wait until the badge says "model ready".
echo Close this window to stop the app.
if "%PY:~0,2%"=="py" (%PY% app\server.py) else ("%PY%" app\server.py)
pause
