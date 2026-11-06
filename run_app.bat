@echo off
rem Starts the browser application (http://127.0.0.1:8000). Models load in ~10-20 s.
cd /d "%~dp0"
python app\server.py
pause
