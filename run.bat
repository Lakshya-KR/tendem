@echo off
REM Auto-activate venv and run CrowdWisdomTrading

cd /d "%~dp0"

if not exist "venv" (
    echo CriterionError Virtual environment not found. Run: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python main.py %*
pause
