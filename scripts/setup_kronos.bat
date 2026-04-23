@echo off
REM Installer for Kronos (github.com/shiyu-coder/Kronos) on Windows.
REM Clones the repo into .kronos and installs CPU torch + Kronos requirements.

setlocal
set ROOT_DIR=%~dp0..
set KRONOS_DIR=%ROOT_DIR%\.kronos

if not exist "%KRONOS_DIR%" (
    echo Cloning Kronos into %KRONOS_DIR% ...
    git clone --depth 1 https://github.com/shiyu-coder/Kronos "%KRONOS_DIR%"
    if errorlevel 1 (
        echo [setup_kronos] git clone failed. Install git or check network.
        exit /b 1
    )
) else (
    echo Kronos already cloned at %KRONOS_DIR%
)

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if exist "%KRONOS_DIR%\requirements.txt" (
    python -m pip install -r "%KRONOS_DIR%\requirements.txt"
)

echo.
echo Kronos installed under %KRONOS_DIR%
echo Run: python scripts\kronos_infer.py --asset BTC --input ^<csv^>
endlocal
