@echo off
cd /d "%~dp0"

echo.
echo =============================================
echo    SPY 1DTE Vol Pipeline - Windows runner
echo =============================================
echo.

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: python not found in PATH.
    echo Install it from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo %%v detected.
echo.

echo ^> Checking / installing dependencies...
python -m pip install --upgrade -q -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: pip install failed.  Check your internet connection.
    pause
    exit /b 1
)
echo    Dependencies OK.
echo.

echo ^> Running pipeline...
echo.
python spy_1dte_vol_pipeline.py
set PIPELINE_STATUS=%ERRORLEVEL%

echo.
if %PIPELINE_STATUS% equ 0 (
    echo Pipeline finished successfully.
    echo Output files are in: %CD%\data\
) else (
    echo Pipeline exited with error code %PIPELINE_STATUS%.
)

echo.
pause
