@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PROJECT_ROOT=%~dp0"
set "DELIVERY=%PROJECT_ROOT%phase1_eda_delivery"
set "VENV=%DELIVERY%\.venv"
cd /d "%PROJECT_ROOT%"
if not exist "%DELIVERY%\run_eda_all.py" (
  echo ERROR: missing phase1_eda_delivery\run_eda_all.py
  pause
  exit /b 1
)
if not exist "%VENV%\Scripts\python.exe" (
  echo Creating an isolated Python virtual environment...
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3.13 --version >nul 2>&1
    if not errorlevel 1 (
      py -3.13 -m venv "%VENV%"
    ) else (
      py -3 -m venv "%VENV%"
    )
  ) else (
    python -m venv "%VENV%"
  )
  if errorlevel 1 (
    echo ERROR: Could not create Python virtual environment. Install Python 3.13.
    pause
    exit /b 1
  )
)
rem A trailing backslash inside quoted --project-root would escape the closing quote in Python argv.
rem Append a dot so Windows resolves the same folder without an escaped closing quote.
"%VENV%\Scripts\python.exe" -u "%DELIVERY%\run_eda_all.py" --project-root "%PROJECT_ROOT%." %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
  echo Runner finished successfully. See reports\phase1\eda_send_to_chat\
) else (
  echo Runner stopped or validation failed. Review the displayed error and any generated ZIP.
)
echo Press any key to close.
pause >nul
exit /b %EXIT_CODE%
