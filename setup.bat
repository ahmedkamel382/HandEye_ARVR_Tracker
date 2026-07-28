@echo off
setlocal
echo ==========================================
echo        Project Environment Setup
echo ==========================================
echo.
:: --------------------------------------------------
:: Check if Python 3.11 is already installed
:: --------------------------------------------------

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    goto INSTALL_PY311
)
echo Python 3.11 detected.
goto CREATE_VENV


:INSTALL_PY311
echo Python 3.11 was not found.
echo.
echo Installing Python 3.11.9...
echo (Your existing Python installations will NOT be modified.)
echo.

winget install ^
    -e ^
    --id Python.Python.3.11 ^
    --version 3.11.9 ^
    --override "InstallAllUsers=0 PrependPath=0 Include_launcher=1" ^
    --accept-source-agreements ^
    --accept-package-agreements
echo.
echo ==========================================
echo Python 3.11 has been installed.
echo.
echo Please CLOSE this terminal,
echo open a NEW terminal,
echo then run setup.bat again.
echo ==========================================
pause
exit /b


:CREATE_VENV
if not exist ".venv" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv

    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Existing virtual environment found.
)
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Upgrading pip...

python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)
echo.

:: --------------------------------------------------
:: NEW: Install Pybind11 directly via pip
:: --------------------------------------------------
echo Installing Pybind11 (C++/Python Bridge)...
pip install pybind11
if errorlevel 1 (
    echo Failed to install Pybind11.
    pause
    exit /b 1
)
echo.

echo Installing project dependencies...

pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)
echo.
echo ==========================================
echo Setup completed successfully.
echo ==========================================
pause