@echo off
chcp 65001 > nul
echo ===================================================
echo   RL-Miro Enhanced GUI Runner
echo ===================================================

if not exist "miro_env\Scripts\activate.bat" (
    echo [Info] Virtual environment 'miro_env' not found.
    echo [Info] Creating a new virtual environment...
    python -m venv miro_env
    if %errorlevel% neq 0 (
        echo [Error] Failed to create virtual environment. 
        echo Please ensure Python is installed and added to PATH.
        pause
        exit /b
    )
    echo [Info] Virtual environment created successfully.
)

echo [Info] Activating virtual environment (miro_env)...
call miro_env\Scripts\activate.bat

echo [Info] Checking and installing packages from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [Warning] Package installation encountered an error.
    pause
)

echo [Info] Starting RL Maze Q-Learning GUI...
python src\main.py

echo [Info] Program finished.
pause
