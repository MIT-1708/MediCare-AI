@echo off
rem ============================================
rem Batch file to activate the project's virtual environment
rem and launch the Streamlit application.

rem Determine the directory where this script resides
set SCRIPT_DIR=%~dp0

echo Activating virtual environment...
call "%SCRIPT_DIR%\.venv\Scripts\activate.bat"

echo Starting the Streamlit application...
streamlit run streamlit_app.py

echo Application has stopped. Press any key to exit.
pause >nul
