@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Create .venv and install requirements.txt first.
 pause
 exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m streamlit run app\streamlit_app.py
pause
