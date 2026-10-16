@echo off
REM GlucoTwin dashboard - run from the project folder (YETTO_VitVellore)
cd /d "%~dp0"
python -m streamlit run app.py
pause
