@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title SLAD 배치 방식 비교 시뮬레이터

rem ---- 파이썬 찾기: python -> py -3 -> %USERPROFILE%\myenv ----
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY ( if exist "%USERPROFILE%\myenv\Scripts\python.exe" set "PY=%USERPROFILE%\myenv\Scripts\python.exe" )
if not defined PY (
  echo [오류] 파이썬을 찾지 못했습니다. https://www.python.org 에서 설치 후 다시 실행하세요.
  pause
  exit /b 1
)
echo [1/2] 필요한 패키지 확인/설치 중... (처음 한 번만 몇 분 걸릴 수 있습니다)
%PY% -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo [경고] 패키지 설치 중 문제가 있었습니다. 이미 설치된 패키지로 실행을 시도합니다.
)
echo [2/2] 앱을 시작합니다. 브라우저가 자동으로 열립니다. 안 열리면 http://localhost:8501 을 여세요.
echo       끝내려면 이 창을 닫으세요.
%PY% -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
echo.
echo 앱이 종료되었습니다. 위에 오류가 있으면 그 내용을 알려주세요.
pause
