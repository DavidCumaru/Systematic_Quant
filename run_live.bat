@echo off
:: ============================================================
::  Systematic Alpha — Launcher
::  Duplo-clique para iniciar o scheduler diario.
::  Deixe esta janela aberta. Feche para parar.
:: ============================================================

title Systematic Alpha — Scheduler B3
cd /d "%~dp0"

:: Usa o Python do venv se existir, caso contrario usa o do sistema
if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON="%~dp0venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

echo.
echo  =====================================================
echo   Systematic Alpha — Scheduler B3
echo   Iniciando...
echo  =====================================================
echo.

%PYTHON% scheduler.py

echo.
echo  Scheduler encerrado. Pressione qualquer tecla para fechar.
pause >nul
