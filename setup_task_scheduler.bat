@echo off
:: ============================================================
::  Systematic Alpha — Agendador de Tarefas Windows
::  EXECUTE COMO ADMINISTRADOR
:: ============================================================

set PROJ=%~dp0
set PYTHON=%PROJ%venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

echo.
echo  =====================================================
echo   Systematic Alpha — Configurando Task Scheduler
echo  =====================================================
echo.

:: Remove tarefas antigas
schtasks /delete /tn "SysAlpha_Acordar"   /f >nul 2>&1
schtasks /delete /tn "SysAlpha_Scheduler" /f >nul 2>&1
schtasks /delete /tn "SysAlpha_Dormir"    /f >nul 2>&1

:: TAREFA 1: Acorda o PC as 09:20 (dias uteis)
schtasks /create ^
  /tn "SysAlpha_Acordar" ^
  /tr "cmd /c echo acordar" ^
  /sc WEEKLY ^
  /d MON,TUE,WED,THU,FRI ^
  /st 09:20 ^
  /rl HIGHEST ^
  /f
:: Habilita wake do sleep para esta tarefa
powershell -Command "& { $t = Get-ScheduledTask 'SysAlpha_Acordar'; $t.Settings.WakeToRun = $true; Set-ScheduledTask $t }" >nul 2>&1

:: TAREFA 2: Inicia o scheduler as 09:25
schtasks /create ^
  /tn "SysAlpha_Scheduler" ^
  /tr "\"%PYTHON%\" \"%PROJ%scheduler.py\"" ^
  /sc WEEKLY ^
  /d MON,TUE,WED,THU,FRI ^
  /st 09:25 ^
  /rl HIGHEST ^
  /f

:: TAREFA 3: Encerra o scheduler as 18:00
schtasks /create ^
  /tn "SysAlpha_Dormir" ^
  /tr "taskkill /F /IM python.exe /FI \"WINDOWTITLE eq*scheduler*\"" ^
  /sc WEEKLY ^
  /d MON,TUE,WED,THU,FRI ^
  /st 18:00 ^
  /rl HIGHEST ^
  /f

:: TAREFA 4: Retreinamento sabado 07:00
schtasks /delete /tn "SysAlpha_Retreino" /f >nul 2>&1
schtasks /create ^
  /tn "SysAlpha_Retreino" ^
  /tr "\"%PYTHON%\" \"%PROJ%scheduler.py\" --agora" ^
  /sc WEEKLY ^
  /d SAT ^
  /st 07:00 ^
  /rl HIGHEST ^
  /f
powershell -Command "& { $t = Get-ScheduledTask 'SysAlpha_Retreino'; $t.Settings.WakeToRun = $true; Set-ScheduledTask $t }" >nul 2>&1

:: Habilita wake-from-sleep no Windows
powercfg /change standby-timeout-ac 0   >nul 2>&1
echo   Wake-from-sleep: habilitado

if %errorlevel%==0 (
  echo.
  echo   OK - Tarefas registradas:
  echo     09:20 seg-sex  Acorda o PC do sleep
  echo     09:25 seg-sex  Inicia o scheduler
  echo     18:00 seg-sex  Encerra o scheduler
  echo     07:00 sabado   Retreinamento semanal
  echo.
  echo   O PC pode ficar em suspensao a noite.
  echo   O Windows acorda automaticamente as 09:20.
) else (
  echo   ERRO - Execute como Administrador.
)

pause
