@echo off
title Semantix - Demarrage
echo ============================================================
echo   SEMANTIX - Script de demarrage
echo ============================================================
echo.

:: Verifier si Python est installe
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    echo Installez Python depuis https://www.python.org/
    pause
    exit /b 1
)

:: Verifier si Bun est installe
where bun >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Bun n'est pas installe ou pas dans le PATH
    echo Installez Bun depuis https://bun.sh/
    pause
    exit /b 1
)

:: Verifier si les dependances Python sont installees
echo [1/3] Verification des dependances Python...
python -c "import flask; import spacy" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installation des dependances Python...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Echec de l'installation des dependances
        pause
        exit /b 1
    )
)

:: Verifier si le modele spaCy est installe
python -c "import spacy; spacy.load('fr_core_news_sm')" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Telechargement du modele spaCy francais...
    python -m spacy download fr_core_news_sm
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Echec du telechargement du modele spaCy
        pause
        exit /b 1
    )
)

:: Demarrer le service de lemmatisation en arriere-plan
echo [2/3] Demarrage du service de lemmatisation (port 3001)...
start /b "" python scripts\lemmatizer_service.py >nul 2>nul

:: Attendre que le service soit pret
timeout /t 3 /nobreak >nul

:: Verifier que le service fonctionne
curl -s http://localhost:3001/health >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARN] Le service de lemmatisation n'a pas demarre correctement
    echo        Le jeu fonctionnera sans lemmatisation
)

:: Demarrer le serveur principal
echo [3/3] Demarrage du serveur principal (port 3000)...
echo.
echo ============================================================
echo   Le jeu est accessible sur: http://localhost:3000
echo   Appuyez sur Ctrl+C pour arreter les serveurs
echo ============================================================
echo.

bun run src\server.ts

:: Nettoyage - tuer le processus Python quand Bun s'arrete
taskkill /f /im python.exe /fi "WINDOWTITLE eq *lemmatizer*" >nul 2>nul

pause
