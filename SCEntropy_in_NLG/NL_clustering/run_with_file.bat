@echo off
echo Running Fuzashang with question from file...
echo.

set /p questionNum="Enter question number (1-6) to run (or press Enter for default 1): "
if "%questionNum%"=="" set questionNum=1

if %questionNum% GTR 6 (
    echo Invalid question number. Please enter a number between 1 and 6.
    pause
    exit /b 1
)

set questionFile=QUESTION%questionNum%.txt

if not exist "%questionFile%" (
    echo Question file %questionFile% does not exist.
    pause
    exit /b 1
)

echo Reading question from: %questionFile%
type %questionFile%
echo.
echo.

cd /d "c:\Users\someo\Desktop\all\fuzashang"
python fuzashang.py --question-file %questionFile% --zhipu-api-key 97994c67e23992560a6fd138fa7fb72b.9WY83JWifieCtMqA --deepseek-api-key sk-c129887e59be45e8be3d5a8761fc0392

pause