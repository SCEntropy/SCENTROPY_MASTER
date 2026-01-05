@echo off
echo Setting up Hugging Face model cache directory for GPT-2 model...
echo.

REM Create the cache directory structure
set HF_CACHE_DIR=%USERPROFILE%\.cache\huggingface\transformers
set MODEL_DIR=%HF_CACHE_DIR%\models--gpt2\snapshots\1234567890abcdef1234567890abcdef12345678

echo Creating cache directory: %HF_CACHE_DIR%
if not exist "%HF_CACHE_DIR%" mkdir "%HF_CACHE_DIR%" 2>nul
if errorlevel 1 (
    echo Error: Could not create cache directory %HF_CACHE_DIR%
    echo Please check permissions or create the directory manually.
    pause
    exit /b 1
)

echo Creating model directory: %MODEL_DIR%
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%" 2>nul
if errorlevel 1 (
    echo Error: Could not create model directory %MODEL_DIR%
    echo Please check permissions or create the directory manually.
    pause
    exit /b 1
)

echo.
echo Cache directory structure created successfully!
echo.
echo Now you need to manually download the GPT-2 model files from:
echo https://huggingface.co/gpt2
echo.
echo Download these files to the directory:
echo %MODEL_DIR%
echo.
echo Files needed:
echo - config.json
echo - pytorch_model.bin
echo - tokenizer_config.json
echo - tokenizer.json
echo - vocab.json
echo - merges.txt
echo - special_tokens_map.json
echo.
echo After downloading the files, you can run the program in offline mode.
echo.
pause