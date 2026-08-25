@echo off
cd /d "%~dp0.."
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "CUDA_PATH_V13_3=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CUDAToolkitDir=%CUDA_PATH%"

set "OPENCV_DIR=%~dp0..\third_party\opencv\build"
if exist "%OPENCV_DIR%\OpenCVConfig.cmake" (
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DHAR_WITH_CUDA=ON -DHAR_WITH_OPENCV=ON -DOpenCV_DIR="%OPENCV_DIR%" -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=86-real
) else (
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DHAR_WITH_CUDA=ON -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=86-real
)
if errorlevel 1 exit /b 1

cmake --build build --config Release --target har_cnn
exit /b %ERRORLEVEL%
