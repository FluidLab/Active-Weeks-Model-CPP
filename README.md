# Weeks-Model-CPP
 
Running the program (Windows 11):
Must have MSYS2 installed: https://www.msys2.org/
 
After install, I use a .bat script with the desired .cpp file inside
"g++ -std=c++11 -O2 -fopenmp -I "C:/Program Files/c++ libraries/eigen-3.4.0" -fdiagnostics-color=always -g desired_fname.cpp -o desired_fname.exe"
 
Say the .bat script is called example.bat. Running this in terminal requires going to the directory and running ".\example.bat"
 
After execution, run the command ".\desired_fname.exe" and the code will run