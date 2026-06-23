import PyInstaller.__main__
import os
import shutil

def build():
    # Caminho do script principal
    main_script = "app.py"
    
    # Nome do executável final
    name = "IHC_Gesture_Viewer"
    
    # Argumentos do PyInstaller
    args = [
        main_script,
        f"--name={name}",
        "--noconfirm",             # Sobrescreve sem perguntar
        "--windowed",              # Oculta o console do Windows
        "--add-data=hand_landmarker.task;.", # Inclui o modelo do MediaPipe
        "--collect-all=customtkinter",       # Coleta todos os arquivos e dados de customtkinter
        "--collect-all=mediapipe",           # Coleta todos os arquivos e dados de mediapipe
        # Exclusões para economizar espaço e evitar erro de disco cheio
        "--exclude-module=torch",
        "--exclude-module=torchvision",
        "--exclude-module=matplotlib",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--exclude-module=PyQt5",
        "--exclude-module=PyQt6",
        "--exclude-module=IPython",
        "--exclude-module=numpy.tests",
    ]
    
    print(f"Iniciando build do PyInstaller com os argumentos: {args}")
    PyInstaller.__main__.run(args)
    print("Build finalizado com sucesso!")

if __name__ == "__main__":
    build()
