import os

def generar_lista_tif(directorio, archivo_salida="lista_tif.txt"):
    lista_tif = []

    for root, dirs, files in os.walk(directorio, followlinks=True):
        # evitar entrar en directorios llamados "check"
        dirs[:] = [d for d in dirs if d != "check"]

        for file in files:
            if file.lower().endswith(".tif"):
                lista_tif.append(os.path.join(root, file))

    lista_tif.sort()
    with open(archivo_salida, "w") as f:
        for ruta in lista_tif:
            f.write(ruta + "\n")

    print(f"Se encontraron {len(lista_tif)} archivos .tif")