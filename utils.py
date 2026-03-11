import os

def generar_lista_tif(directorio, archivo_salida="lista_tif.txt"):
    lista_tif = []

    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith(".tif"):
                ruta_completa = os.path.join(root, file)
                lista_tif.append(ruta_completa)

    with open(archivo_salida, "w") as f:
        for ruta in lista_tif:
            f.write(ruta + "\n")

    print(f"Se encontraron {len(lista_tif)} archivos .tif")
    print(f"Lista guardada en: {archivo_salida}")