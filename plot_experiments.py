import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob

def find_losses_files(base_dir):
    """
    Busca archivos losses.txt recursivamente en el directorio base.
    Soporta la estructura anterior (losses.txt) y la nueva (losses/losses.txt).
    """
    patterns = [
        os.path.join(base_dir, "**/losses.txt"),
        os.path.join(base_dir, "losses/losses.txt") # por si acaso
    ]
    files = []
    for p in patterns:
        # Usamos glob recursivo para encontrar todos los niveles
        files.extend(glob.glob(os.path.join(base_dir, "**", "losses.txt"), recursive=True))
    
    # Eliminar duplicados y ordenar
    return sorted(list(set(files)))

def get_label_from_path(path, base_dir):
    """
    Intenta crear una etiqueta legible basada en la ruta del experimento.
    """
    rel_path = os.path.relpath(path, base_dir)
    parts = rel_path.split(os.sep)
    # Filtramos nombres de archivo y carpetas genéricas como 'losses'
    clean_parts = [p for p in parts if p not in ["losses.txt", "losses"]]
    return " | ".join(clean_parts)

def plot_experiments(base_dir, filter_str=None, mode='both', output='comparison.png'):
    files = find_losses_files(base_dir)
    
    if filter_str:
        files = [f for f in files if filter_str in f]
    
    if not files:
        print(f"No se encontraron archivos losses.txt en {base_dir} con filtro '{filter_str}'")
        return

    plt.figure(figsize=(12, 7))
    
    found_data = False
    for f in files:
        label = get_label_from_path(f, base_dir)
        
        try:
            # Cargamos datos asumiendo formato CSV con cabecera: Epoch, TrainLoss, ValLoss
            data = np.genfromtxt(f, delimiter=',', skip_header=1)
            if data.size == 0: continue
            if data.ndim == 1: data = data.reshape(1, -1)
            
            epochs = data[:, 0]
            train_loss = data[:, 1]
            val_loss = data[:, 2]
            
            if mode in ['train', 'both']:
                plt.plot(epochs, train_loss, label=f"{label} (Train)", linestyle='-')
            if mode in ['val', 'both']:
                plt.plot(epochs, val_loss, label=f"{label} (Val)", linestyle='--')
            
            found_data = True
                
        except Exception as e:
            print(f"Error leyendo {f}: {e}")

    if not found_data:
        print("No se pudieron extraer datos válidos de los archivos encontrados.")
        return

    # Robust scaling logic for SURE losses (can be negative)
    all_l = []
    # I need to collect all plotted data to calculate scaling
    for f in files:
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        if data.size == 0: continue
        if data.ndim == 1: data = data.reshape(1, -1)
        if mode in ['train', 'both']: all_l.append(data[:, 1])
        if mode in ['val', 'both']: all_l.append(data[:, 2])
    
    if all_l:
        all_l = np.concatenate(all_l)
        min_l, max_l = np.min(all_l), np.max(all_l)
        if min_l > 0 and (max_l / (min_l + 1e-12) > 100):
            plt.yscale('log')
        elif min_l < 0:
            if max_l - min_l > 1000:
                plt.yscale('symlog', linthresh=1.0)
            else:
                plt.yscale('linear')
        else:
            plt.yscale('linear')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Comparación de Experimentos - Modo: {mode}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output, dpi=200)
    print(f"Gráfica guardada en: {output}")
    # plt.show() # Descomentar si se corre en entorno con display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graficador de experimentos")
    parser.add_argument("--base_dir", type=str, default="./results", help="Directorio raíz de resultados")
    parser.add_argument("--filter", type=str, default=None, help="Texto que debe contener la ruta para incluir el experimento")
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'both'], default='both', help="Qué losses graficar")
    parser.add_argument("--output", type=str, default="comparison.png", help="Nombre del archivo de salida")
    args = parser.parse_args()
    
    plot_experiments(args.base_dir, args.filter, args.mode, args.output)
