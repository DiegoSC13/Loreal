import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob

def find_losses_files(base_dir):
    """
    Busca archivos losses.txt recursivamente en el directorio base.
    """
    files = glob.glob(os.path.join(base_dir, "**", "losses.txt"), recursive=True)
    return sorted(list(set(files)))

def get_label_from_path(path, base_dir):
    """
    Crea una etiqueta legible basada en la ruta del experimento.
    """
    rel_path = os.path.relpath(path, base_dir)
    parts = rel_path.split(os.sep)
    # Filtramos nombres de carpeta genéricos
    clean_parts = [p for p in parts if p not in ["losses.txt", "losses"]]
    return " | ".join(clean_parts)

def plot_experiments(base_dir, filter_str=None, mode='both', metric='loss', output='comparison.png'):
    files = find_losses_files(base_dir)
    
    if filter_str:
        files = [f for f in files if filter_str in f]
    
    if not files:
        print(f"No se encontraron archivos losses.txt en {base_dir} con filtro '{filter_str}'")
        return

    plt.figure(figsize=(10, 6))
    
    found_data = False
    for f in files:
        label = get_label_from_path(f, base_dir)
        
        try:
            # Format: Epoch, TrainLoss, ValLoss, [ValPSNR, ValSSIM]
            with open(f, 'r') as file:
                header = file.readline().strip().split(', ')
                data = np.genfromtxt(file, delimiter=',')
            
            if data.size == 0: continue
            if data.ndim == 1: data = data.reshape(1, -1)
            
            epochs = data[:, 0]
            
            if metric == 'loss':
                train_data = data[:, 1]
                val_data = data[:, 2]
                y_label = 'Loss'
            elif metric == 'psnr':
                if data.shape[1] < 4: 
                    print(f"  [!] Skipped {label}: No PSNR data found.")
                    continue
                val_data = data[:, 3]
                y_label = 'PSNR (dB)'
                mode = 'val' # metric is only available for validation
                # Hack for legacy data to avoid ruining Y-axis scale
                if len(val_data) > 1 and val_data[0] == 0.0:
                    epochs = epochs[1:]
                    val_data = val_data[1:]
            elif metric == 'ssim':
                if data.shape[1] < 5: 
                    print(f"  [!] Skipped {label}: No SSIM data found.")
                    continue
                val_data = data[:, 4]
                y_label = 'SSIM'
                mode = 'val'
                if len(val_data) > 1 and val_data[0] == 0.0:
                    epochs = epochs[1:]
                    val_data = val_data[1:]
            
            p = None
            marker = '.' if len(epochs) < 20 else None
            if mode in ['train', 'both'] and metric == 'loss':
                p = plt.plot(epochs, train_data, label=f"{label} (Train)", linestyle='-', alpha=0.5, marker=marker)
            
            if mode in ['val', 'both']:
                metric_label = "Val" if metric == 'loss' else metric.upper()
                # Si ya pintamos el train, usamos el mismo color para el val
                color = p[0].get_color() if p else None
                plt.plot(epochs, val_data, label=f"{label} ({metric_label})", 
                         linestyle='--' if metric=='loss' else '-', 
                         marker=marker, color=color)
            
            found_data = True
                
        except Exception as e:
            print(f"Error leyendo {f}: {e}")

    if not found_data:
        print(f"No se pudieron extraer datos para la métrica '{metric}'.")
        return

    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(f'Comparación de Experimentos - {y_label}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    
    # Auto-scaling logic
    if metric == 'loss':
        plt.yscale('log')
    
    plt.savefig(output, dpi=200)
    print(f"Gráfica guardada en: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graficador de experimentos multi-métrica")
    parser.add_argument("--base_dir", type=str, default="./results", help="Directorio raíz de resultados")
    parser.add_argument("--filter", type=str, default=None, help="Texto que debe contener la ruta para incluir el experimento")
    parser.add_argument("--mode", type=str, choices=['train', 'val', 'both'], default='both', help="Qué losses graficar (solo aplica a metric=loss)")
    parser.add_argument("--metric", type=str, choices=['loss', 'psnr', 'ssim'], default='loss', help="Métrica a graficar (loss, psnr, ssim)")
    parser.add_argument("--output", type=str, default="comparison.png", help="Nombre del archivo de salida")
    args = parser.parse_args()
    
    plot_experiments(args.base_dir, args.filter, args.mode, args.metric, args.output)
