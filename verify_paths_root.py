import os
import sys

def load_local_paths():
    config_path = os.path.join(os.path.dirname(__file__), 'env_paths.sh')
    print(f"Checking config at: {config_path}")
    if os.path.exists(config_path):
        env_vars = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    # Expand variables like ${WORKDIR} or $WORKDIR
                    for k, v in env_vars.items():
                        value = value.replace(f'${{{k}}}', v)
                        value = value.replace(f'${k}', v)
                    env_vars[key] = value
                    
                    if key == 'EXTERNAL_CODES_DIR':
                        print(f"Adding to sys.path: {value}")
                        if value not in sys.path:
                            sys.path.append(value)
    else:
        print("Config file not found")

load_local_paths()

# Reintentar el import que fallaba
try:
    from models_FastDVDnet_sans_noise_map import FastDVDnet
    print("SUCCESS: Module imported correctly!")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"Generic Failure: {e}")
