import os
import uproot
import numpy as np
import awkward as ak

def sample_root_files(input_dir, output_dir, sample_fraction=0.1):
    """
    Samplea un porcentaje de eventos de todos los archivos .root en un directorio.
    
    Args:
        input_dir (str): Directorio con los archivos .root originales.
        output_dir (str): Directorio donde se guardarán los archivos sampleados.
        sample_fraction (float): Fracción de eventos a samplear (por defecto 1%).
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Listar todos los archivos .root en el directorio de entrada
    root_files = [f for f in os.listdir(input_dir) if f.endswith('.root')]

    for root_file in root_files:
        input_path = os.path.join(input_dir, root_file)
        output_path = os.path.join(output_dir, root_file)

        # Abrir el archivo .root
        with uproot.open(input_path) as file:
            # Obtener el árbol (TTree) del archivo
            tree = file["tree"]  # Cambia "Events" si el árbol tiene otro nombre

            # Obtener el número total de eventos
            num_events = tree.num_entries

            # Calcular el número de eventos a samplear
            num_samples = int(num_events * sample_fraction)

            # Seleccionar índices aleatorios
            indices = np.random.choice(num_events, num_samples, replace=False)

            # Leer todos los eventos y luego seleccionar los índices específicos
            all_data = tree.arrays()
            sampled_data = all_data[indices]

            # Guardar los eventos sampleados en un nuevo archivo .root
            with uproot.recreate(output_path) as new_file:
                # Crear un diccionario para los datos compatibles
                compatible_data = {}

                # Procesar cada campo en los datos sampleados
                for field in sampled_data.fields:
                    data = sampled_data[field]

                    # Guardar el campo directamente (uproot manejará los arrays jagged)
                    compatible_data[field] = data

                # Escribir los datos en el nuevo archivo .root
                new_file["tree"] = compatible_data  # Cambia "Events" si el árbol tiene otro nombre

        print(f"Sampleado {num_samples} eventos de {num_events} en {root_file}")

if __name__ == "__main__":
   # Directorios de entrada y salida
    #/home/catalinariano/Documentos/particle_transformer-main/JetClass
    input_dir = "/home/catalinariano/Documentos/particle_transformer-main/JetClass/Pythia/val_5M"  # Cambia esto por la ruta correcta
    output_dir = "/home/catalinariano/Documentos/particle_transformer-main/JetClass/Pythia/val_100k"  # Cambia esto por la ruta deseada

    # Fracción de sampleo (10%)
    sample_fraction = 0.02

    # Ejecutar el sampleo
    sample_root_files(input_dir, output_dir, sample_fraction)

