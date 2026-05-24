import os

def crear_txt_donde_haya_fotos(base_path):
    # Clases de emociones correctas (FER2013 dataset)
    clases = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6
    }
    
    contador = 0
    contador_train = 0
    contador_val = 0
    
    # Procesamos las carpetas train/ y test/
    carpetas_origen = [
        ("train", os.path.join(os.path.dirname(base_path), "train"), "train"),
        ("test", os.path.join(os.path.dirname(base_path), "test"), "val")
    ]
    
    for nombre_carpeta, ruta_carpeta, tipo in carpetas_origen:
        if not os.path.exists(ruta_carpeta):
            print(f"⚠️  Carpeta no encontrada: {ruta_carpeta}")
            continue
            
        print(f"\nProcesando {nombre_carpeta}/ (como {tipo})...")
        
        # Recorremos las subcarpetas de emociones
        for emocion_folder in os.listdir(ruta_carpeta):
            emocion_path = os.path.join(ruta_carpeta, emocion_folder)
            if not os.path.isdir(emocion_path):
                continue
            
            emocion_lower = emocion_folder.lower()
            clase_id = clases.get(emocion_lower)
            
            if clase_id is None:
                print(f"  ⚠️  Emoción no reconocida: {emocion_folder}")
                continue
            
            # Creamos la carpeta de destino si no existe
            dest_label_dir = os.path.join(base_path, 'labels', tipo)
            os.makedirs(dest_label_dir, exist_ok=True)
            
            # Procesamos las imágenes de esta emoción
            for file in os.listdir(emocion_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Nombre con prefijo de emoción (si aplica)
                    nombre_nuevo = f"{emocion_folder}_{file}"
                    txt_name = os.path.splitext(nombre_nuevo)[0] + ".txt"
                    txt_path = os.path.join(dest_label_dir, txt_name)
                    
                    with open(txt_path, 'w') as f:
                        f.write(f"{clase_id} 0.5 0.5 0.8 0.8\n")
                    
                    contador += 1
                    if tipo == 'train':
                        contador_train += 1
                    else:
                        contador_val += 1
            
            print(f"  ✓ {emocion_folder}: {len([f for f in os.listdir(emocion_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])} imágenes")

    print(f"\n✅ ¡Éxito! Se han creado {contador} archivos .txt")
    print(f"   - Train: {contador_train}")
    print(f"   - Val: {contador_val}")

crear_txt_donde_haya_fotos("dataset4")