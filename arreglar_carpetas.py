import os
import shutil

def aplanar_carpetas(ruta_base):
    for tipo in ['train', 'val']:
        ruta_fotos = os.path.join(ruta_base, 'images', tipo)
        if not os.path.exists(ruta_fotos): continue
        
        # Listamos las subcarpetas de emociones
        subcarpetas = [d for d in os.listdir(ruta_fotos) if os.path.isdir(os.path.join(ruta_fotos, d))]
        
        for carpeta in subcarpetas:
            ruta_emocion = os.path.join(ruta_fotos, carpeta)
            fotos = os.listdir(ruta_emocion)
            
            for foto in fotos:
                # Crear nombre único con prefijo de emoción para evitar conflictos
                nombre_base, extension = os.path.splitext(foto)
                nombre_nuevo = f"{carpeta}_{foto}"
                
                # Movemos cada foto a la carpeta padre (train o val) con nombre único
                ruta_origen = os.path.join(ruta_emocion, foto)
                ruta_destino = os.path.join(ruta_fotos, nombre_nuevo)
                shutil.move(ruta_origen, ruta_destino)
            
            # Borramos la subcarpeta ahora que está vacía
            os.rmdir(ruta_emocion)
            print(f"✅ Fotos de {carpeta} movidas a {tipo} ({len(fotos)} archivos)")

print("🚀 Iniciando reorganización de carpetas...")
aplanar_carpetas("dataset4")
print("\n✅ ¡Reorganización completada!")