# Interpretación arquitectónica — Sensibilidad por grosor

## Resultados numéricos

| Categoría | Definición (radio en px sobre 512×512) | Sensibilidad media |
|-----------|------------------------------------------------------|--------------------|
| Fino      | radio ≤ 2                         | 0.7526 ± 0.0539 |
| Medio     | 2 < radio ≤ 4 | 0.9795 ± 0.0320 |
| Grueso    | radio > 4                        | 0.9498 ± 0.1334 |

Brecha grueso − fino: +0.1972

## Explicación arquitectónica

1. **Receptive field y downsampling.** U-Net hace 4 maxpoolings (factor 16
   total). Un capilar de radio 1-2 px en la entrada queda con radio
   subpíxel en el bottleneck → la información se diluye en BatchNorm y se
   pierde irrecuperablemente.

2. **Skip connections.** Las skip de niveles altos (poco downsampleadas)
   sí preservan detalle fino, pero el decoder debe combinarlas con
   features semánticas profundas. En los Attention Gates, la señal de
   gating del decoder es de baja resolución, por lo que su atención
   tiende a privilegiar estructuras grandes — penalizando capilares.

3. **Desbalance intra-clase.** Aunque "vaso" es una sola clase, los
   capilares aportan muchos menos píxeles que las arterias gruesas.
   La Dice loss, agregada sobre todos los píxeles-vaso, está dominada
   por los gruesos, así que el gradiente "premia" más segmentar bien una
   arteria que un capilar.

4. **Anti-aliasing en resize.** Al redimensionar a 512×512
   con interpolación bilinear, los capilares de 1-px del original quedan
   sub-resolvibles. La máscara GT se redimensiona con NEAREST (preserva
   binariedad) pero pierde continuidad.

## Mitigaciones posibles (a discutir en el informe)

- Reducir el número de downsamplings (depth=3 en lugar de 4).
- Entrenar en parches de alta resolución sin resize (e.g. 96×96 sin
  reescalar).
- Añadir una loss específica de capilares (Dice solo sobre la categoría
  "fino").
- Pretraining con tareas de detección de bordes (Sobel, Canny) para
  sesgar el modelo hacia frecuencias altas.
