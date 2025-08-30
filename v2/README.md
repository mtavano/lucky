# Lucky v2 - Compatibilidad con v1

Este paquete proporciona una interfaz compatible con Lucky v1, permitiendo usarlo como drop-in replacement.

## Migración desde v1

### Antes (Lucky v1)
```go
import "github.com/mtavano/lucky"

config := &lucky.Config{
    LabelsPath:       "labels.txt",
    TrainingDataPath: "train.txt",
    Threshold:        0.3,
    Verbose:          true,
}

config.Fit()
result := config.Predict("descripcion transaccion")
fmt.Printf("ID: %d, Name: %s, Score: %.3f\n", 
    result.ID, result.Name, result.Score)
```

### Después (Lucky v2)
```go
import "github.com/mtavano/lucky/v2"

config := &v2.Config{
    LabelsPath:       "labels.txt", 
    TrainingDataPath: "train.txt",
    Threshold:        0.3,
    Verbose:          true,
}

config.Fit()
result := config.Predict("descripcion transaccion")
fmt.Printf("ID: %d, Name: %s, Score: %.3f\n", 
    result.ID, result.Name, result.Score)
```

**¡Solo cambia el import!** La API es 100% compatible.

## Mejoras en v2

- **Algoritmo actualizado**: Char n-gramas (3-5) con TF-IDF y centroides
- **Mejor performance**: ~7.8μs por predicción (vs ~ms en v1)
- **Normalización mejorada**: Placeholders para amounts/dates/codes
- **Menor memoria**: Vectorización sparse y hashing trick
- **Sin dependencias externas**: Implementación standalone

## Funcionalidades Nuevas

### Guardar/Cargar Modelo
```go
// Guardar modelo entrenado
err := config.SaveModel("mi_modelo.json")

// Cargar modelo previamente entrenado
newConfig := &v2.Config{Threshold: 0.3}
err = newConfig.LoadModel("mi_modelo.json")
```

### Ruta de Modelo por Defecto
```go
modelPath := config.GetModelPath() // deriva de LabelsPath
```

## Performance

- **Latencia**: ~7.8μs por predicción (M1 Pro)
- **Memoria**: Modelo compacto con vectores sparse
- **Throughput**: ~127K predicciones/segundo por core

## Diferencias Internas

Aunque la API es compatible, internamente v2 usa:

| Aspecto | v1 | v2 |
|---------|----|----|
| Algoritmo | Word n-gramas + voting | Char n-gramas + centroides |
| Vectorización | Manual | TF-IDF + hashing trick |
| Clasificación | Majority voting | Cosine similarity |
| Normalización | Básica | Avanzada (placeholders) |
| Performance | ~1-3ms | ~7.8μs |

## Evaluación

Usar comandos standalone para evaluar:
```bash
# Entrenar
go run . -train=train.txt -labels=labels.txt -out=model.json

# Evaluar (80/20 split)
go run . -eval -labels=labels.txt -data=train.txt -thr=0.3

# Predicción interactiva
go run . -predict -model=model.json -thr=0.3
```

## Tests

```bash
# Tests de compatibilidad
go test -v

# Benchmark de performance
go test -bench=.
```

