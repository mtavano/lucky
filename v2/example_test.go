package v2

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCompatibility verifica que v2 puede usarse como drop-in replacement de v1
func TestCompatibility(t *testing.T) {
	// Crear archivos de prueba temporales
	tmpDir := t.TempDir()
	labelsFile := filepath.Join(tmpDir, "labels.txt")
	trainFile := filepath.Join(tmpDir, "train.txt")

	// Crear datos de prueba
	labelsData := `1#comida
2#transporte
3#entretenimiento`

	trainData := `1#compra en supermercado verduras
1#almuerzo restaurante pizza
2#uber viaje al centro
2#metro tarjeta bip
3#netflix suscripcion mensual
3#entrada cine pelicula`

	err := os.WriteFile(labelsFile, []byte(labelsData), 0644)
	if err != nil {
		t.Fatalf("Error creando archivo labels: %v", err)
	}

	err = os.WriteFile(trainFile, []byte(trainData), 0644)
	if err != nil {
		t.Fatalf("Error creando archivo train: %v", err)
	}

	// Probar interfaz compatible v1
	config := &Config{
		LabelsPath:       labelsFile,
		TrainingDataPath: trainFile,
		Verbose:          false,
		AsPkg:            true,
		Threshold:        0.1,
	}

	// Test Fit (entrena el modelo)
	config.Fit()

	// Verificar que CatStr se cargó correctamente
	if len(config.CatStr) == 0 {
		t.Error("CatStr no se cargó correctamente")
	}

	expectedLabels := map[uint]string{
		1: "comida",
		2: "transporte",
		3: "entretenimiento",
	}

	for id, name := range expectedLabels {
		if config.CatStr[id] != name {
			t.Errorf("Label %d: esperado '%s', obtenido '%s'", id, name, config.CatStr[id])
		}
	}

	// Test Predict (compatible con v1)
	testCases := []struct {
		text     string
		expected uint // categoría esperada
	}{
		{"compra en verduleria", 1}, // comida
		{"viaje en metro", 2},       // transporte
		{"pelicula en cine", 3},     // entretenimiento
	}

	for _, tc := range testCases {
		result := config.Predict(tc.text)

		// Verificar que result tiene la estructura correcta (compatible con v1)
		if result == nil {
			t.Errorf("Predict retornó nil para texto: %s", tc.text)
			continue
		}

		// Verificar campos de BestCategory
		if result.ID == 0 {
			t.Logf("UNKNOWN prediction para '%s' (score: %.3f)", tc.text, result.Score)
		} else {
			t.Logf("Prediction para '%s': ID=%d, Name='%s', Score=%.3f",
				tc.text, result.ID, result.Name, result.Score)
		}

		// Verificar que Name corresponde al ID
		if result.ID > 0 && config.CatStr[result.ID] != result.Name {
			t.Errorf("Inconsistencia: ID=%d debería tener Name='%s', obtenido '%s'",
				result.ID, config.CatStr[result.ID], result.Name)
		}
	}

	// Test SaveModel/LoadModel
	modelFile := filepath.Join(tmpDir, "test_model.json")
	err = config.SaveModel(modelFile)
	if err != nil {
		t.Errorf("Error guardando modelo: %v", err)
	}

	// Crear nueva config y cargar modelo
	config2 := &Config{
		Threshold: 0.1,
	}

	err = config2.LoadModel(modelFile)
	if err != nil {
		t.Errorf("Error cargando modelo: %v", err)
	}

	// Verificar que funciona después de cargar
	result := config2.Predict("compra en supermercado")
	if result == nil {
		t.Error("Predict falló después de LoadModel")
	} else {
		t.Logf("Después de LoadModel - Prediction: ID=%d, Name='%s', Score=%.3f",
			result.ID, result.Name, result.Score)
	}
}

// BenchmarkPredict verifica la performance de predicción
func BenchmarkPredict(b *testing.B) {
	// Setup con datos mínimos
	tmpDir := b.TempDir()
	labelsFile := filepath.Join(tmpDir, "labels.txt")
	trainFile := filepath.Join(tmpDir, "train.txt")

	labelsData := `1#comida
2#transporte`

	trainData := `1#compra supermercado
2#viaje metro`

	os.WriteFile(labelsFile, []byte(labelsData), 0644)
	os.WriteFile(trainFile, []byte(trainData), 0644)

	config := &Config{
		LabelsPath:       labelsFile,
		TrainingDataPath: trainFile,
		AsPkg:            true,
		Threshold:        0.1,
	}

	config.Fit()

	b.ResetTimer()

	// Benchmark prediction
	for i := 0; i < b.N; i++ {
		result := config.Predict("compra en tienda")
		_ = result // evitar optimización del compilador
	}
}

// TestModelPersistence verifica que SaveModel y LoadModel funcionan correctamente
func TestModelPersistence(t *testing.T) {
	// Crear archivos de prueba temporales
	tmpDir := t.TempDir()
	labelsFile := filepath.Join(tmpDir, "labels.txt")
	trainFile := filepath.Join(tmpDir, "train.txt")
	modelFile := filepath.Join(tmpDir, "model.json")

	// Crear datos de prueba
	labelsData := `1#comida
2#transporte`

	trainData := `1#compra supermercado verduras
2#viaje metro centro`

	err := os.WriteFile(labelsFile, []byte(labelsData), 0644)
	if err != nil {
		t.Fatalf("Error creando archivo labels: %v", err)
	}

	err = os.WriteFile(trainFile, []byte(trainData), 0644)
	if err != nil {
		t.Fatalf("Error creando archivo train: %v", err)
	}

	// Entrenar y guardar modelo
	config := &Config{
		LabelsPath:       labelsFile,
		TrainingDataPath: trainFile,
		Threshold:        0.1,
		AsPkg:            true,
	}

	config.Fit()

	// Test SaveModel
	err = config.SaveModel(modelFile)
	if err != nil {
		t.Errorf("Error guardando modelo: %v", err)
	}

	// Verificar que el archivo se creó
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		t.Error("Archivo de modelo no fue creado")
	}

	// Test LoadModel con nueva instancia
	newConfig := &Config{Threshold: 0.1}
	err = newConfig.LoadModel(modelFile)
	if err != nil {
		t.Errorf("Error cargando modelo: %v", err)
	}

	// Verificar que CatStr se cargó
	if len(newConfig.CatStr) == 0 {
		t.Error("CatStr no se cargó después de LoadModel")
	}

	// Verificar que las predicciones funcionan
	result := newConfig.Predict("compra en tienda")
	if result == nil {
		t.Error("Predict retornó nil después de LoadModel")
	} else {
		t.Logf("Después de LoadModel: '%s' → %s (score: %.3f)",
			"compra en tienda", result.Name, result.Score)
	}

	// Comparar predicciones entre modelo original y cargado
	originalResult := config.Predict("viaje en bus")
	loadedResult := newConfig.Predict("viaje en bus")

	if originalResult.ID != loadedResult.ID {
		t.Errorf("Predicciones diferentes: original ID=%d, loaded ID=%d",
			originalResult.ID, loadedResult.ID)
	}

	// Verificar que scores son similares (pueden tener pequeñas diferencias de precisión)
	scoreDiff := abs(originalResult.Score - loadedResult.Score)
	if scoreDiff > 0.001 {
		t.Errorf("Scores muy diferentes: original=%.6f, loaded=%.6f, diff=%.6f",
			originalResult.Score, loadedResult.Score, scoreDiff)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
