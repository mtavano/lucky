package v2

import (
	"log"
	"math"
	"os"
	"path/filepath"
)

// BestCategory mantiene compatibilidad con model.BestCategory de v1
type BestCategory struct {
	ID    uint    `json:"id"`
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

// ModelSample mantiene compatibilidad con model.Sample de v1
type ModelSample struct {
	Ngram    string
	Freq     float64
	Classes  map[uint]float64
	Prob     float64
	Tfidf    map[uint]float64
	Maximum  float64
	Minimum  float64
	Weighted bool
}

// Config mantiene compatibilidad con lucky.Config de v1
type Config struct {
	// v1 compatible fields
	Model            map[string]*ModelSample // no usado en v2, pero mantenemos compatibilidad
	CatStr           map[uint]string         // mapeado desde Model.LabelName
	LabelsPath       string
	TrainingDataPath string
	Verbose          bool
	AsPkg            bool
	InvalidWords     []string // no usado en v2 (usa char n-gramas)
	Threshold        float64

	// v2 internal state
	model *Model // modelo entrenado de v2
}

// Fit entrena el modelo usando el pipeline v2
func (config *Config) Fit() {
	if config.Verbose {
		log.Println(">> Init model v2.")
	}

	// validar archivos de entrada
	if config.LabelsPath == "" {
		log.Fatal("LabelsPath required")
	}
	if config.TrainingDataPath == "" {
		log.Fatal("TrainingDataPath required")
	}

	if config.Verbose {
		log.Printf("> Loading %s\n", config.LabelsPath)
		log.Printf("> Loading %s\n", config.TrainingDataPath)
	}

	// cargar labels para CatStr compatibility
	config.CatStr = readLabels(config.LabelsPath)

	if config.Verbose {
		log.Println(">> Fit model v2.")
	}

	// usar funciones v2 para entrenar
	labelName := readLabels(config.LabelsPath)
	samples := readData(config.TrainingDataPath)

	// compute DF for IDF using hybrid features
	df := make([]int, Dim)
	docs := len(samples)
	for _, s := range samples {
		text := normalize(s.Text)
		seen := map[int]bool{}
		for idx := range hybridFeaturize(text, nil, 0, 3, 5) {
			if !seen[idx] {
				df[idx] = df[idx] + 1
				seen[idx] = true
			}
		}
	}

	// accumulate centroids using hybrid features
	acc := make(map[uint]map[int]float64)
	count := make(map[uint]int)
	for _, s := range samples {
		text := normalize(s.Text)
		vec := hybridFeaturize(text, df, docs, 3, 5)
		if acc[s.Label] == nil {
			acc[s.Label] = map[int]float64{}
		}
		for i, v := range vec {
			acc[s.Label][i] += v
		}
		count[s.Label]++
	}

	// average & normalize centroids
	centroids := make(map[uint][]float64, len(acc))
	for lab, m := range acc {
		vec := make([]float64, Dim)
		c := float64(count[lab])
		if c == 0 {
			c = 1
		}
		for i, v := range m {
			vec[i] = v / c
		}
		// L2-normalize centroid
		l2 := 0.0
		for _, v := range vec {
			l2 += v * v
		}
		l2 = math.Sqrt(l2)
		if l2 > 0 {
			for i := range vec {
				vec[i] /= l2
			}
		}
		centroids[lab] = vec
	}

	// build feature map for hierarchical voting
	featureMap := BuildFeatureMap(samples)

	// crear modelo interno v2
	config.model = &Model{
		Centroids:  centroids,
		LabelName:  labelName,
		DF:         df,
		Docs:       docs,
		FeatureMap: featureMap,
		Params: map[string]any{
			"ngrams":     "hybrid(char:3-5,word:1-3)+voting(word:1-3)",
			"dim":        Dim,
			"classifier": "centroid-cosine+hierarchical-voting",
			"version":    "2.3",
		},
		Weights: &DefaultWeights,
	}

	if config.Verbose {
		log.Println(">> Model v2 ready.")
	}
}

// Predict predice la categoría de un texto usando el modelo v2
func (config *Config) Predict(test string) *BestCategory {
	if config.model == nil {
		log.Fatal("Model not trained. Call Fit() first.")
	}

	// usar predictOneHierarchical de v2.3 con topk=1
	preds := predictOneHierarchical(config.model, test, 1)
	if len(preds) == 0 {
		return &BestCategory{
			ID:    0,
			Name:  "UNKNOWN",
			Score: 0.0,
		}
	}

	best := preds[0]

	// aplicar threshold si está configurado
	if config.Threshold > 0 && best.Score < config.Threshold {
		return &BestCategory{
			ID:    0,
			Name:  "UNKNOWN",
			Score: best.Score,
		}
	}

	return &BestCategory{
		ID:    best.ID,
		Name:  best.Name,
		Score: best.Score,
	}
}

// SaveModel guarda el modelo entrenado
func (config *Config) SaveModel(path string) error {
	if config.model == nil {
		log.Fatal("Model not trained. Call Fit() first.")
	}
	saveModel(path, config.model)
	return nil
}

// LoadModel carga un modelo previamente entrenado
func (config *Config) LoadModel(path string) error {
	// verificar que el archivo existe
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return err
	}

	config.model = loadModel(path)

	// reconstruir CatStr para compatibilidad
	config.CatStr = config.model.LabelName

	return nil
}

// GetModelPath retorna la ruta del modelo por defecto
func (config *Config) GetModelPath() string {
	if config.LabelsPath != "" {
		dir := filepath.Dir(config.LabelsPath)
		return filepath.Join(dir, "model.json")
	}
	return "model.json"
}
