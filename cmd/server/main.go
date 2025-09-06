package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	v2 "github.com/mtavano/lucky/v2"
)

type Server struct {
	classifier *v2.Config
	port       string
	mu         sync.RWMutex
}

type PredictionRequest struct {
	Text string `json:"text"`
}

type PredictionResponse struct {
	ID       uint    `json:"id"`
	Name     string  `json:"name"`
	Score    float64 `json:"score"`
	Duration string  `json:"duration,omitempty"`
}

type HealthResponse struct {
	Status    string `json:"status"`
	Timestamp string `json:"timestamp"`
	Version   string `json:"version"`
	Uptime    string `json:"uptime"`
}

type BatchRequest struct {
	Texts []string `json:"texts"`
}

type BatchResponse struct {
	Results  []PredictionResponse `json:"results"`
	Count    int                  `json:"count"`
	Duration string               `json:"duration"`
}

var startTime = time.Now()

func main() {
	var (
		modelPath = flag.String("model", getEnv("MODEL_PATH", "model.json"), "Path to model file")
		port      = flag.String("port", getEnv("PORT", "8080"), "Server port")
		threshold = flag.Float64("threshold", getEnvFloat("THRESHOLD", 0.3), "Prediction threshold")
		health    = flag.Bool("health", false, "Health check mode")
		verbose   = flag.Bool("verbose", getEnvBool("VERBOSE", false), "Enable verbose logging")
	)
	flag.Parse()

	// Health check mode for Docker/K8s probes
	if *health {
		resp, err := http.Get(fmt.Sprintf("http://localhost:%s/health", *port))
		if err != nil || resp.StatusCode != 200 {
			os.Exit(1)
		}
		os.Exit(0)
	}

	// Load model
	classifier := &v2.Config{
		Threshold: *threshold,
		Verbose:   *verbose,
	}

	log.Printf("Loading model from: %s", *modelPath)
	if err := classifier.LoadModel(*modelPath); err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	log.Printf("Model loaded successfully with %d categories", len(classifier.CatStr))
	if *verbose {
		for id, name := range classifier.CatStr {
			log.Printf("  Category %d: %s", id, name)
		}
	}

	server := &Server{
		classifier: classifier,
		port:       *port,
	}

	server.setupRoutes()

	log.Printf("Lucky v2 Classifier server starting on port %s", *port)
	log.Printf("Model threshold: %.3f", *threshold)
	log.Fatal(http.ListenAndServe(":"+*port, nil))
}

func (s *Server) setupRoutes() {
	http.HandleFunc("/health", s.healthHandler)
	http.HandleFunc("/predict", s.predictHandler)
	http.HandleFunc("/batch", s.batchHandler)
	http.HandleFunc("/categories", s.categoriesHandler)
	http.HandleFunc("/", s.indexHandler)
}

func (s *Server) indexHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "text/html")
	fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
    <title>Lucky v2 Text Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .endpoint { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 5px; }
        code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>Lucky v2 Text Classifier API</h1>
    <p>High-performance text classification service powered by character n-grams and TF-IDF.</p>
    
    <div class="endpoint">
        <h3>POST /predict</h3>
        <p>Classify a single text</p>
        <code>{"text": "Restaurant pizza order"}</code>
    </div>
    
    <div class="endpoint">
        <h3>POST /batch</h3>
        <p>Classify multiple texts</p>
        <code>{"texts": ["Restaurant order", "Bus ticket"]}</code>
    </div>
    
    <div class="endpoint">
        <h3>GET /health</h3>
        <p>Health check endpoint</p>
    </div>
    
    <div class="endpoint">
        <h3>GET /categories</h3>
        <p>List available categories</p>
    </div>
    
    <p><strong>Categories:</strong> %d | <strong>Threshold:</strong> %.3f | <strong>Uptime:</strong> %s</p>
</body>
</html>`, len(s.classifier.CatStr), s.classifier.Threshold, time.Since(startTime).String())
}

func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	categories := len(s.classifier.CatStr)
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")

	response := HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Version:   "v2.0.0",
		Uptime:    time.Since(startTime).String(),
	}

	if categories == 0 {
		response.Status = "unhealthy"
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	json.NewEncoder(w).Encode(response)
}

func (s *Server) predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Text == "" {
		http.Error(w, "Text field is required", http.StatusBadRequest)
		return
	}

	start := time.Now()
	s.mu.RLock()
	result := s.classifier.Predict(req.Text)
	s.mu.RUnlock()
	duration := time.Since(start)

	response := PredictionResponse{
		ID:       result.ID,
		Name:     result.Name,
		Score:    result.Score,
		Duration: duration.String(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) batchHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req BatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Texts) == 0 {
		http.Error(w, "Texts array is required and cannot be empty", http.StatusBadRequest)
		return
	}

	if len(req.Texts) > 1000 {
		http.Error(w, "Maximum 1000 texts per batch", http.StatusBadRequest)
		return
	}

	start := time.Now()
	results := make([]PredictionResponse, len(req.Texts))

	s.mu.RLock()
	for i, text := range req.Texts {
		result := s.classifier.Predict(text)
		results[i] = PredictionResponse{
			ID:    result.ID,
			Name:  result.Name,
			Score: result.Score,
		}
	}
	s.mu.RUnlock()

	duration := time.Since(start)

	response := BatchResponse{
		Results:  results,
		Count:    len(results),
		Duration: duration.String(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) categoriesHandler(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	categories := make(map[uint]string)
	for id, name := range s.classifier.CatStr {
		categories[id] = name
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"categories": categories,
		"count":      len(categories),
		"threshold":  s.classifier.Threshold,
	})
}

// Helper functions for environment variables
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if f, err := fmt.Sscanf(value, "%f", &defaultValue); err == nil && f == 1 {
			return defaultValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return value == "true" || value == "1" || value == "yes"
	}
	return defaultValue
}
