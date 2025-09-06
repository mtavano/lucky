# Lucky - Fast Text Classification for Go

Lucky is a high-performance text classifier written in Go, designed for categorizing transaction descriptions, log entries, or any short text snippets. It uses character n-grams with TF-IDF and centroid-based classification for fast and accurate predictions.

## Features

- **Fast**: ~7.8μs per prediction (300x faster than v1)
- **Lightweight**: No external dependencies, compact memory footprint
- **Accurate**: Character n-grams (3-5) with advanced normalization
- **Production-ready**: Save/load trained models, threshold-based filtering
- **Compatible**: v2 is a drop-in replacement for v1

## Quick Start

### 1. Install

```bash
go get github.com/mtavano/lucky/v2
```

### 2. Prepare Training Data

Create two files with your training data:

**labels.txt** (category definitions):

```text
1#Food & Dining
2#Transportation  
3#Entertainment
4#Shopping
```

**training.txt** (labeled examples):

```text
1#Restaurant pizza delivery
1#Grocery store vegetables
2#Uber ride downtown
2#Metro card top-up
3#Netflix subscription
3#Movie theater tickets
4#Online shopping Amazon
4#Department store clothes
```

<<<<<<< Updated upstream
## Basic model
```go
type Samples struct {
    Ngram    string
    Freq     float64
    Classes  map[uint]float64
    Probs    map[uint]float64
    Maximum  float64
    Minimum  float64
    Weighted bool
}
```

## Setup structure
```go
type Lucky struct {
    Model            map[string]*model.Samples
    CatNum           map[uint]float64
    CatStr           map[uint]string
    LabelsPath       string
    TrainingDataPath string
    URL              string
}
```

## Public response
```go
type BestCategory struct {
  ID    uint    // category id
  Name  string  // category name
  Score float64 // category probability
}
```

### Public Methods
```go
Fit() void
Predict(test string) (*model.BestCategory)
```

To see the classifier in action see `lucky_test.go`
<<<<<<< Updated upstream


---

## V2 integration

```go
import v2 "github.com/mtavano/lucky/v2"

// Para tu servicio, solo necesitas esto:
func initClassifier(modelPath string) (*v2.Config, error) {
    config := &v2.Config{
        Threshold: 0.3,  // tu umbral de confianza
        Verbose:   false, // sin logs en prod
    }
    
    err := config.LoadModel(modelPath)  // ← AQUÍ ES EL LOAD
    if err != nil {
        return nil, fmt.Errorf("error cargando modelo: %v", err)
    }
    
    return config, nil
}

// Usar en tu servicio:
func main() {
    classifier, err := initClassifier("/path/to/your/model.json")
=======
=======
### 3. Train and Use

```go
package main

import (
    "fmt"
    "log"
    v2 "github.com/mtavano/lucky/v2"
)

func main() {
    // Configure classifier
    config := &v2.Config{
        LabelsPath:       "labels.txt",
        TrainingDataPath: "training.txt", 
        Threshold:        0.3, // minimum confidence
        Verbose:          true,
    }
    
    // Train model
    config.Fit()
    
    // Save trained model
    err := config.SaveModel("model.json")
>>>>>>> Stashed changes
    if err != nil {
        log.Fatal(err)
    }
    
<<<<<<< Updated upstream
    // Listo para usar
    result := classifier.Predict("COMPRA SUPERMERCADO")
    fmt.Printf("Categoría: %s (ID: %d, Score: %.3f)\n", 
        result.Name, result.ID, result.Score)
}
```
=======
    // Make predictions
    result := config.Predict("Coffee shop purchase")
    fmt.Printf("Category: %s (ID: %d, Score: %.3f)\n", 
        result.Name, result.ID, result.Score)
}
```

### 4. Production Usage (Load Pre-trained Model)

```go
package main

import (
    "fmt"
    "log"
    v2 "github.com/mtavano/lucky/v2"
)

func main() {
    // Load pre-trained model
    config := &v2.Config{
        Threshold: 0.3,
        Verbose:   false, // no logs in production
    }
    
    err := config.LoadModel("model.json")
    if err != nil {
        log.Fatal("Failed to load model:", err)
    }
    
    // Ready to classify
    result := config.Predict("Supermarket groceries")
    if result.ID == 0 {
        fmt.Println("UNKNOWN category (below threshold)")
    } else {
        fmt.Printf("Category: %s (confidence: %.3f)\n", 
            result.Name, result.Score)
    }
}
```

## Advanced Usage

### Command Line Tools

Train a model:

```bash
cd v2/
go run . -train=training.txt -labels=labels.txt -out=model.json
```

Evaluate model accuracy:

```bash
go run . -eval -labels=labels.txt -data=training.txt -thr=0.3
```

Interactive prediction:

```bash
go run . -predict -model=model.json -thr=0.3
```

### Performance Testing

```bash
cd v2/
go test -bench=. -benchmem
```

## Migration from v1

Lucky v2 is 100% API compatible with v1. Simply change the import:

**Before (v1):**

```go
import "github.com/mtavano/lucky"

config := &lucky.Config{
    LabelsPath: "labels.txt",
    TrainingDataPath: "training.txt",
    Threshold: 0.3,
}
```

**After (v2):**

```go
import v2 "github.com/mtavano/lucky/v2"

config := &v2.Config{  // Only change needed
    LabelsPath: "labels.txt",
    TrainingDataPath: "training.txt", 
    Threshold: 0.3,
}
```

All methods (`Fit()`, `Predict()`) work exactly the same.

## Algorithm Details

### v1 vs v2 Comparison

| Aspect | v1 | v2 |
|--------|----|----|
| Algorithm | Word n-grams + voting | Character n-grams + centroids |
| Vectorization | Manual | TF-IDF + hashing trick |
| Classification | Majority voting | Cosine similarity |
| Normalization | Basic | Advanced (amount/date/code placeholders) |
| Performance | ~1-3ms | ~7.8μs |
| Memory | Higher | Compact sparse vectors |

### v2 Technical Features

- **Character n-grams (3-5)**: Better handling of typos and partial matches
- **TF-IDF weighting**: Improved feature importance scoring
- **Hashing trick**: 65K feature space with collision handling
- **Advanced normalization**: Replaces amounts, dates, and codes with placeholders
- **Centroid classification**: Fast cosine similarity with pre-computed centroids
- **Sparse vectors**: Memory-efficient representation

## Data Format

### Training Data Format
Each line: `categoryID#description`

```text
1#Restaurant lunch with colleagues
2#Bus ticket for commute
3#Movie theater evening show
```

### Labels Format  
Each line: `categoryID#categoryName`

```text
1#Food & Dining
2#Transportation
3#Entertainment
```

## API Reference

### Config Struct

```go
type Config struct {
    LabelsPath       string  // path to labels file
    TrainingDataPath string  // path to training data
    Threshold        float64 // minimum confidence (0.0-1.0)
    Verbose          bool    // enable training logs
}
```

### Methods

```go
func (c *Config) Fit()                        // train the model
func (c *Config) Predict(text string) *BestCategory // classify text
func (c *Config) SaveModel(path string) error       // save trained model
func (c *Config) LoadModel(path string) error       // load pre-trained model
func (c *Config) GetModelPath() string              // default model path
```

### Response Type

```go
type BestCategory struct {
    ID    uint    // category ID (0 = unknown/below threshold)
    Name  string  // category name
    Score float64 // confidence score (0.0-1.0)
}
```

## Testing

Run all tests:

```bash
make test
```

Or manually:

```bash
cd v2/
go test -v
go test -bench=.
```

## License

MIT License - see LICENSE file for details.
>>>>>>> Stashed changes
>>>>>>> Stashed changes
