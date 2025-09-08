package v2

const Dim = 1 << 16 // feature space size (hashing trick) - 65K en lugar de 1M

// HybridWeights defines the relative importance of char vs word features
type HybridWeights struct {
	CharWeight float64 // weight for character n-grams (robustness)
	WordWeight float64 // weight for word n-grams (semantics)
}

// Default weights based on empirical testing - favor char n-grams for typo robustness
var DefaultWeights = HybridWeights{
	CharWeight: 0.6, // slightly favor char n-grams for typo robustness
	WordWeight: 0.4, // but include word semantics
}

// Model represents a trained classifier model
type Model struct {
	Centroids map[uint][]float64 `json:"centroids"` // categoryID -> centroid vector
	LabelName map[uint]string    `json:"label_name"`
	DF        []int              `json:"df"`     // doc freq per hashed index (for IDF)
	Docs      int                `json:"docs"`   // number of docs used to compute IDF
	Params    map[string]any     `json:"params"` // metadata (ngrams, τ, etc.)
	Weights   *HybridWeights     `json:"weights,omitempty"` // hybrid feature weights
}

// Sample represents a labeled training sample
type Sample struct {
	Label uint
	Text  string
}

// Pred represents a prediction result
type Pred struct {
	ID    uint    `json:"id"`
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

// SetHybridWeights configures the weights for hybrid feature extraction
func (m *Model) SetHybridWeights(charWeight, wordWeight float64) {
	m.Weights = &HybridWeights{
		CharWeight: charWeight,
		WordWeight: wordWeight,
	}
}

// GetHybridWeights returns the hybrid weights, using defaults if not set
func (m *Model) GetHybridWeights() HybridWeights {
	if m.Weights != nil {
		return *m.Weights
	}
	return DefaultWeights
}
