package v2

import (
	"reflect"
	"testing"
)

func TestWordNgrams(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		nMin     int
		nMax     int
		expected []string
	}{
		{
			name:  "simple text unigrams",
			input: "uber ride",
			nMin:  1,
			nMax:  1,
			expected: []string{
				"<START>", "uber", "ride", "<END>",
			},
		},
		{
			name:  "simple text bigrams",
			input: "uber ride",
			nMin:  2,
			nMax:  2,
			expected: []string{
				"<START> uber", "uber ride", "ride <END>",
			},
		},
		{
			name:  "simple text unigrams and bigrams",
			input: "uber ride",
			nMin:  1,
			nMax:  2,
			expected: []string{
				// unigrams
				"<START>", "uber", "ride", "<END>",
				// bigrams
				"<START> uber", "uber ride", "ride <END>",
			},
		},
		{
			name:     "empty text",
			input:    "",
			nMin:     1,
			nMax:     2,
			expected: nil,
		},
		{
			name:  "single word",
			input: "restaurant",
			nMin:  1,
			nMax:  2,
			expected: []string{
				// unigrams
				"<START>", "restaurant", "<END>",
				// bigrams
				"<START> restaurant", "restaurant <END>",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := wordNgrams(tt.input, tt.nMin, tt.nMax)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("wordNgrams(%q, %d, %d) = %v, want %v", 
					tt.input, tt.nMin, tt.nMax, result, tt.expected)
			}
		})
	}
}

func TestCharVsWordFeatureIndexes(t *testing.T) {
	// Test that char and word features don't collide
	charIdx := charFeatureIdx("test")
	wordIdx := wordFeatureIdx("test")
	
	if charIdx == wordIdx {
		t.Errorf("Collision detected: char and word features have same index %d for 'test'", charIdx)
	}
	
	// Test consistency
	charIdx2 := charFeatureIdx("test")
	wordIdx2 := wordFeatureIdx("test")
	
	if charIdx != charIdx2 {
		t.Errorf("charFeatureIdx inconsistent: %d != %d", charIdx, charIdx2)
	}
	if wordIdx != wordIdx2 {
		t.Errorf("wordFeatureIdx inconsistent: %d != %d", wordIdx, wordIdx2)
	}
}

func TestHybridFeaturize(t *testing.T) {
	text := "uber ride downtown"
	vec := hybridFeaturize(text, nil, 0, 3, 5)
	
	if len(vec) == 0 {
		t.Error("hybridFeaturize returned empty vector")
	}
	
	// We can't easily test specific indices due to hashing, but we can test
	// that the vector has reasonable size and values
	for idx, val := range vec {
		if val <= 0 {
			t.Errorf("Feature at index %d has non-positive value %f", idx, val)
		}
	}
	
	// Basic sanity checks
	if len(vec) < 5 {
		t.Errorf("Expected more features, got %d", len(vec))
	}
}

func TestHybridWeights(t *testing.T) {
	// Test default weights
	defaultW := DefaultWeights
	if defaultW.CharWeight <= 0 || defaultW.WordWeight <= 0 {
		t.Errorf("Default weights should be positive: char=%f, word=%f", 
			defaultW.CharWeight, defaultW.WordWeight)
	}
	
	// Test model weight methods
	model := &Model{}
	
	// Should return defaults when no weights set
	weights := model.GetHybridWeights()
	if weights.CharWeight != DefaultWeights.CharWeight || weights.WordWeight != DefaultWeights.WordWeight {
		t.Errorf("GetHybridWeights() should return defaults when not set")
	}
	
	// Test setting custom weights
	model.SetHybridWeights(0.7, 0.3)
	weights = model.GetHybridWeights()
	if weights.CharWeight != 0.7 || weights.WordWeight != 0.3 {
		t.Errorf("SetHybridWeights not working: got char=%f, word=%f", 
			weights.CharWeight, weights.WordWeight)
	}
}

func TestHybridFeaturizeWithWeights(t *testing.T) {
	text := "restaurant food"
	
	// Test with different weights
	weights1 := HybridWeights{CharWeight: 1.0, WordWeight: 0.0} // char only
	weights2 := HybridWeights{CharWeight: 0.0, WordWeight: 1.0} // word only
	weights3 := HybridWeights{CharWeight: 0.5, WordWeight: 0.5} // balanced
	
	vec1 := hybridFeaturizeWithWeights(text, nil, 0, 3, 5, weights1)
	vec2 := hybridFeaturizeWithWeights(text, nil, 0, 3, 5, weights2)
	vec3 := hybridFeaturizeWithWeights(text, nil, 0, 3, 5, weights3)
	
	if len(vec1) == 0 || len(vec2) == 0 || len(vec3) == 0 {
		t.Error("hybridFeaturizeWithWeights returned empty vectors")
	}
	
	// vec1 should have different features than vec2 due to different weights
	// (though they might overlap due to hashing, the values should be different)
	
	// Check that balanced weights produces non-zero vectors
	for idx, val := range vec3 {
		if val <= 0 {
			t.Errorf("Balanced weights produced non-positive value %f at index %d", val, idx)
		}
	}
}

// Benchmark tests
func BenchmarkCharNgramsOnly(b *testing.B) {
	text := "compra en supermercado con tarjeta de credito"
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		// Simulate old char-only approach
		vec := map[int]float64{}
		ngrams := charNgrams(text, 3, 5)
		for _, ng := range ngrams {
			idx := charFeatureIdx(ng)
			vec[idx] += 1.0
		}
	}
}

func BenchmarkHybridFeaturize(b *testing.B) {
	text := "compra en supermercado con tarjeta de credito"
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_ = hybridFeaturize(text, nil, 0, 3, 5)
	}
}

func BenchmarkWordNgrams(b *testing.B) {
	text := "compra en supermercado con tarjeta de credito"
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_ = wordNgrams(text, 1, 3)
	}
}
