package v2

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode/utf8"
)

// ---------- N-gram Generation ----------

// wordNgrams generates word-level n-grams from normalized text
func wordNgrams(s string, nMin, nMax int) []string {
	words := strings.Fields(s)
	if len(words) == 0 {
		return nil
	}
	
	// Add boundary markers for word context
	words = append([]string{"<START>"}, words...)
	words = append(words, "<END>")
	
	var ngrams []string
	for n := nMin; n <= nMax; n++ {
		for i := 0; i+n <= len(words); i++ {
			ngram := strings.Join(words[i:i+n], " ")
			ngrams = append(ngrams, ngram)
		}
	}
	return ngrams
}

// charNgrams generates character-level n-grams from text
func charNgrams(s string, nMin, nMax int) []string {
	// add boundaries for better signal
	s = "^" + s + "$"
	ngrams := make([]string, 0, len(s)*2)
	for n := nMin; n <= nMax; n++ {
		for i := 0; i+n <= len(s); i++ {
			sub := s[i : i+n]
			if utf8.ValidString(sub) {
				ngrams = append(ngrams, sub)
			}
		}
	}
	return ngrams
}

// ---------- Hash Functions ----------

// hidx generates hash index for n-gram
func hidx(ng string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(ng))
	return int(h.Sum32()) & (Dim - 1)
}

// charFeatureIdx generates hash index for character n-grams with prefix to avoid collisions
func charFeatureIdx(ng string) int {
	return hidx("c:" + ng)
}

// wordFeatureIdx generates hash index for word n-grams with prefix to avoid collisions
func wordFeatureIdx(ng string) int {
	return hidx("w:" + ng)
}

// ---------- Feature Extraction ----------

// hybridFeaturize combines character and word n-grams with configurable weights
func hybridFeaturize(doc string, df []int, docs int, nMin, nMax int) (vec map[int]float64) {
	vec = map[int]float64{}
	
	// Character n-grams (robustness to typos)
	charNgrams := charNgrams(doc, nMin, nMax)
	for _, ng := range charNgrams {
		i := charFeatureIdx(ng)
		vec[i] += DefaultWeights.CharWeight
	}
	
	// Word n-grams (semantic understanding) - unigrams, bigrams, trigrams
	wordNgrams := wordNgrams(doc, 1, 3)
	for _, ng := range wordNgrams {
		i := wordFeatureIdx(ng)
		vec[i] += DefaultWeights.WordWeight
	}
	
	// Apply TF-IDF weighting
	for i, tf := range vec {
		idf := 1.0
		if df != nil && docs > 0 {
			dfi := df[i]
			idf = math.Log(float64(1+docs)/float64(1+dfi)) + 1.0
		}
		vec[i] = tf * idf
	}
	
	// L2 normalize
	l2 := 0.0
	for _, v := range vec {
		l2 += v * v
	}
	l2 = math.Sqrt(l2)
	if l2 > 0 {
		for i, v := range vec {
			vec[i] = v / l2
		}
	}
	
	return vec
}

// hybridFeaturizeWithWeights uses custom weights for feature extraction
func hybridFeaturizeWithWeights(doc string, df []int, docs int, nMin, nMax int, weights HybridWeights) (vec map[int]float64) {
	vec = map[int]float64{}
	
	// Character n-grams (robustness to typos)
	charNgrams := charNgrams(doc, nMin, nMax)
	for _, ng := range charNgrams {
		i := charFeatureIdx(ng)
		vec[i] += weights.CharWeight
	}
	
	// Word n-grams (semantic understanding) - unigrams, bigrams, trigrams
	wordNgrams := wordNgrams(doc, 1, 3)
	for _, ng := range wordNgrams {
		i := wordFeatureIdx(ng)
		vec[i] += weights.WordWeight
	}
	
	// Apply TF-IDF weighting
	for i, tf := range vec {
		idf := 1.0
		if df != nil && docs > 0 {
			dfi := df[i]
			idf = math.Log(float64(1+docs)/float64(1+dfi)) + 1.0
		}
		vec[i] = tf * idf
	}
	
	// L2 normalize
	l2 := 0.0
	for _, v := range vec {
		l2 += v * v
	}
	l2 = math.Sqrt(l2)
	if l2 > 0 {
		for i, v := range vec {
			vec[i] = v / l2
		}
	}
	
	return vec
}

// featurize - backward compatibility wrapper (will be deprecated)
func featurize(doc string, df []int, docs int, nMin, nMax int) (vec map[int]float64) {
	return hybridFeaturize(doc, df, docs, nMin, nMax)
}
