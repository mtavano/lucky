package v2

import (
	"strings"
)

// Vote represents accumulated votes for a category
type Vote struct {
	Count uint    // number of votes
	Score float64 // accumulated score
}

// BestGram represents the best n-gram for a category
type BestGram struct {
	Gram string  // the n-gram text
	Prob float64 // probability/score
}

// VotingWeights configures the balance between voting and centroid systems
type VotingWeights struct {
	VotingWeight   float64 // weight for hierarchical voting (0.0-1.0)
	CentroidWeight float64 // weight for centroid prediction (0.0-1.0)
}

// Default voting weights - favor voting for semantic accuracy
var DefaultVotingWeights = VotingWeights{
	VotingWeight:   0.7, // hierarchical voting gets more weight
	CentroidWeight: 0.3, // centroid as fallback
}

// FeatureMap maps n-grams to their category probabilities (adapted from v1's Sample)
type FeatureMap map[string]map[uint]float64

// BuildFeatureMap constructs a feature map from training data for voting
func BuildFeatureMap(samples []Sample) FeatureMap {
	featureMap := make(FeatureMap)
	categoryCount := make(map[uint]int)
	
	// Count samples per category
	for _, sample := range samples {
		categoryCount[sample.Label]++
	}
	
	// Build n-gram statistics
	for _, sample := range samples {
		text := normalize(sample.Text)
		
		// Process word n-grams (1-3) like v1
		for n := 1; n <= 3; n++ {
			ngrams := wordNgrams(text, n, n)
			for _, ngram := range ngrams {
				if featureMap[ngram] == nil {
					featureMap[ngram] = make(map[uint]float64)
				}
				featureMap[ngram][sample.Label]++
			}
		}
	}
	
	// Convert counts to probabilities
	for ngram, categoryFreqs := range featureMap {
		totalFreq := 0.0
		for _, freq := range categoryFreqs {
			totalFreq += freq
		}
		
		if totalFreq > 0 {
			for categoryID, freq := range categoryFreqs {
				// Calculate probability: freq / total_freq_for_ngram
				featureMap[ngram][categoryID] = freq / totalFreq
			}
		}
	}
	
	return featureMap
}

// HierarchicalVoting performs v1-style hierarchical voting using word n-grams
func HierarchicalVoting(text string, featureMap FeatureMap, labelName map[uint]string, threshold float64) *Pred {
	votes := make(map[uint]*Vote)
	
	// Normalize text for consistent processing
	normalizedText := normalize(text)
	
	// Step 1: Find best unigram
	unigrams := wordNgrams(normalizedText, 1, 1)
	bestUnigrams := createBestGram(unigrams, featureMap)
	bestUnigram, bestKey, bestProb := getBestGram(bestUnigrams)
	
	if bestKey != 0 {
		addVote(votes, bestKey, bestProb)
	}
	
	// Step 2: Find bigrams that contain the best unigram
	bigrams := wordNgrams(normalizedText, 2, 2)
	var filteredBigrams []string
	for _, bigram := range bigrams {
		if strings.Contains(bigram, bestUnigram) {
			filteredBigrams = append(filteredBigrams, bigram)
		}
	}
	
	bestBigrams := createBestGram(filteredBigrams, featureMap)
	bestBigram, bestKey, bestProb := getBestGram(bestBigrams)
	
	if bestKey != 0 {
		addVote(votes, bestKey, bestProb)
	}
	
	// Step 3: Find trigrams that contain the best bigram
	trigrams := wordNgrams(normalizedText, 3, 3)
	var filteredTrigrams []string
	for _, trigram := range trigrams {
		if strings.Contains(trigram, bestBigram) {
			filteredTrigrams = append(filteredTrigrams, trigram)
		}
	}
	
	bestTrigrams := createBestGram(filteredTrigrams, featureMap)
	_, bestKey, bestProb = getBestGram(bestTrigrams)
	
	if bestKey != 0 {
		addVote(votes, bestKey, bestProb)
	}
	
	// Get final vote result
	finalID, finalScore := maxVote(votes, threshold)
	
	return &Pred{
		ID:    finalID,
		Name:  labelName[finalID],
		Score: finalScore,
	}
}

// addVote adds or updates a vote for a category
func addVote(votes map[uint]*Vote, categoryID uint, score float64) {
	if vote, exists := votes[categoryID]; exists {
		vote.Count++
		vote.Score += score
	} else {
		votes[categoryID] = &Vote{
			Count: 1,
			Score: score,
		}
	}
}

// createBestGram finds the best n-gram for each category (adapted from v1)
func createBestGram(ngrams []string, featureMap FeatureMap) map[uint]*BestGram {
	bestGrams := make(map[uint]*BestGram)
	
	for _, ngram := range ngrams {
		if categoryProbs, exists := featureMap[ngram]; exists {
			// Find the category with highest probability for this n-gram
			var bestCategory uint
			var bestProb float64
			
			for categoryID, prob := range categoryProbs {
				if prob > bestProb {
					bestCategory = categoryID
					bestProb = prob
				}
			}
			
			if bestCategory != 0 {
				if existing, exists := bestGrams[bestCategory]; exists {
					if bestProb > existing.Prob {
						existing.Prob = bestProb
						existing.Gram = ngram
					}
				} else {
					bestGrams[bestCategory] = &BestGram{
						Gram: ngram,
						Prob: bestProb,
					}
				}
			}
		}
	}
	
	return bestGrams
}

// getBestGram finds the n-gram with highest probability across all categories
func getBestGram(grams map[uint]*BestGram) (string, uint, float64) {
	var maxKey uint
	var maxGram string
	var maxProb float64
	
	for key, value := range grams {
		if value.Prob > maxProb {
			maxKey = key
			maxGram = value.Gram
			maxProb = value.Prob
		}
	}
	
	return maxGram, maxKey, maxProb
}

// maxVote determines the winning category from accumulated votes (adapted from v1)
func maxVote(votes map[uint]*Vote, threshold float64) (uint, float64) {
	if len(votes) == 0 {
		return 0, 0.0
	}
	
	var maxKey uint
	var maxCount uint
	var maxScore float64
	
	// First pass: find highest vote count
	for key, vote := range votes {
		if vote.Count > maxCount {
			maxCount = vote.Count
			maxKey = key
			maxScore = vote.Score
		}
	}
	
	// Second pass: if there's a tie in count, use highest score
	tieCount := 0
	for _, vote := range votes {
		if vote.Count == maxCount {
			tieCount++
		}
	}
	
	if tieCount > 1 {
		maxScore = 0.0
		for key, vote := range votes {
			if vote.Count == maxCount && vote.Score > maxScore {
				maxKey = key
				maxScore = vote.Score
			}
		}
	}
	
	// Apply threshold
	avgScore := maxScore / float64(maxCount)
	if avgScore < threshold {
		return 0, avgScore
	}
	
	return maxKey, avgScore
}

// CombinedPredict combines hierarchical voting with centroid prediction
func CombinedPredict(m *Model, text string, featureMap FeatureMap, weights VotingWeights, topk int) []Pred {
	// Get centroid prediction (baseline)
	centroidPreds := predictOne(m, text, topk)
	
	// Get voting prediction
	votingPred := HierarchicalVoting(text, featureMap, m.LabelName, 0.0) // no threshold for voting
	
	// Combine predictions with weighted scores
	combined := make(map[uint]float64)
	
	// Add centroid scores
	for _, pred := range centroidPreds {
		combined[pred.ID] += pred.Score * weights.CentroidWeight
	}
	
	// Add voting score
	if votingPred.ID != 0 {
		combined[votingPred.ID] += votingPred.Score * weights.VotingWeight
	}
	
	// Convert back to sorted predictions
	type pair struct {
		id    uint
		score float64
	}
	
	var pairs []pair
	for id, score := range combined {
		pairs = append(pairs, pair{id, score})
	}
	
	// Sort by score descending
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].score < pairs[j].score {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}
	
	// Limit to topk
	if topk > len(pairs) {
		topk = len(pairs)
	}
	
	result := make([]Pred, topk)
	for i := 0; i < topk; i++ {
		result[i] = Pred{
			ID:    pairs[i].id,
			Name:  m.LabelName[pairs[i].id],
			Score: pairs[i].score,
		}
	}
	
	return result
}
