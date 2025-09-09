package v2

import (
	"testing"
)

func TestBuildFeatureMap(t *testing.T) {
	samples := []Sample{
		{Label: 1, Text: "restaurant food"},
		{Label: 1, Text: "pizza restaurant"},
		{Label: 2, Text: "uber ride"},
		{Label: 2, Text: "taxi ride"},
	}
	
	featureMap := BuildFeatureMap(samples)
	
	// Should have word n-grams
	if featureMap["restaurant"] == nil {
		t.Error("Missing 'restaurant' in feature map")
	}
	
	// Check probabilities
	if prob, exists := featureMap["restaurant"][1]; !exists || prob != 1.0 {
		t.Errorf("Expected 'restaurant' to have probability 1.0 for category 1, got %f", prob)
	}
	
	if prob, exists := featureMap["ride"][2]; !exists || prob != 1.0 {
		t.Errorf("Expected 'ride' to have probability 1.0 for category 2, got %f", prob)
	}
}

func TestHierarchicalVoting(t *testing.T) {
	// Create test feature map
	featureMap := FeatureMap{
		"restaurant": {1: 0.8, 2: 0.2},
		"food":       {1: 0.9, 3: 0.1},
		"uber":       {2: 1.0},
		"ride":       {2: 0.7, 4: 0.3},
		"restaurant food": {1: 1.0},
		"uber ride":       {2: 1.0},
	}
	
	labelName := map[uint]string{
		1: "food",
		2: "transport",
		3: "entertainment",
		4: "other",
	}
	
	// Test food-related text
	result := HierarchicalVoting("restaurant food", featureMap, labelName, 0.0)
	if result.ID != 1 {
		t.Errorf("Expected category 1 (food), got %d (%s)", result.ID, result.Name)
	}
	
	// Test transport-related text
	result = HierarchicalVoting("uber ride", featureMap, labelName, 0.0)
	if result.ID != 2 {
		t.Errorf("Expected category 2 (transport), got %d (%s)", result.ID, result.Name)
	}
}

func TestMaxVote(t *testing.T) {
	votes := map[uint]*Vote{
		1: {Count: 2, Score: 1.5},
		2: {Count: 1, Score: 0.8},
		3: {Count: 2, Score: 1.2}, // tie in count, lower score
	}
	
	winnerID, avgScore := maxVote(votes, 0.0)
	
	// Should pick category 1 (higher score among tied counts)
	if winnerID != 1 {
		t.Errorf("Expected winner to be category 1, got %d", winnerID)
	}
	
	expectedAvg := 1.5 / 2.0 // score / count
	if avgScore != expectedAvg {
		t.Errorf("Expected average score %.3f, got %.3f", expectedAvg, avgScore)
	}
}

func TestMaxVoteThreshold(t *testing.T) {
	votes := map[uint]*Vote{
		1: {Count: 1, Score: 0.3}, // below threshold
	}
	
	winnerID, avgScore := maxVote(votes, 0.5) // threshold = 0.5
	
	// Should return 0 (unknown) due to threshold
	if winnerID != 0 {
		t.Errorf("Expected winner to be 0 (below threshold), got %d", winnerID)
	}
	
	if avgScore != 0.3 {
		t.Errorf("Expected score 0.3, got %.3f", avgScore)
	}
}

func TestCombinedPredict(t *testing.T) {
	// Create minimal model for testing
	model := &Model{
		Centroids: map[uint][]float64{
			1: make([]float64, Dim), // dummy centroids
			2: make([]float64, Dim),
		},
		LabelName: map[uint]string{
			1: "food",
			2: "transport",
		},
		DF:   make([]int, Dim),
		Docs: 10,
	}
	
	featureMap := FeatureMap{
		"restaurant": {1: 0.8},
		"uber":       {2: 0.9},
	}
	
	weights := VotingWeights{
		VotingWeight:   0.6,
		CentroidWeight: 0.4,
	}
	
	// Test combined prediction
	results := CombinedPredict(model, "restaurant", featureMap, weights, 2)
	
	if len(results) == 0 {
		t.Error("Expected at least one prediction result")
	}
	
	// Should have valid category IDs
	for _, result := range results {
		if result.ID == 0 {
			t.Error("Got unknown category (ID=0) in results")
		}
		if result.Name == "" {
			t.Error("Got empty category name")
		}
	}
}

func TestVotingWeights(t *testing.T) {
	// Test default weights
	if DefaultVotingWeights.VotingWeight <= 0 || DefaultVotingWeights.CentroidWeight <= 0 {
		t.Error("Default voting weights should be positive")
	}
	
	// Test that weights sum to reasonable range (not necessarily 1.0)
	sum := DefaultVotingWeights.VotingWeight + DefaultVotingWeights.CentroidWeight
	if sum <= 0 || sum > 2.0 {
		t.Errorf("Voting weights sum seems unreasonable: %.3f", sum)
	}
}

func TestAddVote(t *testing.T) {
	votes := make(map[uint]*Vote)
	
	// Add first vote
	addVote(votes, 1, 0.8)
	if votes[1].Count != 1 || votes[1].Score != 0.8 {
		t.Errorf("First vote not added correctly: count=%d, score=%.3f", 
			votes[1].Count, votes[1].Score)
	}
	
	// Add second vote to same category
	addVote(votes, 1, 0.6)
	if votes[1].Count != 2 || votes[1].Score != 1.4 {
		t.Errorf("Second vote not accumulated correctly: count=%d, score=%.3f", 
			votes[1].Count, votes[1].Score)
	}
	
	// Add vote to different category
	addVote(votes, 2, 0.5)
	if votes[2].Count != 1 || votes[2].Score != 0.5 {
		t.Errorf("Vote for different category not added correctly: count=%d, score=%.3f", 
			votes[2].Count, votes[2].Score)
	}
}

func TestGetBestGram(t *testing.T) {
	grams := map[uint]*BestGram{
		1: {Gram: "restaurant", Prob: 0.8},
		2: {Gram: "uber", Prob: 0.9},
		3: {Gram: "movie", Prob: 0.6},
	}
	
	bestGram, bestKey, bestProb := getBestGram(grams)
	
	if bestKey != 2 {
		t.Errorf("Expected best key to be 2, got %d", bestKey)
	}
	if bestGram != "uber" {
		t.Errorf("Expected best gram to be 'uber', got '%s'", bestGram)
	}
	if bestProb != 0.9 {
		t.Errorf("Expected best prob to be 0.9, got %.3f", bestProb)
	}
}

func TestCreateBestGram(t *testing.T) {
	ngrams := []string{"restaurant", "food", "unknown"}
	featureMap := FeatureMap{
		"restaurant": {1: 0.8, 2: 0.2},
		"food":       {1: 0.9},
		// "unknown" not in feature map
	}
	
	bestGrams := createBestGram(ngrams, featureMap)
	
	// Should have category 1 with "food" (higher prob than "restaurant")
	if bestGrams[1] == nil {
		t.Error("Expected category 1 in best grams")
	} else if bestGrams[1].Gram != "food" || bestGrams[1].Prob != 0.9 {
		t.Errorf("Expected category 1 to have 'food' with prob 0.9, got '%s' with prob %.3f",
			bestGrams[1].Gram, bestGrams[1].Prob)
	}
	
	// Category 2 should not be in best grams because category 1 has higher prob for "restaurant"
	// The algorithm picks the category with highest probability for each n-gram
	if bestGrams[2] != nil {
		t.Errorf("Did not expect category 2 in best grams, but got '%s' with prob %.3f",
			bestGrams[2].Gram, bestGrams[2].Prob)
	}
	
	// Should only have 1 category (category 1 wins both n-grams)
	if len(bestGrams) != 1 {
		t.Errorf("Expected 1 category in best grams, got %d", len(bestGrams))
	}
}
