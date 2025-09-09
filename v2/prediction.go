package v2

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

// RunPredict runs interactive prediction from stdin
func RunPredict(modelPath string, thr float64, topk int) {
	m := loadModel(modelPath)
	in := bufio.NewScanner(os.Stdin)
	fmt.Println("# enter one transaction description per line:")
	for in.Scan() {
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		preds := predictOneHierarchical(m, line, topk)
		best := preds[0]
		if best.Score < thr {
			fmt.Printf("{\"id\":0,\"name\":\"UNKNOWN\",\"score\":%.4f,\"topk\":%s}\n", best.Score, mustJSON(preds))
		} else {
			fmt.Printf("{\"id\":%d,\"name\":\"%s\",\"score\":%.4f,\"topk\":%s}\n", best.ID, best.Name, best.Score, mustJSON(preds))
		}
	}
}

// predictOne predicts using centroids only (for backward compatibility)
func predictOne(m *Model, text string, topk int) []Pred {
	// Use model's hybrid weights if available, otherwise use defaults
	weights := m.GetHybridWeights()
	vec := hybridFeaturizeWithWeights(normalize(text), m.DF, m.Docs, 3, 5, weights)
	type pair struct {
		id uint
		s  float64
	}
	bag := make([]pair, 0, len(m.Centroids))
	for id, c := range m.Centroids {
		// cosine(vec, c) = sum_i vec[i]*c[i]  (c ya normalizado aprox. por promedio)
		var dot float64
		for i, v := range vec {
			dot += v * c[i]
		}
		bag = append(bag, pair{id, dot})
	}
	sort.Slice(bag, func(i, j int) bool { return bag[i].s > bag[j].s })
	if topk > len(bag) {
		topk = len(bag)
	}
	out := make([]Pred, 0, topk)
	for i := 0; i < topk; i++ {
		id := bag[i].id
		out = append(out, Pred{ID: id, Name: m.LabelName[id], Score: bag[i].s})
	}
	return out
}

// predictOneHierarchical predicts using combined centroid + hierarchical voting
func predictOneHierarchical(m *Model, text string, topk int) []Pred {
	// If no feature map available, fall back to centroid-only
	if m.FeatureMap == nil || len(m.FeatureMap) == 0 {
		return predictOne(m, text, topk)
	}
	
	// Use combined prediction with voting
	return CombinedPredict(m, text, m.FeatureMap, DefaultVotingWeights, topk)
}
