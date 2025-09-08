package v2

import (
	"fmt"
	"math"
)

// RunTrain trains a model from labeled data and saves it
func RunTrain(labelsPath, dataPath, outPath string) {
	labelName := readLabels(labelsPath)
	samples := readData(dataPath)

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
	// average & densify to fixed slice (sparse -> dense is simple here for speed to prod)
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
		// L2-normalize centroid => cosine scores in [0,1]
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

	model := &Model{
		Centroids: centroids,
		LabelName: labelName,
		DF:        df,
		Docs:      docs,
		Params: map[string]any{
			"ngrams":     "hybrid(char:3-5,word:1-3)",
			"dim":        Dim,
			"classifier": "centroid-cosine",
			"version":    "2.2",
			"features":   "char+word",
			"weights":    fmt.Sprintf("c:%.1f,w:%.1f", DefaultWeights.CharWeight, DefaultWeights.WordWeight),
		},
		Weights: &DefaultWeights,
	}
	saveModel(outPath, model)
	fmt.Println("ok: wrote", outPath)
}

// RunEval evaluates model accuracy using 80/20 split
func RunEval(labelsPath, dataPath string, thr float64) {
	labelName := readLabels(labelsPath)
	_ = labelName
	all := readData(dataPath)
	byLabel := map[uint][]string{}
	for _, s := range all {
		byLabel[s.Label] = append(byLabel[s.Label], s.Text)
	}
	var train, test []Sample
	for lab, arr := range byLabel {
		n := int(math.Max(1, math.Round(float64(len(arr))*0.8)))
		for i, t := range arr {
			if i < n {
				train = append(train, Sample{Label: lab, Text: t})
			} else {
				test = append(test, Sample{Label: lab, Text: t})
			}
		}
	}
	// build a temp model from 'train'
	tmpData := "/tmp/train_split.txt"
	writeData(tmpData, train)
	RunTrain(labelsPath, tmpData, "model.tmp.json")
	m := loadModel("model.tmp.json")

	// evaluate
	var correct, total, covered int
	for _, s := range test {
		best := predictOne(m, s.Text, 1)[0]
		total++
		if best.Score >= thr {
			covered++
		}
		if best.ID == s.Label {
			correct++
		}
	}
	acc := float64(correct) / float64(total)
	cov := float64(covered) / float64(total)
	fmt.Printf("accuracy=%.4f coverage@thr=%.4f total=%d\n", acc, cov, total)
}
