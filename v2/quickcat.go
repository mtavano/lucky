package v2

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const Dim = 1 << 16 // feature space size (hashing trick) - 65K en lugar de 1M

// Spanish stopwords map for native filtering (no external dependencies)
var spanishStopwords = map[string]bool{
	// Artículos
	"el": true, "la": true, "los": true, "las": true,
	"un": true, "una": true, "unos": true, "unas": true,
	// Preposiciones
	"de": true, "del": true, "al": true, "en": true,
	"con": true, "por": true, "para": true, "sin": true,
	"sobre": true, "bajo": true, "entre": true, "desde": true,
	"hasta": true, "hacia": true, "durante": true,
	// Conjunciones y conectores
	"y": true, "o": true, "que": true, "pero": true,
	"si": true, "como": true, "cuando": true, "donde": true,
	"quien": true, "cual": true, "cuyo": true,
	// Pronombres
	"se": true, "le": true, "lo": true, "me": true,
	"te": true, "nos": true, "les": true, "su": true,
	"mi": true, "tu": true, "yo": true, "él": true,
	"ella": true, "eso": true, "esto": true, "esos": true,
	// Verbos auxiliares y ser/estar
	"es": true, "son": true, "fue": true, "ser": true,
	"esta": true, "está": true, "están": true, "estar": true,
	"ha": true, "han": true, "he": true, "haber": true,
	"hay": true, "había": true, "hubo": true,
	// Adverbios comunes
	"no": true, "muy": true, "más": true,
	"menos": true, "tan": true, "tanto": true, "ya": true,
	"aún": true, "también": true, "solo": true, "sólo": true,
	// Específicos financieros que pueden ser ruido
	"pago": true, "cobro": true, "cargo": true, "abono": true,
	"saldo": true, "cuenta": true, "banco": true, "tarjeta": true,
}

type Model struct {
	Centroids map[uint][]float64 `json:"centroids"` // categoryID -> centroid vector
	LabelName map[uint]string    `json:"label_name"`
	DF        []int              `json:"df"`     // doc freq per hashed index (for IDF)
	Docs      int                `json:"docs"`   // number of docs used to compute IDF
	Params    map[string]any     `json:"params"` // metadata (ngrams, τ, etc.)
}

type Sample struct {
	Label uint
	Text  string
}

// ---------- Normalizer ----------

var amountRe = regexp.MustCompile(`(?:\$|CLP|\bUSD\b)?\s*\d{1,3}(?:[\.\s]\d{3})*(?:,\d{1,2})?`)
var dateRe = regexp.MustCompile(`\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b`)
var codeRe = regexp.MustCompile(`\b[A-Z0-9]{6,}\b`)

// removeSpanishStopwords filters out Spanish stopwords from text
func removeSpanishStopwords(s string) string {
	words := strings.Fields(s)
	filtered := make([]string, 0, len(words))
	
	for _, word := range words {
		// Keep word if it's not a stopword and has minimum length
		if !spanishStopwords[word] && len(word) > 1 {
			filtered = append(filtered, word)
		}
	}
	
	return strings.Join(filtered, " ")
}

func normalize(s string) string {
	// lowercase + remove accents
	s = strings.ToLower(stripAccents(s))
	// remove Spanish stopwords (after lowercase for proper matching)
	s = removeSpanishStopwords(s)
	// placeholders
	s = amountRe.ReplaceAllString(s, "<amount>")
	s = dateRe.ReplaceAllString(s, "<date>")
	s = codeRe.ReplaceAllString(s, "<code>")
	// simple merchant aliases (extend as needed)
	s = merchantAlias(s)
	// remove punctuation (keep slashes & dots that can be signal if needed)
	builder := strings.Builder{}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsSpace(r) || r == '/' || r == '.' {
			builder.WriteRune(r)
		} else {
			builder.WriteRune(' ')
		}
	}
	out := strings.Join(strings.Fields(builder.String()), " ")
	return out
}

func stripAccents(s string) string {
	// simplistic accent removal
	var b strings.Builder
	for _, r := range s {
		decomp := unicode.SimpleFold(r)
		_ = decomp
		switch r {
		case 'á', 'à', 'ä', 'â', 'ã', 'å':
			r = 'a'
		case 'é', 'è', 'ë', 'ê':
			r = 'e'
		case 'í', 'ì', 'ï', 'î':
			r = 'i'
		case 'ó', 'ò', 'ö', 'ô', 'õ':
			r = 'o'
		case 'ú', 'ù', 'ü', 'û':
			r = 'u'
		case 'ñ':
			r = 'n'
		}
		b.WriteRune(r)
	}
	return b.String()
}

func merchantAlias(s string) string {
	repl := []struct{ from, to string }{
		// existentes / ejemplo
		{"apl itunes.com/bill", "apple itunes"},
		{"itunes.com/bill", "apple itunes"},
		// autopistas/tag (Chile)
		{"aut. central", "autopista central"},
		{"autopista central", "autopista central"},
		{"costanera norte", "autopista costanera norte"},
		{"vesp. sur", "autopista vespucio sur"},
		{"vesp sur", "autopista vespucio sur"},
		{"vespucio sur", "autopista vespucio sur"},
		{"vespucio norte express", "autopista vespucio norte express"},
		{"tag vespucio", "autopista tag"},
		{"pago tag", "autopista tag"},
		{"prepago tag", "autopista tag"},
		{"boleta tag", "autopista tag"},
		{"cobro tag", "autopista tag"},
	}
	for _, r := range repl {
		s = strings.ReplaceAll(s, r.from, r.to)
	}
	return s
}

// ---------- Hashing TF-IDF (char n-grams) ----------

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

func hidx(ng string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(ng))
	return int(h.Sum32()) & (Dim - 1)
}

func featurize(doc string, df []int, docs int, nMin, nMax int) (vec map[int]float64) {
	// term frequencies
	vec = map[int]float64{}
	ngrams := charNgrams(doc, nMin, nMax)
	for _, ng := range ngrams {
		i := hidx(ng)
		vec[i] += 1.0
	}
	// TF-IDF: tf * log((1+N)/(1+df)) + 1
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
	return
}

// ---------- Training (centroids) ----------

func RunTrain(labelsPath, dataPath, outPath string) {
	labelName := readLabels(labelsPath)
	samples := readData(dataPath)

	// compute DF for IDF
	df := make([]int, Dim)
	docs := len(samples)
	for _, s := range samples {
		text := normalize(s.Text)
		seen := map[int]bool{}
		for idx := range featurize(text, nil, 0, 3, 5) {
			if !seen[idx] {
				df[idx] = df[idx] + 1
				seen[idx] = true
			}
		}
	}

	// accumulate centroids
	acc := make(map[uint]map[int]float64)
	count := make(map[uint]int)
	for _, s := range samples {
		text := normalize(s.Text)
		vec := featurize(text, df, docs, 3, 5)
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
			"ngrams":     "char(3-5)",
			"dim":        Dim,
			"classifier": "centroid-cosine",
		},
	}
	saveModel(outPath, model)
	fmt.Println("ok: wrote", outPath)
}

// ---------- Predict ----------

type Pred struct {
	ID    uint    `json:"id"`
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

func RunPredict(modelPath string, thr float64, topk int) {
	m := loadModel(modelPath)
	in := bufio.NewScanner(os.Stdin)
	fmt.Println("# enter one transaction description per line:")
	for in.Scan() {
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		preds := predictOne(m, line, topk)
		best := preds[0]
		if best.Score < thr {
			fmt.Printf("{\"id\":0,\"name\":\"UNKNOWN\",\"score\":%.4f,\"topk\":%s}\n", best.Score, mustJSON(preds))
		} else {
			fmt.Printf("{\"id\":%d,\"name\":\"%s\",\"score\":%.4f,\"topk\":%s}\n", best.ID, best.Name, best.Score, mustJSON(preds))
		}
	}
}

func predictOne(m *Model, text string, topk int) []Pred {
	vec := featurize(normalize(text), m.DF, m.Docs, 3, 5)
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

// ---------- Eval (80/20 quick & dirty) ----------

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
	tmpData := filepath.Join(os.TempDir(), "train_split.txt")
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

// ---------- IO ----------

func readLabels(path string) map[uint]string {
	f := mustOpen(path)
	defer f.Close()
	out := map[uint]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "#", 2)
		if len(parts) != 2 {
			continue
		}
		var id uint
		fmt.Sscanf(parts[0], "%d", &id)
		out[id] = strings.TrimSpace(parts[1])
	}
	return out
}

func readData(path string) []Sample {
	f := mustOpen(path)
	defer f.Close()
	var out []Sample
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "#", 2)
		if len(parts) != 2 {
			continue
		}
		var id uint
		fmt.Sscanf(parts[0], "%d", &id)
		out = append(out, Sample{Label: id, Text: parts[1]})
	}
	return out
}

func writeData(path string, data []Sample) {
	f := mustCreate(path)
	defer f.Close()
	for _, s := range data {
		fmt.Fprintf(f, "%d#%s\n", s.Label, s.Text)
	}
}

func saveModel(path string, m *Model) {
	b, _ := json.MarshalIndent(m, "", "  ")
	_ = os.WriteFile(path, b, 0644)
}

func loadModel(path string) *Model {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var m Model
	if err := json.Unmarshal(b, &m); err != nil {
		log.Fatal(err)
	}
	return &m
}

func mustOpen(p string) *os.File {
	f, err := os.Open(p)
	if err != nil {
		log.Fatal(err)
	}
	return f
}
func mustCreate(p string) *os.File {
	f, err := os.Create(p)
	if err != nil {
		log.Fatal(err)
	}
	return f
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
