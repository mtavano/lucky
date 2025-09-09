package v2

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
)

// ---------- File I/O ----------

// readLabels reads category labels from file
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

// readData reads training data from file
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

// writeData writes training data to file
func writeData(path string, data []Sample) {
	f := mustCreate(path)
	defer f.Close()
	for _, s := range data {
		fmt.Fprintf(f, "%d#%s\n", s.Label, s.Text)
	}
}

// saveModel saves model to JSON file
func saveModel(path string, m *Model) {
	b, _ := json.MarshalIndent(m, "", "  ")
	_ = os.WriteFile(path, b, 0644)
}

// loadModel loads model from JSON file
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

// ---------- Utility Functions ----------

// mustOpen opens a file or panics
func mustOpen(p string) *os.File {
	f, err := os.Open(p)
	if err != nil {
		log.Fatal(err)
	}
	return f
}

// mustCreate creates a file or panics
func mustCreate(p string) *os.File {
	f, err := os.Create(p)
	if err != nil {
		log.Fatal(err)
	}
	return f
}

// mustJSON converts value to JSON string
func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
