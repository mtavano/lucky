package main

import (
	"flag"
	"log"

	v2 "github.com/mtavano/lucky/v2"
)

func main() {
	cmd := flag.String("cmd", "train", "train|predict|eval")
	labels := flag.String("labels", "labels.txt", "labels file path (id#name)")
	data := flag.String("data", "train.txt", "training data (id#text)")
	modelPath := flag.String("model", "model.json", "model output/input")
	threshold := flag.Float64("thr", 0.60, "prediction threshold (cosine)")
	topk := flag.Int("topk", 3, "top-k predictions")
	flag.Parse()

	switch *cmd {
	case "train":
		v2.RunTrain(*labels, *data, *modelPath)
	case "predict":
		v2.RunPredict(*modelPath, *threshold, *topk)
	case "eval":
		v2.RunEval(*labels, *data, *threshold)
	default:
		log.Fatalf("unknown cmd: %s", *cmd)
	}
}
