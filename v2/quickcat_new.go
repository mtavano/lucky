package v2

import (
	"flag"
	"fmt"
	"os"
	"strconv"
)

// CLI entry point and main functions
func main() {
	var (
		trainFlag   = flag.String("train", "", "Training data path")
		labelsFlag  = flag.String("labels", "", "Labels file path")
		outFlag     = flag.String("out", "model.json", "Output model path")
		evalFlag    = flag.Bool("eval", false, "Evaluate model accuracy")
		predictFlag = flag.Bool("predict", false, "Interactive prediction mode")
		modelFlag   = flag.String("model", "model.json", "Model file path")
		thrFlag     = flag.Float64("thr", 0.3, "Confidence threshold")
		topkFlag    = flag.Int("topk", 3, "Top-k predictions")
	)
	flag.Parse()

	switch {
	case *trainFlag != "" && *labelsFlag != "":
		// Training mode
		fmt.Printf("Training model from %s and %s...\n", *trainFlag, *labelsFlag)
		RunTrain(*labelsFlag, *trainFlag, *outFlag)
		
	case *evalFlag:
		// Evaluation mode
		if *labelsFlag == "" {
			fmt.Println("Error: -labels required for evaluation")
			os.Exit(1)
		}
		dataPath := *trainFlag
		if dataPath == "" {
			fmt.Println("Error: -train required for evaluation (as test data)")
			os.Exit(1)
		}
		fmt.Printf("Evaluating model with threshold %.2f...\n", *thrFlag)
		RunEval(*labelsFlag, dataPath, *thrFlag)
		
	case *predictFlag:
		// Prediction mode
		fmt.Printf("Loading model from %s...\n", *modelFlag)
		RunPredict(*modelFlag, *thrFlag, *topkFlag)
		
	default:
		// Show usage
		fmt.Println("Lucky v2.2 - Fast Text Classification")
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  Training:   go run . -train=data.txt -labels=labels.txt -out=model.json")
		fmt.Println("  Evaluation: go run . -eval -labels=labels.txt -train=data.txt -thr=0.3")
		fmt.Println("  Prediction: go run . -predict -model=model.json -thr=0.3")
		fmt.Println()
		fmt.Println("Flags:")
		flag.PrintDefaults()
	}
}

// ParseArgs parses command line arguments for standalone usage
func ParseArgs() (mode string, config map[string]interface{}) {
	if len(os.Args) < 2 {
		return "help", nil
	}
	
	config = make(map[string]interface{})
	
	// Simple argument parsing for backwards compatibility
	for i, arg := range os.Args[1:] {
		switch arg {
		case "-train":
			mode = "train"
			if i+1 < len(os.Args[1:]) {
				config["train"] = os.Args[i+2]
			}
		case "-eval":
			mode = "eval"
		case "-predict":
			mode = "predict"
		}
		
		// Parse key=value arguments
		if len(arg) > 1 && arg[0] == '-' && len(os.Args) > i+2 {
			key := arg[1:]
			value := os.Args[i+2]
			
			switch key {
			case "labels":
				config["labels"] = value
			case "out":
				config["out"] = value
			case "model":
				config["model"] = value
			case "thr":
				if f, err := strconv.ParseFloat(value, 64); err == nil {
					config["thr"] = f
				}
			case "topk":
				if i, err := strconv.Atoi(value); err == nil {
					config["topk"] = i
				}
			}
		}
	}
	
	return mode, config
}
