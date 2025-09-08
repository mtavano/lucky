package v2

import (
	"regexp"
	"strings"
	"unicode"
)

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

// Regular expressions for normalization
var (
	amountRe = regexp.MustCompile(`(?:\$|CLP|\bUSD\b)?\s*\d{1,3}(?:[\.\s]\d{3})*(?:,\d{1,2})?`)
	dateRe   = regexp.MustCompile(`\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b`)
	codeRe   = regexp.MustCompile(`\b[A-Z0-9]{6,}\b`)
)

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

// normalize applies comprehensive text normalization
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

// stripAccents removes accents from Spanish text
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

// merchantAlias applies domain-specific merchant name normalization
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
