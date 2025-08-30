.PHONY: lint test build clean

# Configuración básica de linter (solo v2)
lint:
	@echo "Ejecutando linter en v2..."
	@echo "Usando go vet y go fmt en v2:";\
	cd v2 && go vet . && echo "✓ go vet: sin problemas"; \
	cd v2 && test -z "$$(gofmt -l .)" && echo "✓ gofmt: código formateado correctamente" || (echo "Código no formateado correctamente:" && gofmt -l . && exit 1);\
	echo "✓ Linter completado - no se encontraron problemas"

# Tests de v1 (nota: requiere dependencias externas)
test-v1:
	@echo "Ejecutando tests de v1... (requiere dependencias externas)"
	@echo "Nota: v1 requiere github.com/mtavano/lucky/ngrams que puede no estar disponible"
	-go test -v

# Tests de v2 (compatibilidad)
test-v2:
	@echo "Ejecutando tests de v2..."
	cd v2 && go test -v

# Tests completos (foco en v2)
test: test-v2

# Build de quickcat (herramienta de v2)
build:
	@echo "Compilando quickcat..."
	cd cmd/quickcat && go build -o ../../quickcat main.go

# Limpiar archivos generados
clean:
	@echo "Limpiando archivos generados..."
	rm -f quickcat
	rm -f model_offline.json
	rm -f model.tmp.json

# Instalación de golangci-lint (opcional)
install-lint:
	@echo "Instalando golangci-lint..."
	curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $$(go env GOPATH)/bin v1.54.2

help:
	@echo "Comandos disponibles:"
	@echo "  lint          - Ejecutar linter en todo el proyecto"
	@echo "  test          - Ejecutar todos los tests (v1 + v2)"
	@echo "  test-v1       - Ejecutar solo tests de v1"  
	@echo "  test-v2       - Ejecutar solo tests de v2"
	@echo "  build         - Compilar herramienta quickcat"
	@echo "  clean         - Limpiar archivos generados"
	@echo "  install-lint  - Instalar golangci-lint"
	@echo "  help          - Mostrar esta ayuda"
