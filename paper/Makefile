# LaTeX Makefile
LATEXCOMPILER=pdflatex
BIBCOMPILER=bibtex

# file names without extensions (.tex, .bib, .pdf)
BIB=refs
TARGET=main


all: $(TARGET)

$(TARGET): $(TARGET).tex $(BIB).bib
	$(LATEXCOMPILER) $(TARGET).tex
	$(BIBCOMPILER) $(TARGET)
	$(LATEXCOMPILER) $(TARGET).tex
	$(LATEXCOMPILER) $(TARGET).tex


.PHONY: clean

clean:
	rm -vf *.aux *.blg *.bbl *.log \
			*.out *.brf *.idx *.ilg \
			*.ind *.lof *.lol *.lot \
			*.toc *.synctex.gz
rmpdf:
	rm -vf $(TARGET).pdf
