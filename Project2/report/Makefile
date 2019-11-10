all: report.tex ref.bib
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex
show: report.pdf
	evince report.pdf
clean:
	rm report.aux report.log report.out report.toc report.bbl report.blg
