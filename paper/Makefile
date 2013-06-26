all: main.pdf

main.pdf: main.tex
	echo x | pdflatex main.tex

clean:
	rm main.aux main.log mainNotes.bib

watch:
	watchmedo shell-command "--patterns=*.tex" --command="make" -w