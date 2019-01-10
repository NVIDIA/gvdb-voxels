Run bib.py in this directory to generate 2 HTML files:
- cudpp_refs.html, a date-sorted list of references that use CUDPP
  (every file in cudpp.bib)
- cudpp_refs_bib.html, the BibTeX for each of those refs

Input files are:
- cudpp.bib. Add new bibtex entries here. 
- cudpp.bst. BibTeX style file for how the resulting HTML will look.
  Bug JDO if you don't like the format. 

bib.py calls two external programs, both from the bibtex2html package.

http://www.lri.fr/~filliatr/bibtex2html/

After you regenerate the two html files, check them into the
repository (also cudpp.bib) so they can be included the next time the
documentation is rebuilt. 
