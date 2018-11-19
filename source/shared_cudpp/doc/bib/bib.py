#!/usr/bin/env python

import sys, os, re

# Generate all pubs from cudpp.bib; put refs into ref.txt
# This just generates a list of the cite keys from cudpp.bib into ref.txt.
os.system("bib2bib -oc ref.txt cudpp.bib")

# sort by reverse-date; don't generate keys; use cudpp.bst as bib style file
# writes into cudpp_refs.html and cudpp_refs_bib.html
os.putenv("openout_any", "r")
os.system("bibtex2html -d -r -dl -nokeys -html-entities --no-footer --no-keywords -citefile ref.txt -s cudpp -nodoc -o cudpp_refs cudpp.bib")
html_file = open('cudpp_refs.html')
html = html_file.read()
html_file.close()

# if we want to munge the resulting text, do it here
# right now this script is only 2 calls and a file remove though

# write the file back
html_file = open('cudpp_refs.html', 'w')
print >> html_file, html
html_file.close()

# clean up temp files
os.remove("ref.txt")
