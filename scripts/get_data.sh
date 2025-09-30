#!/bin/bash
set -e

# URL Dropbox (forcer le téléchargement avec dl=1)
URL="https://www.dropbox.com/scl/fi/1otb6qlx9mqglk5a5os1e/data.zip?rlkey=saxd9b1vaw87rgsh1d9aooud4&st=5924gomj&dl=1"

curl -L "$URL" -o data.zip
unzip -o data.zip -d .
rm data.zip
