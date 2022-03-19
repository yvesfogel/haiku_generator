import csv
import random

row_count = 0

with open('haikus575.csv', newline='') as csvfile:

	row_count = sum(1 for row in csvfile)  # fileObject is your csv.reader

with open('haikus575.csv', newline='') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	rows = list(csv_reader)
	print(rows[random.randint(0, row_count-1)])