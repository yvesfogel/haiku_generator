import csv


# open the file in the write mode
f = open('haikus575.csv', 'w')

writer = csv.writer(f)


with open('haiku_all.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	for row in spamreader:

		try:
			num1 = int(row[-3])
			num2 = int(row[-2])
			num3 = int(row[-1])

			line1 = row[0].strip()
			line2 = row[1].strip()
			line3 = row[2].strip()

			print(row)
			print(line1)
			print(line2)
			print(line3)

			if (num1==5 and num2==7 and num3==5):

				new_haiku = [line1,line2,line3]


				writer.writerow(new_haiku)
		except:
			pass


f.close()