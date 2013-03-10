def process_raw_data():
    data = []

    inFile = open('toronto.txt')
    for line in inFile:
        data.append(line.split('\t')[3:])
    inFile.close()

    outFile = open('data.txt', 'wb')
    for point in data:
        outFile.write(point[0] + ' ' + point[1])
    outFile.close()

def reformat_data():
    data = ['', '']
    inFile = open('data.txt')
    for line in inFile:
        point = line.split(',')
        data[0] += point[0].strip() + ' '
        data[1] += point[1].strip() + ' '
    inFile.close()
    outFile = open('toronto_data.txt', 'wb')
    outFile.write(data[0] + '\n' + data[1])
    outFile.close()

reformat_data()