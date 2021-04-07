
# Begin
ROOT_FOLDER = "/home/speed-marius/Speed/Humans/humans/data/obama/subtitles/"
OUTPUT_FOLDER = "/home/speed-marius/Speed/Humans/humans/data/obama/pp_subtitles/"
with open('/home/speed-marius/Speed/Humans/humans/data/obama/subtitles_list.txt') as input_file_list:
    file_list = input_file_list.readlines()

for input_file in file_list:
    input_file = input_file.replace('\n', '')
    print(input_file)
    with open(ROOT_FOLDER + input_file) as f:
        lines = f.readlines()
    del lines[0:4]
    output_string = ''
    to_be_removed = []
    for line in lines:
        if line[0].isdigit() or line == 'The President:\n' or line == 'President Obama:\n' or line == '\n':
            to_be_removed.append(line)

    for item in to_be_removed:
        lines.remove(item)

    new_list = []
    for line in lines:
        if 'The President:' in line:
            new_line = line.replace('The President:', '').strip()
            new_list.append(new_line)
        elif 'President Obama:' in line:
            new_line = line.replace('President Obama:', '').strip()
            new_list.append(new_line)
        else:
            new_list.append(line)

    for line in new_list:
        output_string += line.replace('\n', ' ')

    with open(OUTPUT_FOLDER + "pp_" + input_file, 'w') as out:
        out.write(output_string)
        out.close()
