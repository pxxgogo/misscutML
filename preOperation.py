from __future__ import print_function
import numpy.random as nr
import argparse
import os
import json
import datetime

MIN_FREQUENCY = 2
now = datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--output_dir', type=str, default='./filtered_%s/' % str(now))
parser.add_argument('--threshold', type=int, default=100)
parser.add_argument('--percent', type=int, default=10)
parser.add_argument('--only_filter', type=int, default=0)
args = parser.parse_args()

input_file = args.input
threhold = args.threshold
only_filter_flag = args.only_filter;

word_map = {}
line_num = 0
print("--------Start file reading-------")
lines = open(input_file, 'r').readlines()
print("--------Finish file reading-------")

print("--------Start file mapping-------")
for line in lines:
  content = line.strip('\n').decode("utf-8")
  char_list = content.split(" ")
  for word in char_list:
    if word == "":
      continue
    try:
      word_map[word] += 1
    except:
      word_map[word] = 1
  line_num += 1
  # if line_num % 10000 == 0:
    # print(".", end=" ")
print("--------Finish file mapping-------")

print("--------Start file filtering-------")
line_num = 0
for line in lines:
  content = line.strip('\n').decode("utf-8")
  char_list = content.split(" ")
  i = 0
  for word in char_list:
    if word == "":
      i += 1
      continue
    try:
      if word_map[word] < threhold:
        char_list[i] = "{{R}}"
        word_map.__delitem__(word)
    except:
      char_list[i] = "{{R}}"
    i += 1
  # if line_num % 10000 == 0:
  #   print(".", end=" ")
  lines[line_num] = " ".join(char_list).encode("utf-8")
  line_num += 1
if(only_filter_flag == 1):
  if not args.output_dir in os.listdir("./"):
    os.mkdir(args.output_dir)
  with open(os.path.join(args.output_dir, 'filter.txt'), 'w') as output:
    for line in lines:
      output.write(line + "\n")
  print("--------Finish file filtering-------")
else: 
  word_map["{{R}}"] = 0
  word_map_int = {}
  No = 0
  for word, num in word_map.items():
    word_map_int[word] = No
    No += 1
  word_map_int["{{vocab_size}}"] = len(word_map)
  if not args.output_dir in os.listdir("./"):
    os.mkdir(args.output_dir)
  with open(os.path.join(args.output_dir, 'words_map.json'), 'w') as f_word_map:
    f_word_map.write(json.dumps(word_map_int))
  print("--------Finish file filtering-------")

  print("--------Start file sampling-------")
  sample_lines = list(filter(lambda x: nr.random() * 100.0 <= args.percent, lines))
  train = []
  valid = []
  test = []
  for line in sample_lines:
    tmp = nr.random()
    if tmp <= 0.75:
        train.append(line)
    elif tmp <= 0.85:
        valid.append(line)
    else:
        test.append(line)
  lines = []
  line_num = 0
  for line in train:
    content = line.strip('\n').decode("utf-8")
    char_list = content.split(" ")
    for i in range(len(char_list)):
      if(char_list[i] == ""):
        continue
      char_list[i] = str(word_map_int[char_list[i]])
    train[line_num] = " ".join(char_list).encode("utf-8") + "\n"
    line_num += 1

  line_num = 0
  for line in test:
    content = line.strip('\n').decode("utf-8")
    char_list = content.split(" ")
    for i in range(len(char_list)):
      if(char_list[i] == ""):
        continue
      char_list[i] = str(word_map_int[char_list[i]])
    test[line_num] = " ".join(char_list).encode("utf-8") + "\n"
    line_num += 1

  line_num = 0
  for line in valid:
    content = line.strip('\n').decode("utf-8")
    char_list = content.split(" ")
    for i in range(len(char_list)):
      if(char_list[i] == ""):
        continue
      char_list[i] = str(word_map_int[char_list[i]])
    valid[line_num] = " ".join(char_list).encode("utf-8") + "\n"
    line_num += 1

  with open(os.path.join(args.output_dir, 'ptb.train.txt'), 'w') as ftrain:
    ftrain.writelines(train)
  with open(os.path.join(args.output_dir, 'ptb.valid.txt'), 'w') as fvalid:
    fvalid.writelines(valid)
  with open(os.path.join(args.output_dir, 'ptb.test.txt'), 'w') as ftest:
    ftest.writelines(test)
print ("Completed!")
