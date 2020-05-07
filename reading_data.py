data = open('Raw_Data/first_second_part.txt', 'r',encoding='utf8').read()

data_new = open('Raw_Data/new_data.txt', 'w',encoding='utf8')
#print(data)
sentence=(data.split('.'))
len_sentence=len(sentence)
print(len_sentence)
print(type(sentence))
for item in enumerate(sentence):
    data_new.write(str(item)+'\n')

