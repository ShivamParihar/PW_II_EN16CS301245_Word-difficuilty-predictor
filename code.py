import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


words_list = ['bɒs', 'griːn', 'ɪsˈkeɪp', 'frɛʃ', 'sɪgˈnɪfɪkənt', 'smaɪl', 'gɪft', 'ˈlæŋgwɪʤ', 'səkˈsɛs', 'ˈsʌdn', 'kəˈnɛkt', 'ˌɒpəˈreɪʃən', 'kəʊʧ', 'ˈviːɪkl', 'gæs', 'ˈæktə', 'weɪst', 'ˌkɒmpɪˈtɪʃən', 'ˌrɛprɪˈzɛnt', 'biː', 'tuː', 'juː', 'ænd', 'ɒv', 'ɪn', 'ðæt', 'hæv', 'ɪt', 'fɔː', 'ɒn', 'duː', 'ðɪs', 'gəʊ', 'wɪð', 'miː', 'kæn', 'maɪ', 'nəʊ', 'nɒt', 'wiː', 'æt', 'ˈkʌlʧə', 'piːs', 'ˈɛnɪweə', 'bɪˈjɒnd', 'ˈɔːdiəns', 'əˈfɛkt', 'sjuːt', 'ˈhʌni', 'gəʊld', 'ˈkwaɪət', 'ˈrɛgjʊlə', 'reɪn', 'rɪˈdjuːs', 'haʊˈɛvə', 'kəmˈpjuːtə', 'ˈsɛprɪt', 'sʌn', 'ˈɛnɪmi', 'ˈmɛʒə', 'æp', 'səkˈsiːd', 'dɪˈvaɪd', 'jɛl', 'fraɪ', 'dɪˈtɛkt', 'feɪz', 'ɪmˈbreɪs', 'pɔːt', 'praɪz', 'ʌnˈfɔːʧnɪt', 'ˈstreɪnʤə', 'ˈælaɪ', 'kraʊn', 'hʌg', 'ˈkætɪgəri', 'flɪp', 'ˈprɪznə', 'səsˈpɛnd', 'drɑːft', 'ˈmænə', 'əbˈzɜːv', 'smuːð', 'juːθ', 'pɒt', 'kræp', 'skæn', 'ˈfæntəsi', 'əˈpəʊnənt', 'ˈæpl', 'əˈlɑːm', 'kəˈmɪtmənt', 'hɪp', 'tæp', 'ˈkwɒlɪfaɪ', 'kaʊ', 'təʊn', 'ˈvɪzɪtə', 'ˈpænɪk', 'ɪnˌvaɪərənˈmɛntl', 'ˈgɒgl', 'ˈɒmɪnəs', 'ˈəʊvəsaɪzd', 'ˈsɪtjʊˌeɪt', 'ədˈvɜːsɪti', 'ˈkɒŋkwɛst', 'fɪg', 'vɒlˈkænɪk', 'ˈwɔːdn', 'klɒt', 'ɪnˈtrʌst', 'həˌluːsɪˈneɪʃən', 'ɪˈrɒtɪk', 'ˈmɪrɪəd', 'ˈbəʊldə', 'dɪsˈkrɛdɪt', 'dɪsˈrʌptɪv', 'ˈhjuːbrɪs', 'lɒnˈʤɛvɪti', 'ˈnɔːməlaɪz', 'ˈwɪti', 'ˌdɪsprəˈpɔːʃnɪt', 'ˈfɜːmɛnt', 'ˈmʌskjʊlə', 'ˈɔːfənɪʤ', 'swɒb', 'tju(ː)ˈɪʃən', 'ˈmʌdl', 'rɪˈkɔːdə', 'spæŋk', 'əˈnæləʤi', 'ˈdɪsɪplɪnəri', 'plɑːk', 'ˈsɛnsəri', 'speɪd', 'ˈvaʊʧə', 'ˈkɔːkəs', 'ˌʧɪmpənˈziː', 'ˈmɒdənaɪz', 'ˈaʊtskɜːts', 'ˈʃeɪmfʊl', 'slʌm', 'ˌsjuːpəˈfɪʃəl', 'ˈkɒntɛnt', 'ˈdaʊntɜːn', 'ˈmɪŋgl', 'mɔːg', 'aʊtˈdeɪtɪd', 'ˈmɒkəri', 'mʌnʧ', 'ˈpʌlmənəri', 'ˈrɛpətwɑː', 'ˈwaɪldkæt', 'ˈbʌtək', 'dɪsˈkriːt', 'ˈdɪzməl', 'kɛg', 'məˈruːn', 'ˈrɛdˌhɛd', 'ˈsɪlɪkən', 'trɒf', 'ˌʌpˈsteɪt', 'ɪˈgriːʤəs', 'ˈheɪzlnʌt', 'ɪmˈplɔː', 'ˈmʌmbl', 'ˈpɛdɪstl', 'plaɪ', 'ˈrætɪfaɪ', 'ˈtævən', 'kəʊˈɜːs', 'kənˈsʌmɪt', 'kənˈvɜːtə', 'ˈfɑːmlænd', 'ɪmˈpɑːt', 'ɪnˈdɪfrənt', 'rʌf', 'ʌpˈhiːvəl', 'ˈbɒsi', 'ˈsiːsˌfaɪə', 'ˌkɒntrəˈdɪktəri', 'ˌdiːhaɪˈdreɪtɪd', 'ˈnaɪtfɔːl', 'ˈpɛtrɪfaɪ', 'ˈʃɔːˌlaɪn', 'əˈtɛst', 'ˈbærɪstə', 'əˈlʌmnəs', 'kriːs', 'aɪˈrɒnɪkəli', 'ˈtɛlɪgræm', 'tɛˈstɒstəˌrəʊn', 'əˌfɪlɪˈeɪʃ(ə)n', 'kənˈsɛptjʊəl', 'ˈfæsɪt', 'ˈhɛmɪsfɪə', 'ˈnaɪlən', 'ˈkwɒrəl', 'ˈreɪdiənt', 'mʌt', 'tɛnʃ', 'kəˈlaɪdəskəʊp', 'tɪld', 'ˈtɪmərəs', 'ˈtɔːri', 'ˈtaɪpˌsɛtə', 'ʌnˈgʌvənəbl', 'ˈbæntəm', 'ˈblɑːni', 'ˈblɒkhaʊs', 'ˈbreɪsə', 'ˌkɑːtɪˈlæʤɪnəs', 'kəˈtælɪsɪs', 'ˈʧæstɪzmənt', 'ˌsɜːkəmləˈkjuːʃən', 'ˈklæmərəs', 'ˈkɒləkwi', 'ˈsɪnəzjʊə', 'ˈdaɪədɛm', 'drʌʤ', 'vɪˈrɑːgəʊ', 'ˈvɪsɪd', 'ˈwɒpɪti', 'ˈjæʃmæk', 'ˈjɔːkʃɪə', 'ˈzaɪənɪst', 'ˈɑːftəməʊst', 'ˈæŋgləʊ', 'ˈænəlɪst', 'ˌæpəʊˈzɪʃ(ə)n', 'əˈreɪbɪən', 'ˈɑːmlɪt', 'ˈæsəgaɪ', 'əˈtrɪbjʊtɪv', 'bɔːd', 'ˈbeɪgəm', 'ˈbiːzəm', 'brəˈtænjə', 'ˈsiːkəm', 'ˌkærɪˈbiːən', 'kəˈrɪljən', 'ˌkærɪˈætɪd', 'ˈkætəlɛpsi', 'kɛlt', 'ˈkɛltɪk', 'ʧʌf', 'ˈklævɪkɔːd', 'kɒzˈmɒgrəfi', 'ˈkrɒspiːs', 'kənˈkɒmɪtənt', 'grɪˈgeərɪəs', 'fru(ː)ˈgælɪti', 'ˈdɪkə', 'dɪˌmɪnjʊˈɛndəʊ', 'ˈdʌnlɪn', 'ˈɛnɜːveɪt', 'eɪˈtjuːd', 'ˈjʊərəʊˌvɪʒən', 'ɛksˈpɛktəreɪt', 'ˌfɪlɪˈpiːnəʊ', 'ˈfriːˌbuːtə', 'rɪˌkrɪmɪˈneɪʃən', 'ˌgæzɪˈtɪə', 'ˈgəʊtəm', 'ˈhɪərəˈpɒn', 'ɪnˈeɪliənəb(ə)l', 'aɪˈbɪərɪən', 'ɪˈmjʊə', 'ˌɪntə(ː)ˈlɪnɪə', 'ɪˈreɪnɪən', 'ɪzˈreɪli', 'ˈʒæbəʊ', 'ˈʤʊəri', 'ʤəʊˈkəʊs', 'ˈʤuːdɪkəʧə', 'ɪˈnɪmɪkəl', 'ˈkɪnzˌwʊmən', 'ˈlæpɪdəri', 'lɑːθ', 'ˌlɛbəˈniːz', 'ˈlɪzi', 'ləʊˈbiːljə', 'mɒˈreɪn', 'ˈneɪbɒb', 'dɪˈfʌŋkt', 'ˈswɪndlə', 'ˌzɛnəʊˈfəʊbɪə', 'fɪˈʤuːʃiəri', 'ˈlɔːdətəri', 'ɪnˈhɑːnsmənt', 'ˈspeɪʃəl', 'ˈmeɪhɛm', 'ˈkeɪvɪæt', 'əˈʤuːdɪkeɪt', 'ˈtræʤɪktəri', 'kəʊˈhɪərəns', 'ˈmaɪgrətəri', 'ˌmælɪˈdɪkʃən', 'ˈkɒrʊgeɪtɪd', 'ɪˈlɪptɪkəl']
intents = ['easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'easy', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'medium', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard', 'hard']

def load_dataset():
  intent = intents
  unique_intent = list(set(intent))
  sentences = words_list  
  return (intent, unique_intent, sentences)

intent, unique_intent, sentences = load_dataset()

def cleaning(sentences):
    words = []
    for x in list(sentences):
        characters_list = list(x)
        #print(characters_list)
        temp = []
        for index in range(0, len(characters_list)):
            y = characters_list[index]
            if y == 'ɒ' or y == 'ʌ' or y == 'æ' or y == 'ɛ' or y == 'p' or y == 'b' or y == 't' or y == 'd' or y == 'ʧ' or y == 'ʤ' or y == 'k' or y == 'g' or y == 'f' or y == 'v' or y == 'θ' or y == 'ð' or y == 's' or y == 'z' or y == 'ʃ' or y == 'ʒ' or y == 'm' or y == 'n' or y == 'ŋ' or y == 'h' or y == 'l' or y == 'r' or y == 'w' or y == 'j':
                temp.append(y)
            elif y == 'i' or y == 'u' or y == 'ɜ' or y == 'ɔ' or y == 'ɑ':
                if (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ː':
                    temp.append(y+'ː')
                elif (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ɪ':     
                    temp.append('ɔɪ')
                else:
                    temp.append('i')    
                index = index + 1    
            elif y == 'e':
                if characters_list[index + 1] == 'ɪ':
                    temp.append('eɪ')
                else:
                    temp.append('eə')
                index = index + 1
            elif y == 'ə':
                if (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ʊ':
                    index = index + 1
                    temp.append('əʊ')
                else:
                    temp.append('ə')   
            elif y == 'a':
                if characters_list[index + 1] == 'ɪ':
                    temp.append('aɪ')
                else:
                    temp.append('aʊ')
                index = index + 1
        words.append(temp)        
    return words

cleaned_words = cleaning(sentences)

word2intquestions = dic

vocab_size = len(word2intquestions)
max_length = len(max(cleaned_words, key = len))

def encoding_doc(word2int, cleaned_words):
    result = []
    for sentence in cleaned_words:
        temp = []
        for word in sentence:
            temp.append(word2int[word])
        result.append(temp)
    return result    

encoded_doc = encoding_doc(word2intquestions, cleaned_words)

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

padded_doc = padding_doc(encoded_doc, max_length)

#tokenizer with filter changed
word2intanswers = {}
i = 1

for word in unique_intent:
    word2intanswers[word] = i
    i += 1

encoded_output = []
for each in intent:
    encoded_output.append(word2intanswers[each])

encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

def one_hot(encode):
  o = OneHotEncoder(sparse = False, categories='auto')
  return(o.fit_transform(encode))

output_one_hot = one_hot(encoded_output)

############ Creating Model ############ 
from sklearn.model_selection import train_test_split
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.22)

def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(len(unique_intent), activation = "softmax"))  
  return model

model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()

filename = 'word_difficulty_model2.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

hist = model.fit(train_X, train_Y, epochs = 10, batch_size = 4, validation_data = (val_X, val_Y), callbacks = [checkpoint])


































model = load_model("word_difficulty_model1.h5")

import requests
from bs4 import BeautifulSoup

url = 'https://tophonetics.com/'
data = {'text_to_transcribe': '',
        'submit': 'Show transcription',
        'output_dialect': 'br',
        'output_style': 'only_tr',
        'preBracket':'[',
        'postBracket':']',
        'speech_support': 1
        }

def predictions(text):
    data['text_to_transcribe'] = text
    res = requests.get(url)
    res = requests.post(url, data=data)
    soup = BeautifulSoup(res.content, 'html5lib')
    word = str(soup.find('span', attrs={'class':'transcribed_word'}).text)

    characters_list = list(word)
    temp = []
    for index in range(0, len(characters_list)):
        y = characters_list[index]
        if y == 'ɒ' or y == 'ʌ' or y == 'æ' or y == 'ɛ' or y == 'p' or y == 'b' or y == 't' or y == 'd' or y == 'ʧ' or y == 'ʤ' or y == 'k' or y == 'g' or y == 'f' or y == 'v' or y == 'θ' or y == 'ð' or y == 's' or y == 'z' or y == 'ʃ' or y == 'ʒ' or y == 'm' or y == 'n' or y == 'ŋ' or y == 'h' or y == 'l' or y == 'r' or y == 'w' or y == 'j':
            temp.append(y)
        elif y == 'i' or y == 'u' or y == 'ɜ' or y == 'ɔ' or y == 'ɑ':
            if (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ː':
                temp.append(y+'ː')
            elif (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ɪ':     
                temp.append('ɔɪ')
            else:
                temp.append('i')    
            index = index + 1    
        elif y == 'ɪ':
            if (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ə':
                index = index + 1
                temp.append('ɪə')
            else:
                temp.append('ɪ')
        elif y == 'ʊ':
            if (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ə':
                index = index + 1
                temp.append('ʊə')
            else:
                temp.append('ʊ')
        elif y == 'e':
            if characters_list[index + 1] == 'ɪ':
                temp.append('eɪ')
            else:
                temp.append('eə')
            index = index + 1
        elif y == 'ə':
            if (index + 1)  < len(characters_list) and characters_list[index + 1] == 'ʊ':
                index = index + 1
                temp.append('əʊ')
            else:
                temp.append('ə')   
        elif y == 'a':
            if characters_list[index + 1] == 'ɪ':
                temp.append('aɪ')
            else:
                temp.append('aʊ')
            index = index + 1
    test_ls = []
    for word in temp:
        test_ls.append(word2intquestions[word])
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = padding_doc(test_ls, max_length)
    print(text)
    pred = model.predict_proba(x)
    return pred


def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
  print("%s has confidence = %s" % (classes[0], (predictions[0])))

text = "fresh"
pred = predictions(text)
get_final_output(pred, unique_intent)
