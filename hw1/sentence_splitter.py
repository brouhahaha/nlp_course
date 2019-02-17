import re
from pprint import pprint
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

def get_gold_st(filename):      #открываем текст с готовым разделением
    with open (filename, 'r', encoding='utf8') as text:
        content = text.read()
    return content.split('\n')

def get_source_text(filename): #открываем кусок, неподелённый на предложения
    with open (filename, 'r', encoding='utf8') as text:
        content = text.read()
    new_text = re.sub ('\n', ' ', content)
    return new_text

def tokenize_by_reg(text):   #делим текст по регулярке
    return re.split('(?<=[.!?]) (?=[A-ЯЁA-Z])', text)  

def how_accurate(gold, my_split):    #простая метрика оценки качества - через пересечения множеств
    return len(set(gold) & set(my_split)) / len(set(gold) | set(my_split))

def how_accurate1(sent, gold):  #метрика mipt токенизатора для русского
    tp = 0
    fp = 0
    fn = 0

    for sent in gold:
        if len(tokenizer.tokenize(sent)) == 1:
            tp += 1
        else:
            fp += 1

    for i in range(len(gold)-1):
        sent1, sent2 = gold[i], gold[i+1]
        sent = ' '.join([sent1, sent2])
        if len(tokenizer.tokenize(sent)) == 2:
            tp += 1
        else:
            fn += 1

    results = {'precision' : tp/(tp+fp), 'recall' : tp/(tp+fn), 'f1' : 2*((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))}
    return results
    
def get_full_text(filename):           #открыть полный текст книги
    with open (filename, 'r', encoding='cp1251') as text:
        content = text.read()
    return content
    
def main():
    gold = get_gold_st('gold_standard.txt')          #открыли "золотой стандарт"
    source_text = get_source_text('source_text.txt') #открыли кусок без разделения
    my_split = tokenize_by_reg(source_text)         #разделили текст по регулярному выр-ю
    full_text = get_full_text('full_text.txt')      #открыли полный текст
    trainer.train(full_text)                        #тренируем модель токенайзера на полном тексте
    new_split = tokenizer.tokenize(source_text)     #поделили текст на предложения по модели
    metrics = how_accurate1(new_split, gold)        #измерили качество по mipt-метрике
    with open ('evaluation.txt', 'w', encoding='utf8') as text:             #сохраняем результаты оценки
        text.write('Оценка качества деления по регулярному выражению (пересечение множеств)\n')
        text.write(str(how_accurate(gold, my_split))+'\n')            
        text.write('Оценка качества деления моделью(пересечение множеств)\n')
        text.write(str(how_accurate(gold, new_split))+'\n')
        text.write('Оценка качества деления моделью по mipt-метрике:\n')
        for i in metrics:
            text.write(str(i)+'\t'+str(metrics[i])+'\n')
    print('Готово, результаты в файле Evaluation')

trainer = PunktTrainer()        #задаем параметры обучения модели
trainer.INCLUDE_ALL_COLLOCS = True
tokenizer = PunktSentenceTokenizer(trainer.get_params())

if __name__ == '__main__':
    main()

