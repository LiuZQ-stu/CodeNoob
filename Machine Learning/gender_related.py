import gensim
import sys
sys.path.append('/data/zhaoqi/main_word/seqlbl/algorithm')
import jieba
import pandas as pd
import os
from collections import defaultdict
#from mt_sequence_labeling import do_predict

class Model:
    def __init__(self):

        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../word_emb/10it_1p_80d_goods', binary=True)
        self.maleKeyWordsDict = set()
        self.femaleKeyWordsDict = set()
        self.generalKeyWordsDict = set()
        self.explicitDict =  None
        self.threshold = 0.2
        self.NER_path = '../data/prod_res'

    def match(self, text):

        text = jieba.lcut(text)
        for word in text:
            if word in self.explicitDict or '男' in word or '女' in word:
                return True
        return False

    def predict_w2v(self, texts):

        with open('result_by_w2v','w') as f:

            unmatched = []
            for text in texts:
                text = str(text)
                if self.match(text):
                    #print('[{}] 显性性别相关'.format(text))
                    #f.write('[{}] 显性性别相关'.format(text) + '\n')
                    pass
                else:
                    unmatched.append(text)
            if os.path.exists(self.NER_path):
                product = [line.strip().split('\t')[1] for line in open(self.NER_path).readlines()]
                unmatched = [line.strip().split('\t')[0] for line in open(self.NER_path).readlines()]

            else:
                with open(self.NER_path, 'w') as NER:
                    product = []
                    for i in range(len(unmatched)//1000-2):
                        result = do_predict.inference_wrapper(unmatched[i * 1000:(i + 1) * 1000])
                        for res, unm in zip(result, unmatched[i * 1000:(i + 1) * 1000]):
                            if res['PROD'] != []:
                                product.append(res['PROD'][0])
                                NER.write(unm + '\t' + res['PROD'][0] + '\n')
                            else:
                                product.append(unm)
                                NER.write(unm + '\t' + unm + '\n')
                print('prepared')

            print(len(product), len(unmatched))
            for res, text in zip(product, unmatched):
                if self.w2v_model.wv.__contains__(res):

                    sim1 = abs(self.w2v_model.similarity(res, '男'))
                    sim2 = abs(self.w2v_model.similarity(res, '女'))
                    score = (2 * sim1 * sim2 / (sim1 + sim2)) - abs(sim1 - sim2)
                else:
                    score = 0

                if score >= self.threshold:

                    #print('[{}] 隐性性别相关'.format(text))
                    f.write('{}'.format(text) + '\t' + str(score) + '\n')
                    #f.write('[{}] 隐性性别相关'.format(text) + '\n')
                else:
                    pass
                    #print('[{}] 性别不相关'.format(text))
                    #f.write('[{}] 性别不相关'.format(text) + '\n')

    def predict_by_cat(self, texts, cats):

        with open('result_by_cat', 'w') as f:
            for text, cat in zip(texts, cats):
                if self.match(str(text)):
                    pass
                    #f.write('[{}] 显性性别相关'.format(text) + '\n')
                elif self.match(str(cat)):
                    f.write('{}'.format(text) + '\n')
                    #f.write('[{}] 隐性性别相关'.format(text) + '\n')
                else:
                    pass
                    #f.write('[{}] 性别不相关'.format(text) + '\n')

    def predict_by_overlap(self, texts, prods):

        prod_dict = dict()
        lines = open('data/click_sort_0.5').readlines()
        for line in lines:
            prod_dict[line.strip().split('\t')[0]] = line.strip().split('\t')[4]
        with open('result_by_overlap','w') as f:
            for text, prod in zip(texts, prods):
                if self.match(str(text)):
                    pass
                    #f.write('[{}] 显性性别相关'.format(text) + '\n')
                elif prod in prod_dict:
                    f.write('{}'.format(text) + '\t' + prod_dict[prod] + '\n')
                    #f.write('[{}] 隐性性别相关'.format(text) + '\n')
                else:
                    pass
                    #f.write('[{}] 性别不相关'.format(text) + '\n')

    def test(self):

        self.overlap_res = '/data/zhaoqi/sex_pred/result/result_by_overlap'
        #self.cat_res = '/data/zhaoqi/sex_pred/result/result_by_cat'
        self.w2v_res = '/data/zhaoqi/sex_pred/result/result_by_w2v'
        self.cover_fre = '/data/zhaoqi/sex_pred/result/result_cover_goods'

        overlap = open(self.overlap_res).readlines()
        #cat = open(self.cat_res).readlines()
        w2v = open(self.w2v_res).readlines()
        fre = open(self.cover_fre).readlines()
        raw_data = open(self.NER_path).readlines()

        overlap_dict = dict()
        w2v_dict = dict()
        fre_dict = dict()
        for line in overlap:
            query, score = line.strip().split('\t')
            overlap_dict[query] = score
        for line in w2v:
            query, score = line.strip().split('\t')
            w2v_dict[query] = score
        for line in fre:
            try:
                query, score = line.strip().split('\t')
            except:
                continue
            fre_dict[query] = score
        result = defaultdict(float)

#        with open('final_res', 'w') as f:
#            for line in raw_data:
#                query = line.strip().split('\t')[0]
#                cut = jieba.lcut(query)
#                for word in cut:
#                    score = 0.8 - float(overlap_dict.get(word, 0.6)) + float(w2v_dict.get(word, 0.1)) / 0.67
#                    if word in fre_dict:
#                        score += 0.7
#                    result[query] += score
#                result[query] /= len(cut)
#                f.write(query + '\t' + str(result[query]) + '\n')

        prods = defaultdict(int)
        with open('prod_rec','w') as f:
            for line in raw_data:
                prod = line.strip().split('\t')[1]
                if prod in prods:
                    continue
                if prod in overlap_dict:
                    prods[prod] += 1
                if prod in w2v_dict:
                    prods[prod] += 1
                if prod in fre_dict:
                    prods[prod] += 1
                if prods[prod] >= 3:
                    f.write(prod + '\n')
                prods[prod] = 0





            '''#rank = sorted(result.items(), key=lambda score: score[1], reverse=True)
            for (query, score) in rank:
                f.write(query + '\t' + str(score) + '\n')'''




    def dictInit(self):
        self.maleKeyWordsDict.add("男士")
        self.maleKeyWordsDict.add("男性")
        self.maleKeyWordsDict.add("男款")
        self.maleKeyWordsDict.add("男式")
        self.maleKeyWordsDict.add("男人")
        self.maleKeyWordsDict.add("男孩")
        self.maleKeyWordsDict.add("男生")
        self.maleKeyWordsDict.add("男神")
        self.maleKeyWordsDict.add("叔叔")
        self.maleKeyWordsDict.add("爸爸")
        self.maleKeyWordsDict.add("爷爷")
        self.maleKeyWordsDict.add("大叔")
        self.maleKeyWordsDict.add("大爷")
        self.maleKeyWordsDict.add("大哥")
        self.maleKeyWordsDict.add("男性")
        self.maleKeyWordsDict.add("男装")
        self.maleKeyWordsDict.add("绅士")
        self.maleKeyWordsDict.add("少男")
        self.maleKeyWordsDict.add("男子")
        self.maleKeyWordsDict.add("男")

        self.femaleKeyWordsDict.add("女士")
        self.femaleKeyWordsDict.add("淑女")
        self.femaleKeyWordsDict.add("女款")
        self.femaleKeyWordsDict.add("女式")
        self.femaleKeyWordsDict.add("女人")
        self.femaleKeyWordsDict.add("妈妈")
        self.femaleKeyWordsDict.add("大妈")
        self.femaleKeyWordsDict.add("大婶")
        self.femaleKeyWordsDict.add("大姐")
        self.femaleKeyWordsDict.add("女神")
        self.femaleKeyWordsDict.add("女孩")
        self.femaleKeyWordsDict.add("女性")
        self.femaleKeyWordsDict.add("女装")
        self.femaleKeyWordsDict.add("女鞋")
        self.femaleKeyWordsDict.add("女裤")
        self.femaleKeyWordsDict.add("妹子")
        self.femaleKeyWordsDict.add("女生")
        self.femaleKeyWordsDict.add("女子")
        self.femaleKeyWordsDict.add("妹妹")
        self.femaleKeyWordsDict.add("美眉")
        self.femaleKeyWordsDict.add("妇女")
        self.femaleKeyWordsDict.add("孕妇")
        self.femaleKeyWordsDict.add("姐妹")
        self.femaleKeyWordsDict.add("老太太")
        self.femaleKeyWordsDict.add("少女")
        self.femaleKeyWordsDict.add("妈妈装")
        self.femaleKeyWordsDict.add("女")
        self.femaleKeyWordsDict.add("女友")

        self.generalKeyWordsDict.add("男女")
        self.generalKeyWordsDict.add("情侣")

        self.explicitDict = self.maleKeyWordsDict | self.femaleKeyWordsDict | self.generalKeyWordsDict


if __name__ == '__main__':

    model = Model()
    model.dictInit()
    '''
    texts = pd.read_csv('data/all_query.csv')['query'].to_list()
    model.predict_w2v(texts)

    texts = pd.read_csv('data/all_query.csv')['query'].to_list()
    cats = pd.read_csv('data/all_query.csv')['top_1_name'].to_list()
    model.predict_by_cat(texts, cats)

    texts = []
    prods = []
    lines = open('data/prod_res').readlines()
    for line in lines:
        line = line.strip().split('\t')
        texts.append(line[0])
        prods.append(line[1])
    model.predict_by_overlap(texts, prods)'''

    with open('/data/zhaoqi/nlp_query_tag/query_gender_pred/data/maleKeyWordsDict','w') as f:
        for word in model.maleKeyWordsDict:
            f.write(word + '\n')
    with open('/data/zhaoqi/nlp_query_tag/query_gender_pred/data/femaleKeyWordsDict', 'w') as f:
        for word in model.femaleKeyWordsDict:
            f.write(word + '\n')
    with open('/data/zhaoqi/nlp_query_tag/query_gender_pred/data/generalKeyWordsDict', 'w') as f:
        for word in model.generalKeyWordsDict:
            f.write(word + '\n')
