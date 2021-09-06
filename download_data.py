# coding: utf-8
# py3
# install newspaper
# import download_data
# download_data.read_url('http://www.baomoi.com/')

import newspaper, codecs
from newspaper import Article
from newspaper import fulltext


def read_article(url):
    art = Article(url, language = 'vi')
    #print(fulltext(url, language='vi'))
    art.download()
    art.parse()
    return art.title + '.\n' +art.text + '\n'


def read_url(url):
    paper = newspaper.build(url)
    i=0
    names = url.split('//')
    link = codecs.open(names[1].split('.')[1]+'_link.txt','w')
    for article in paper.articles:
        print ("Process to "+str(article.url))
        link.write(str(article.url))
        fout = codecs.open('/home/tuong/Downloads/Dataset/'+names[1].split('.')[1]+str(i)+'.txt','w')
        fout.write(read_article(article.url))
        fout.close()
        i += 1
    link.close()

