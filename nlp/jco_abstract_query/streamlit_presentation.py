# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 13:21:11 2021

@author: Olivier
"""
import streamlit as st 
import streamlit.components as stc

# Utils
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import pandas as pd 

#Import requetes
import requests
from bs4 import BeautifulSoup
import re

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

http_prefixe='https://www.nature.com'

import altair as alt

### 
# This function is made for scraping the web
#
###
@st.cache #Memorize the dataset
def nature_request(nature_url,key_word,all_pages='all_pages') :

    # Get the html text of the request
    html_text = requests.get(nature_url).text
    
    # Parse the text
    soup = BeautifulSoup(html_text, 'html.parser')
    
    # Attributs de la requete
    attrs = {
        'href': re.compile(r'/articles/s')
    }
    articles_links = [ link.get('href') for link in soup.find_all('a', attrs=attrs)]
    
    #Lecture des autres pages de liens si all_pages est à True
    if all_pages :
        page=2
        nb_links = 1
        while nb_links != 0 :
            url = nature_url + '&page=' + str(page) 
            # Lecture de la page
            soup_tmp = BeautifulSoup(requests.get(url).text, 'html.parser')
            articles_links_tmp = [ link.get('href') for link in soup_tmp.find_all('a', attrs=attrs)]
        
            nb_links = len(articles_links_tmp)
            if nb_links > 0 :
                articles_links = [*articles_links, *articles_links_tmp]
            page += 1
    
    # Lecture des abstracts
    len_max=300
    separateur='##'
    
    
    #Titres à supprimer 
    l_suppress_words = ['Background', 'Methods', 'Results', 'Conclusions', 'Background/objectives', 'Subjects/methods', 'Conclusion', 'Purpose', 'Objectives', 'Objective', 'Aims', 'Study design']
    l_suppress_fields = [ w + separateur for w in l_suppress_words]
    
    #Creation d'un dataframe destiné à contenir les informations recupérées sur les différents articles
    abstract_df = pd.DataFrame(columns=['date', 'type', 'journal', 'title', 'authors', 'abstracts', 'link'])
    

    
    for i in range(len(articles_links)) :# enumerate(tqdm(articles_links, total=len(articles_links))) :
      
      name = articles_links[i]
      
      link = http_prefixe + name
      #print(link)
      html_2_text=requests.get(link).text
      #print(html_2_text)
      soup_2 = BeautifulSoup(html_2_text, 'html.parser')
    
      # Recuperation date
      date_field = soup_2.find('time').get('datetime')
      #print('date_field : ', date_field)
    
      # Recuperation titre
      title_field = soup_2.find('h1', {'class':"c-article-title"}).get_text()
    
    
      # Recuperation auteurs
      l_authors = []
      for l in soup_2.find_all('a', attrs={'data-test':'author-name'}):
        l_authors.append(l.get_text())
      authors_field = ','.join(l_authors)
    
      # Recuperation abstract
      if soup_2.find('div', {'class':"article__teaser"}) != None :
        abstract_field = soup_2.find('div', {'class':"article__teaser"}).get_text(separator=separateur)
      else :
        abstract_field = soup_2.find('div', {'class':"c-article-section__content"}).get_text(separator=separateur)
      for s in l_suppress_fields :
        abstract_field = abstract_field.replace(s, '')
        
      abstract_field = abstract_field.replace(separateur, ' ') 
    
      #print(abstract_field)
      if len(abstract_field.split()) > len_max :
         continue
    
      # Recuperation type
      type_field=" ".join(soup_2.find('li', {'id':'breadcrumb2'}).get_text().split())
    
      # Recuperation journal
      journal_field=" ".join(soup_2.find('li', {'id':'breadcrumb1'}).get_text().split())
    
    
      #Add to dataframe
      field_series = pd.Series([date_field, type_field, journal_field, title_field, authors_field, abstract_field, link], index = abstract_df.columns)
      abstract_df = abstract_df.append(field_series, ignore_index=True)
    
    
    abstract_df['abstracts'] = abstract_df.abstracts.apply(lambda x : x.rstrip())
    abstract_df = abstract_df.replace({'abstracts': r'\[[\w\W]*\]'}, {'abstracts': ''}, regex=True)
    abstract_df = abstract_df.replace({'abstracts': r'\([\w\W]*\)'}, {'abstracts': ''}, regex=True)

   
    return abstract_df

def draw_bar_plot(df, l_fields) :
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    
    ax1.hist(df[l_fields[0]], bins=20)
    ax1.set_title(l_fields[0])
    ax2.hist(df[l_fields[1]], bins=20)
    ax2.set_title(l_fields[1])
    ax3.hist(df[l_fields[2]], bins=20)
    ax3.set_title(l_fields[2])
    
    st.pyplot(fig)
    
def convert_logit_binary(x) :
    if x > 0 :
        return 1
    else :
        return 0
    
def main():
    
    # Remplissage de la sidebar
    st.sidebar.image('./data/sentiment-analysis.jpg')
    

    
    # Choix de la page à lire 
    pages = ['HOME', 'SEARCH ARTICLES', 'ANALYSE ARTICLE SENTIMENT', 'IMDB REVIEWS', 'COVID IMPACT']
    choice = st.sidebar.selectbox("Pages",pages)
    
    l_types = []
    l_journal = []
    abstract_df = pd.DataFrame()
    sentence_uniq_df = pd.DataFrame()
    
    first_p = "first_page"
    all_p = "all_pages"
    quick_search = all_p
    search_word=''
    
    
    if choice == 'HOME' :
        st.title('SENTIMENT ANALYSIS')
        st.write('The main goal of this study is to work on nature abstracts and to be able to classify the abstract whether they are writen in a positive or negative way.')
        st.subheader('ARTICLES SEARCH ')
        st.write('The search stage will scrape all the abstracts of the site https://nature.com in the range of date specified. Data will be stored in a csv file meant to be used during the analysis. ' +
                 'Only the abstracts with a number of less than 300 words will be taken into account.')
        st.subheader('IMDB REVIEWS')
        st.write('This is the calibration step. I ve spent time working on models, beginning with the word2vec, then deep learning basis cnn model, finishing with the most sharp huggingface model based on BART. I ve used a labelled IMDB review as metric. ' +
                 'This stage presents the obtained results with the 2 huggingface models : ' +
                 'siebert/sentiment-roberta-large-english and distilbert-base-uncased-finetuned-sst-2-english.')
        st.subheader('ANALYSE ARTICLE SENTIMENT')
        st.write('The analysis stage will show the results obtained with the most efficient text classification model ' +
                  'based on BERT using the huggingFace transformers: siebert/sentiment-roberta-large-english. ' +
                  'This model is a fine-tuned checkpoint of RoBERTa-large (Liu et al. 2019). It enables reliable binary sentiment analysis for various types of English-language text. For each instance, it predicts either positive (1) or negative (0) sentiment. The model was fine-tuned and evaluated on 15 data sets from diverse text sources to enhance generalization across different types of texts (reviews, tweets, etc.). Consequently, it outperforms models trained on only one type of text (e.g., movie reviews from the popular SST-2 benchmark) when used on new data as shown below. '
                  'Furthermore, the model is applied on the whole abstract, on a selected group of sentences in which the searched word appears and a summary done by a HuggingFace '+
                  'transformers based on model facebook/bart-large-cnn.')
        st.subheader('IMPACT OF COVID')
        st.write('During covid waves, non-emergency operations requiring potential access to a resuscitation bed have been postponed. '+
                 'The purpose of this page is to present the method more than the results.')

    elif choice == 'SEARCH ARTICLES' :
        # Reduction de la recherche à la première page : MODE RAPIDE
        st.title('SCIENTIFIC ABSTRACT SENTIMENT ANALYSIS')
        start_0=time.time()
        quich_search = st.selectbox("how many pages selected for the search", [all_p, first_p])
    
        end_0= time.time()
        #st.write('duree : {}'.format(end_0 - start_0))
        
        # Intervalle de dates à étudier
        l_date = [str(2010 + i) for i in range(0,12)] 
        start_date, end_date = st.select_slider(
            "time range : ", 
            options=l_date,
            value=(l_date[0], l_date[-1]))
        
        # Mot recherché
        search_word = st.text_input("Searched word :")
        
        # Bouton déclencheur
        search_button = st.button("Search the whole Nature")


    
        if search_button :
             nature_query = "{}/search?q={}&order=relevance&date_range={}-{}".format(http_prefixe, search_word, start_date, end_date)
             st.text(nature_query)
             start_0= time.time()
             abstract_df = nature_request(nature_query, search_word, quick_search)
        
             # Chargement des résumés
             st.write("{} abstracts are loaded.".format(abstract_df.shape[0]))
             st.write('duree : ' + str(time.time() - start_0))
        
             st.dataframe(abstract_df[['title', 'link']], width=2000)
             
             # Ecriture des données dans des csv
             abstract_df.to_csv('./input/abstract_df.csv')
             
    elif choice == 'IMDB REVIEWS' :
        st.title('SCIENTIFIC ABSTRACT SENTIMENT ANALYSIS')    
        
        st.write('At this stage, I was considering and making comparisons between the two most performant models, BERT and DISTILBERT. ' +
                 'following litterature, it appears that Bart is the most efficient. ' +
                 'To be able to compare the accuracy of the two models, I used an IMDB review dataset to make metrics.')
        #Lecture des données dans des csv
        file_path_imdb = './input/imdb_sample_short_sentences_df_25082021.csv'
        imdb_df = pd.read_csv(file_path_imdb)
        #Lecture des colonnes
        #st.write("Colonnes du dataframe imdb reviews : ")
        #st.write(imdb_df.columns)
        imdb_df['siebert_binary'] = imdb_df['siebert_label'].map(convert_logit_binary)
        imdb_df['distilbert_binary'] = imdb_df['distilbert_label'].map(convert_logit_binary)
        
        #Altair
        st.subheader('SCORES comparison of the 2 models tested on the IMBD REVIEW dataset.')
        altair_1 = alt.Chart(imdb_df).mark_point().encode(x='siebert_label', y='distilbert_label', color='labels:N', shape = 'labels', tooltip=['sentence'])
        st.altair_chart(altair_1, use_container_width=True)
        
        #sun ray graph
        st.subheader('MODELS SCORES DISTRIBUTION.')
        fig_sun = px.sunburst(imdb_df, path=['true_label','siebert_binary','distilbert_binary'], 
                              color_discrete_map={'(?)':'black', 0:'red', 1:'darkblue'}, 
                              title='Center circle : IMDB labels, second : BERT labels, third : DISTIBERT LABELS', 
                              width=600, height=600)
        
        st.plotly_chart(fig_sun, use_container_width=True)
        
    elif choice == 'ANALYSE ARTICLE SENTIMENT' :
        
        #Lecture des données dans des csv
        file_path = './input/abstract_score_df_avastin_2015_2021_24082021.csv'
        abstract_df = pd.read_csv(file_path)
        
        # Mot cherché
        search_word=file_path.split('_')[3]
        st.write('Searched word :')
        st.write(search_word)
        
        #Type articles
        l_types = abstract_df['type'].unique()

        choice_type = st.multiselect("Type",l_types, default=l_types)
        
        #Type journaux
        l_journaux = abstract_df['journal'].unique()
        choice_journal = st.multiselect("Journal",l_journaux, default=l_journaux)
        
        # Traces
        if st.checkbox('Show columns'):
            st.write(abstract_df.columns)
        #st.write(choice_journal)
        
        # Reduction du dataset de travail
        data_df = abstract_df[(abstract_df['type'].isin(choice_type)) & (abstract_df['journal'].isin(choice_journal))]
        
        if st.checkbox('Show number of rows'):
            st.write('Number of rows in the dataframe {}'.format(data_df.shape[0]))

        #Représentation de la répartition des différentes valeurs
        st.subheader('Score distribution (logits) / siebert model : ')
        
        if st.checkbox('Show logits distribution / bar : '):
            draw_bar_plot(data_df, ['abstract_score_siebert', 'group_score_siebert', 'summary_bart_score_siebert' ])

        
        # generate some random test data
        all_data = [data_df['summary_bart_score_siebert'].to_numpy(), 
                    data_df['abstract_score_siebert'].to_numpy()]
        
        
        if st.checkbox('Show logits distribution / violin and box : '):
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            # plot violin plot
            axs[0].violinplot(all_data,
                              showmeans=False,
                              showmedians=True)
            axs[0].set_title('Violin plot')
            
            # plot box plot
            axs[1].boxplot(all_data)
            axs[1].set_title('Box plot')
            
            # adding horizontal grid lines
            for ax in axs:
                ax.yaxis.grid(True)
                ax.set_xticks([y + 1 for y in range(len(all_data))])
                ax.set_xlabel('Two separate samples')
                ax.set_ylabel('Observed values')
            
            # add x-tick labels
            plt.setp(axs, xticks=[y + 1 for y in range(len(all_data))],
                     xticklabels=['summary_bart_score', 'abstract_score'])
            st.pyplot(fig)
        
        
        if st.checkbox('Show scores comparison : '):
            st.subheader('SCORES of the SENTENCES with the SEARCHED WORD function of ABSTRACTS SCORES : ')
            altair_1 = alt.Chart(data_df).mark_circle().encode(x='abstract_score_siebert', y='group_score_siebert', 
                                                    color='abstract_score_siebert', tooltip=['Unnamed: 0','title', 'link', 'type', 'journal'], href='link')
            st.altair_chart(altair_1, use_container_width=True)

            
            st.subheader('SUMMARY SCORES function of ABSTRACTS SCORES : ')
            altair_2 = alt.Chart(data_df).mark_circle().encode(x='summary_bart_score_siebert', y='abstract_score_siebert', 
                                                    color='abstract_score_siebert', tooltip=['Unnamed: 0','title', 'link', 'type', 'journal'], href='link')
            st.altair_chart(altair_2, use_container_width=True)
            
            l_group_s = data_df['group_sentences'].to_list()
            l_abstract_s = data_df['abstracts'].to_list()            
            l_summary_s = data_df['summary_bart'].to_list()
            numero = st.text_input('Number of the abstract :', value='0')
            if int(numero) > 0 :
                st.subheader('GROUP OF SENTENCES')
                st.write(l_group_s[int(numero)])
                st.subheader('ABSTRACT')
                st.write(l_abstract_s[int(numero)])
                st.subheader('SUMMARY')
                st.write(l_summary_s[int(numero)])
        
    elif choice == 'COVID IMPACT' :
            
        #Lecture des données dans des csv
        hosp_0_file = './input/don_hosp_date_df_31082021.csv'
        hosp_0_df = pd.read_csv(hosp_0_file)
        hosp_1_file = './input/don_hosp_1_df_31082021.csv'
        hosp_1_df = pd.read_csv(hosp_1_file)
        
        st.title("Display of hospitalization and resuscitation data due to COVID.")
        
        st.write('All this data was found on a open data warehouse.')
        
        #Altair
        st.subheader("Evolution of the number of resuscitation and hospitalization.")
        fig1, ax = plt.subplots()
        ax.plot(hosp_0_df['rea']) #, don_hosp_date_df.index())
        
        ax.plot(hosp_0_df['hosp']) #, don_hosp_date_df.index())
        
        ax.set(xlabel='jour', ylabel='Number of patients : resuscitation (blue) hospitalisation (orange)',
               title='Evolution of the number of patients due to COVID in hospital')

        #ax.grid()
        
        #fig.savefig("test.png")
        st.pyplot(fig1)
        
        

        #st.subheader("Evolution du rapport hosp/rea.")
        hosp_0_df['rap_hosp_rea']=hosp_0_df['hosp']/hosp_0_df['rea']
        fig2, ax = plt.subplots()
        ax.plot(hosp_0_df['rap_hosp_rea']) #, don_hosp_date_df.index())
        
        
        ax.set(xlabel='jour', ylabel='rapport du nombre personnes  hospitalisées / en réanimation ',
               title='Evolution du rapport')
        #ax.grid()
        
        #fig.savefig("test.png")
        #st.pyplot(fig2)
        
        st.subheader("Evolution fo the number of resuscitation / area.")
        
        # Recuperation des dates:
        l_jours = hosp_1_df['jour'].unique()
        #l_jours.shape
        #st.write(hosp_1_df.columns)
        # Creation du dictionire des regions
        l_regions = hosp_1_df['nomReg'].unique()
        
        choice_regions = st.multiselect("Regions",l_regions, default=l_regions)
        
        dic_regions = {}
        for region in choice_regions[:] :
          reg_df = hosp_1_df[hosp_1_df['nomReg'] == region]
          l_rea = reg_df['rea'].to_list()
          dic_regions[region]= l_rea
          
        fig3, ax = plt.subplots()
        ax.stackplot(l_jours, dic_regions.values(),
                     labels=dic_regions.keys())
        #ax.legend(loc='upper left')
        ax.legend(loc='right', bbox_to_anchor=(1., 0.5, 0.3, 0.5))
        
        xticks = [l_jours[i] for i in np.arange(0, len(l_jours), 100)]
        ax.set_xticks(xticks)
        ax.set_title('Resuscitation by area')
        ax.set_xlabel('day')
        ax.set_ylabel('COVID patients')
        st.pyplot(fig3)

        
        st.title("Display of hospitalization operations on a test year.")
        
        st.write('All this data was found on the test/practice wharehouse of the health data hub. ' +
                 'They are extracted from the PMSI/MCO tables.')

        #Lecture des données dans les csv
        mco_0_file = './input/data_1_df_30082021.csv'
        mco_0_df = pd.read_csv(mco_0_file)
        
        #Affichage des contenus des tables
        l_tables_cat=['A', 'B', 'C', 'D', 'E']
        l_tables_texte = ['Acte CCAM', 'Description séjour', 'NIR patient et date de soin', 'Diagnostique associé', 'Etablissement']
        for i in range(5) : 
            t_mco_df = pd.read_csv('./input/T_MCOaa_nn{}.csv'.format(l_tables_cat[i])) 
            # Traces
            if st.checkbox('Show columns {}'.format(l_tables_texte[i])):
                 st.write(t_mco_df.columns)

        
        mco_df = mco_0_df[['CDC_ACT', 'reg_name', 'month', 'day']].copy()
        mco_df = mco_df.groupby(['reg_name', 'month']).count()
        # Recuperation des dates:
        l_months = [i for i in range(0, 12)]
        
        # Creation du dictionire des regions
        l_regions_2=['CORSICA','BRETAGNE','ILE_DE_FRANCE','HAUTS_DE_FRANCE','ALSACE_FRANCHE_COMTE','NORMANDIE','AQUITAINE','OCCITANIE','BOURGOGNE', 'OUTRE-MER']
        choice_regions_2 = st.multiselect("Regions",l_regions_2, default=l_regions_2)
        dic_regions = {}
        for region in choice_regions_2 :
          l_nb_op = mco_df.loc[region]['CDC_ACT'].to_list()
          dic_regions[region]= l_nb_op
        
        fig4, ax = plt.subplots()
        ax.stackplot(l_months, dic_regions.values(),
                     labels=dic_regions.keys())
        ax.legend(loc='right', bbox_to_anchor=(1., 0.5, 0.3, 0.5))
        ax.set_title('Number of operations in an area')
        ax.set_xlabel('Month')
        ax.set_ylabel('Operations number')
        
        st.pyplot(fig4)



if __name__ == '__main__':
    main()