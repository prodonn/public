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



import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

import altair as alt


def draw_bar_plot(df, l_fields) :
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    
    ax1.hist(df[l_fields[0]], bins=20)
    ax1.set_title(l_fields[0])
    ax2.hist(df[l_fields[1]], bins=20)
    ax2.set_title(l_fields[1])
    ax3.hist(df[l_fields[2]], bins=20)
    ax3.set_title(l_fields[2])
    
    st.pyplot(fig)
    
    
def main():
    
    # Remplissage de la sidebar
    st.sidebar.image('./data/sentiment-analysis.jpg')
    
   
    # Choix de la page à lire 
    pages = ['HOME', 'COMPARE MODELS', 'SIEBERT SENTIMENT ANALYSIS', 'FINBERT SENTIMENT ANALYSIS']
    choice = st.sidebar.selectbox("Pages",pages)
     

    search_word=''
    
    
    if choice == 'HOME' :
        st.title('COMPARE MODELS')
        st.write('This is the calibration step. I ve spent time working on models, beginning with the word2vec, ' +
                 'then deep learning basis cnn model, finishing with the most sharp huggingface model based on BERT.' +
                 'This stage presents the obtained results with the 2 huggingface models : ' +
                 'siebert/sentiment-roberta-large-english and ProsAI/finbert.' + 
                 'Siebert has already been evaluated on an annotated IMDB dataset with a score of 95%.' +
                 'Siebert is leading to answers of POSITIVE/NEGATIVE sentiments ' +
                 'whereas Finbert is more sharp giving POSITIVE/NEGATIVE/NEUTRAL answers.')
        st.subheader('SIEBERT SENTIMENT ANALYSIS')
        st.write('This analysis stage will show the results obtained with the most efficient text classification model ' +
                  'based on BERT using the huggingFace transformers: siebert/sentiment-roberta-large-english. ' +
                  'This model is a fine-tuned checkpoint of RoBERTa-large (Liu et al. 2019). '+
                  'It enables reliable binary sentiment analysis for various types of English-language text. ' +
                  'For each instance, it predicts either positive (1) or negative (0) sentiment. '+
                  'The model was fine-tuned and evaluated on 15 data sets from diverse text sources to enhance generalization'+
                  ' across different types of texts (reviews, tweets, etc.). Consequently, it outperforms models trained on '+ 
                  'only one type of text (e.g., movie reviews from the popular SST-2 benchmark) when used on new data as shown below. ' +
                  'Furthermore, the model is applied on the whole abstract, on a selected group of sentences in which the searched word appears and a summary done by a HuggingFace '+
                  'transformers based on model facebook/bart-large-cnn.')
        st.subheader('FINBERT SENTIMENT ANALYSIS')
        st.write('This analysis stage will show the results obtained with an efficient text classification model ' +
                  'based on BERT using the huggingFace transformers: ProsAI/finbert. ' +
                  'It enables reliable Positve/Negative/Neutral sentiment analysis for various types of English-language text. ' +
                  'The model was fine-tuned and evaluated on a data sets from financial sources.')
       
                 
    elif choice == 'COMPARE MODELS' :
        st.title('MEDICAL ARTICLES CONCLUSION SENTIMENT ANALYSIS COMPARISON')    
        
        st.write('At this stage, I am considering and making comparisons between the two most models, Siebert and Finbert.')
        #Lecture des données dans des csv
        file_path_jco = './data/jco_scores_gpu_df_bevacizumab_30092021.csv'
        jco_df = pd.read_csv(file_path_jco)
        #Lecture des colonnes
        st.write("Colonnes du dataframe jco_conclusions : ")
        st.write(jco_df.columns)
        jco_df['siebert_pred'] = [np.argmax(np.array(l)) for l in jco_df[['siebert_prob_negative', 'siebert_prob_positive',]].values.tolist()]
        jco_df['finbert_pred'] = [np.argmax(np.array(l)) for l in jco_df[['finbert_prob_positive','finbert_prob_negative','finbert_prob_neutral']].values.tolist()]

        #Altair
        st.subheader('SCORES comparison of the 2 models.') 
        #altair_1 = alt.Chart(jco_df).mark_point().encode(x='siebert_pred', y='finbert_pred', color='siebert_pred:N', shape = 'siebert_pred', tooltip=['conclusion'])
        
        altair_1 = alt.Chart(jco_df).mark_point().encode(x='siebert_logit_negative', y='finbert_logit_positive', color='siebert_pred:N', shape='siebert:O', tooltip=['conclusion'])
        st.altair_chart(altair_1, use_container_width=True)
        
        #sun ray graph
        st.subheader('MODELS SCORES DISTRIBUTION.')
        fig_sun = px.sunburst(jco_df, path=['finbert_pred','siebert_pred'], 
                             color_discrete_map={'(?)':'black', 0:'red', 1:'darkblue'}, 
                             title='Center circle : finbert, second : siebert', 
                             width=600, height=800)
        
        st.plotly_chart(fig_sun, use_container_width=True)
        
    elif choice == 'SIEBERT SENTIMENT ANALYSIS' :
        
        #Lecture des données dans des csv
        file_path_jco = './data/jco_scores_gpu_df_bevacizumab_30092021.csv'
        jco_df = pd.read_csv(file_path_jco)
        
        # Mot cherché
        search_word=file_path.split('_')[3]
        st.write('Searched word :')
        st.write('bevacizumab')
        
        #Type articles
        #l_types = abstract_df['type'].unique()

        #choice_type = st.multiselect("Type",l_types, default=l_types)
        
        #Type journaux
        #l_journaux = abstract_df['journal'].unique()
        #choice_journal = st.multiselect("Journal",l_journaux, default=l_journaux)
        
        # Traces
        #if st.checkbox('Show columns'):
          #  st.write(abstract_df.columns)
        #st.write(choice_journal)
        
        # Reduction du dataset de travail
        #data_df = abstract_df[(abstract_df['type'].isin(choice_type)) & (abstract_df['journal'].isin(choice_journal))]
        
        #if st.checkbox('Show number of rows'):
          #  st.write('Number of rows in the dataframe {}'.format(data_df.shape[0]))

        #Représentation de la répartition des différentes valeurs
        #st.subheader('Score distribution (logits) / siebert model : ')
        
        #if st.checkbox('Show logits distribution / bar : '):
         #   draw_bar_plot(data_df, ['abstract_score_siebert', 'group_score_siebert', 'summary_bart_score_siebert' ])

        
        # generate some random test data
        #all_data = [data_df['summary_bart_score_siebert'].to_numpy(), 
         #           data_df['abstract_score_siebert'].to_numpy()]
        
        
        #if st.checkbox('Show logits distribution / violin and box : '):
         #   fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
            # plot violin plot
          #  axs[0].violinplot(all_data,
                #              showmeans=False,
                 #             showmedians=True)
            #axs[0].set_title('Violin plot')
            
            # plot box plot
            #axs[1].boxplot(all_data)
            #axs[1].set_title('Box plot')
            
            # adding horizontal grid lines
            #for ax in axs:
             #   ax.yaxis.grid(True)
             #   ax.set_xticks([y + 1 for y in range(len(all_data))])
             #   ax.set_xlabel('Two separate samples')
             #   ax.set_ylabel('Observed values')
            
            # add x-tick labels
            #plt.setp(axs, xticks=[y + 1 for y in range(len(all_data))],
             #        xticklabels=['summary_bart_score', 'abstract_score'])
            #st.pyplot(fig)
        
        
        #if st.checkbox('Show scores comparison : '):
         #   st.subheader('SCORES of the SENTENCES with the SEARCHED WORD function of ABSTRACTS SCORES : ')
          #  altair_1 = alt.Chart(data_df).mark_circle().encode(x='abstract_score_siebert', y='group_score_siebert', 
            #                                        color='abstract_score_siebert', tooltip=['Unnamed: 0','title', 'link', 'type', 'journal'], href='link')
           # st.altair_chart(altair_1, use_container_width=True)

            
            #st.subheader('SUMMARY SCORES function of ABSTRACTS SCORES : ')
            #altair_2 = alt.Chart(data_df).mark_circle().encode(x='summary_bart_score_siebert', y='abstract_score_siebert', 
             #                                       color='abstract_score_siebert', tooltip=['Unnamed: 0','title', 'link', 'type', 'journal'], href='link')
            #st.altair_chart(altair_2, use_container_width=True)
            
            #l_group_s = data_df['group_sentences'].to_list()
            #l_abstract_s = data_df['abstracts'].to_list()            
            #l_summary_s = data_df['summary_bart'].to_list()
            #numero = st.text_input('Number of the abstract :', value='0')
            #if int(numero) > 0 :
             #   st.subheader('GROUP OF SENTENCES')
              #  st.write(l_group_s[int(numero)])
               # st.subheader('ABSTRACT')
             #   st.write(l_abstract_s[int(numero)])
              #  st.subheader('SUMMARY')
               # st.write(l_summary_s[int(numero)])
        
   


if __name__ == '__main__':
    main()