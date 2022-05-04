
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random
import itertools

#st.set_page_config(page_title='Prediction_app')


def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataframe (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    
    
def app():
    
    st.write('')
    st.write('')
    
    st.markdown("<h6 style='text-align: right; color: black;'>적용 제품: 중방식, PCM, 일반공업 제품 </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: right; color: black;'>데이터 수집기간: 2020-07 ~ 2022-03 </h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: right; color: black;'>총 데이터 갯수: 279 조건 </h6>", unsafe_allow_html=True)

    st.write("")

    
    with st.expander("Predict New Conditions Guide"):
        st.write(
                "1. 정확도 확인 : Model accuracy 버튼 클릭.\n"
                "2. 조색제 투입량에 따른 색차 예측과 초기 색차에 따른 조색제 량 예측.\n"
        )


    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
  
    #---------------------------------#
    # Main panel

    # Displays the dataset

    # Displays the dataset
    
    st.sidebar.header('1. 학습 데이터 및 예측 모델 ')
    st.sidebar.markdown("<h5 style='text-align: center; color: black;'> 파일 자동 업로드 </h5>", unsafe_allow_html=True)
    
    st.subheader('**1. 학습 데이터 및 예측 모델**')
    st.write('')

    #if uploaded_file is not None:
    #    def load_csv():
    #        csv = pd.read_csv(uploaded_file)
    #        return csv
    df = pd.read_csv('https://raw.githubusercontent.com/gigantic65/Color/main/train.csv')
    
        
    x = list(df.columns[:-3])
    y = list(df.columns[df.shape[1]-3:])

        #Selected_X = st.sidebar.multiselect('X variables', x, x)
        #Selected_y = st.sidebar.multiselect('Y variables', y, y)
            
    Selected_X = np.array(x)
    Selected_y = np.array(y)
        
    st.write('**1.1 X인자 수:**',Selected_X.shape[0],'**학습된 조색제 수:**',Selected_X.shape[0]-6)
    st.info(list(Selected_X))

    
    st.write('**1.2 Y인자 수:**',Selected_y.shape[0])
    st.info(list(Selected_y))

    df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
        #df2 = df[df.columns == Selected_X]
    
        #Selected_xy = np.array((Selected_X,Selected_y))

    st.write(df2)

   


        #if uploaded_file2 is not None:
    def load_model(model):
        loaded_model = pickle.load(model)
        return loaded_model
            
    
    with open('https://github.com/gigantic65/Color/blob/main/M_single.pickle', 'rb') as f:
        models = pickle.load(f)
    
            
    st.write('')   
            
    st.write('**1.3 선정된 최적 예측 모델 :**')
    st.write(models)
    st.write('')
            
        
    st.write('**1.4 예측모델 정확도 :**')
    X = df[Selected_X]
    y = df[Selected_y]
            
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
            
            
        
    results = []

    msg = []
    mean = []
    std = []
    names = []
    Title = []
    MAE = []
    MSE = []
    R2 = []



    model = models
    name = 'LE'
            

            
            
            
            
            
    if st.button('Check Model Accuracy'):
                
            
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        results.append(abs(cv_results))
        names.append(name)
        #    names.append(name)
        msg.append('%s' % (name))
        mean.append('%f' %  (abs(cv_results.mean())))
        std.append('%f' % (cv_results.std()))
#    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    
    
        model = model
        model.fit(X_train, y_train)
                
        predictions = model.predict(X_test)
        predictions2 = model.predict(X)
        
        y_test = np.array(y_test)
        y_test = pd.DataFrame(y_test)
        
        predictions = pd.DataFrame(predictions)
        predictions2 = pd.DataFrame(predictions2)
    
    
        R2.append('%f' %  round(sm.r2_score(y,predictions2),5))

   
                    
        F_result3 = pd.DataFrame(np.transpose(msg))
        F_result3.columns = ['Machine_Learning_Model']
        F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
        F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
            #st.write(F_result3)    

        st.write('Model Accuracy for Test data ($R^2$):')
            
        R2_mean = list(F_result3['R2_Mean'].values)
        st.info( R2_mean[0] )
                
        st.write('Model Accuracy for Total data ($R^2$):')
                
        R2 = list(R2)
        st.info( R2[0] )
                        
   
                
        length = range(y_test.shape[0])
                
            #fig, axs = plt.subplots(ncols=3)
            
        fig, ax = plt.subplots(figsize=(10,4))
            
        g = sns.lineplot(x=length,y=predictions[0],color='blue',label='prediction')
        g = sns.lineplot(x=length,y=y_test[0],ax=g.axes, color='red',label='actual')
        g.set_ylabel("Delta L", fontsize = 10)
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
        st.pyplot()
            
        g1 = sns.lineplot(x=length,y=predictions[1],color='blue',label='prediction')
        g1 = sns.lineplot(x=length,y=y_test[1],ax=g1.axes, color='red',label='actual')
        g1.set_ylabel("Delta a", fontsize = 10)
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
        st.pyplot()
            
        g2 = sns.lineplot(x=length,y=predictions[2],color='blue',label='prediction')
        g2 = sns.lineplot(x=length,y=y_test[2],ax=g2.axes, color='red',label='actual')
        g2.set_ylabel("Delta b", fontsize = 10)
        plt.legend()
            
        st.set_option('deprecation.showPyplotGlobalUse', False)
            
            
        st.pyplot()
                
            
 
 
                        
            
        df2 = df[Selected_X]
        columns = df2.columns
    

    #st.sidebar.write('3.1 Predict Single Condition')
            
    
    st.sidebar.header('2. 색차 및 조색제 예측 선택')
                            
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')
                
    select = ['Select','색차 예측','조색제 예측']
    selected = st.sidebar.selectbox("예측 모델 선정 : ", select)


                
    st.write('')
    st.write('')
    st.write('')        
                             
    Target_n = []
    Target_v = []
            
    st.subheader('**2. 색차 및 조색제 예측**')
                
            
    if selected == '색차 예측':
                
        st.write('**2.1 색차 예측**')
                
        select2 = ['Select','중방식 도료','PCM 도료','일반공업 도료']
        selected2 = st.selectbox("제품군 선택 : ", select2)
                
                
                
        if selected2 =='중방식 도료':
                    
            Targets = ['Target_L','Target_a','Target_b']
                    
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Targets[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_L')
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Targets[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_a')
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Targets[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_b')
                Target_v.append(value3)
                        
            Initial = ['Ini_DL','Ini_Da','Ini_Db']
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Initial[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[0])
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Initial[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[1])
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Initial[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[2])
                Target_v.append(value3)
                        
                        
                        
                    
            color_list = ['EP174(T)PTA-BLACK', 'EP174(T)PTA-BLUE', 'YE2308V', 'YE2326G', 'YE2330B', 'YE2335Y' ,'YE2336W'
                          ,'YE2339K', 'YE2347Y', 'YE2391R' ,'YM0407K', 'YM0410G', 'YM0410Y', 'YM0422Y', 'YM0430R', 'YX0305V', 'YX0315G'
                          ,'YX0328Y' ,'YX0337R' ,'YX0397K'  ]
            colors = st.multiselect('조색제 선택',color_list)
                    
            for color1 in colors:
                value = st.number_input(color1,0.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(color1)
                Target_v.append(value)
                        
                    
                    
            New_x = pd.DataFrame([Target_v],columns=list(Target_n))
                    
            New_x2 = pd.DataFrame(X_train.iloc[0,:])
                
            New_x2 = New_x2.T
                    #New_x2 = 0.0
                    
                    #st.write(New_x2)
                    
            col1,col2 = st.columns([1,1])
                    
                
      
            if st.button('Run Prediction'): 
                        
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                    for col2 in New_x.columns:
                        if col == col2:
                            New_x2[col] = New_x[col2].values
                            #st.write(New_x2[col])     
                                
                New_x2.index = ['New_case']        
                
                st.write(New_x2)

                
                model.fit(X_train, y_train)
                        
                predictions = model.predict(New_x2)
                        
                predictions = pd.DataFrame(predictions,columns = ['Delta_L','Delta_a','Delta_b'])
                
                predictions['Delta_E'] = (predictions['Delta_L']**2+predictions['Delta_a']**2+predictions['Delta_b']**2)**0.5 
                        
                    
                st.write('**2.2 색차 예측 결과**')
                        
                predictions.index = ['Results']      
                        
                st.write(predictions)
                        
                

        if selected2 =='PCM 도료':
                    
            Targets = ['Target_L','Target_a','Target_b']
                    
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Targets[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_L')
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Targets[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_a')
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Targets[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_b')
                Target_v.append(value3)
                    
            Initial = ['Ini_DL','Ini_Da','Ini_Db']
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Initial[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[0])
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Initial[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[1])
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Initial[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[2])
                Target_v.append(value3)
                        
                        
                        
                    
            color_list = ['YE2204Y', 'YE2205V', 'YE2206B', 'YE2209R' ,'YE2210G', 'YE2210K', 'YE2231W' ,'YE2263Y'
                          ,'YF3401K', 'YF3401V' ,'YF3406B', 'YF3410K', 'YF3414Y', 'YF3416Y', 'YF3417G' ,'YF3429W'
                          ,'YF3430R' ]
            colors = st.multiselect('조색제 선택',color_list)
                    
            for color1 in colors:
                value = st.number_input(color1,0.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(color1)
                Target_v.append(value)
                        
                    
                    
            New_x = pd.DataFrame([Target_v],columns=list(Target_n))
                
            New_x2 = pd.DataFrame(X_train.iloc[0,:])
                    
            New_x2 = New_x2.T
                    #New_x2 = 0.0
                    
                    #st.write(New_x2)
                    
            if st.button('Run Prediction'): 
                        
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                    for col2 in New_x.columns:
                        if col == col2:
                            New_x2[col] = New_x[col2].values
                            #st.write(New_x2[col])     
                            
                New_x2.index = ['New_case']        
                    
                st.write(New_x2)
                
                model.fit(X_train, y_train)
                
                predictions = model.predict(New_x2)
                        
                predictions = pd.DataFrame(predictions,columns = ['Delta_L','Delta_a','Delta_b'])
                
                predictions['Delta_E'] = (predictions['Delta_L']**2+predictions['Delta_a']**2+predictions['Delta_b']**2)**0.5 
                        
                    
                st.write('**2.2 색차 예측 결과**')
                        
                predictions.index = ['Results']      
                        
                st.write(predictions)
                

        if selected2 =='일반공업 도료':
                    
            Targets = ['Target_L','Target_a','Target_b']
                    
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Targets[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_L')
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Targets[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_a')
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Targets[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_b')
                Target_v.append(value3)
                        
            Initial = ['Ini_DL','Ini_Da','Ini_Db']
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Initial[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[0])
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Initial[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[1])
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Initial[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[2])
                Target_v.append(value3)
                        
                        
                        
                    
            color_list = ['YE2326G', 'YE2339K', 'YE2347Y', 'YW2698K', 'YX1205V' ,'YX1210O', 'YX1222W', 'YX1228Y' ,'YX1229Y'
                              ,'YY1103G', 'YY1111R' ,'YY1112B', 'YY1116Y', 'YY1120K', 'YY1137W']
            colors = st.multiselect('조색제 선택',color_list)
                    
            for color1 in colors:
                value = st.number_input(color1,0.0, 1000.00, 0.0,format="%.2f")
                Target_n.append(color1)
                Target_v.append(value)
                        
                    
                    
            New_x = pd.DataFrame([Target_v],columns=list(Target_n))
                    
            New_x2 = pd.DataFrame(X_train.iloc[0,:])
            
            New_x2 = New_x2.T
                    #New_x2 = 0.0
                    
                    #st.write(New_x2)
                    
            if st.button('Run Prediction'): 
                        
                for col in New_x2.columns:
                    New_x2[col] = 0.0
                    for col2 in New_x.columns:
                        if col == col2:
                            New_x2[col] = New_x[col2].values
                            #st.write(New_x2[col])     
                                
                New_x2.index = ['New_case']        
                        
                st.write(New_x2)
                
                model.fit(X_train, y_train)
                    
                predictions = model.predict(New_x2)
                        
                predictions = pd.DataFrame(predictions,columns = ['Delta_L','Delta_a','Delta_b'])
                
                predictions['Delta_E'] = (predictions['Delta_L']**2+predictions['Delta_a']**2+predictions['Delta_b']**2)**0.5 
                        
                    
                st.write('**2.2 색차 예측 결과**')
                        
                predictions.index = ['Results']      
                        
                st.write(predictions)

                        
                  
                
        
            
        
        
    if selected == '조색제 예측':
 
        st.write('**2.3 조색제 예측**')
                
        select3 = ['Select','중방식 도료','PCM 도료','일반공업 도료']
        selected3 = st.selectbox("제품군 선택 : ", select3)
                
                
        if selected3 =='중방식 도료':
                    
            Targets = ['Target_L','Target_a','Target_b']
                    
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Targets[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_L')
                
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Targets[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_a')
                
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Targets[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_b')
                
                Target_v.append(value3)
                        
            Initial = ['Ini_DL','Ini_Da','Ini_Db']
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Initial[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[0])
                
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Initial[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[1])
                
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Initial[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[2])
                
                Target_v.append(value3)
                    
                
                
                
            color_list = ['EP174(T)PTA-BLACK', 'EP174(T)PTA-BLUE', 'YE2308V', 'YE2326G', 'YE2330B', 'YE2335Y' ,'YE2336W'
                          ,'YE2339K', 'YE2347Y', 'YE2391R' ,'YM0407K', 'YM0410G', 'YM0410Y', 'YM0422Y', 'YM0430R', 'YX0305V', 'YX0315G'
                          ,'YX0328Y' ,'YX0337R' ,'YX0397K'  ]
            
            colors = st.multiselect('조색제 선택',color_list)
                    
                    
            name2=[]
            test2=[]
                    
            for color2 in colors:
                        
                max1 = round(float(df[color2].max()),3)
                min1 = round(float(df[color2].min()),3)
                        
                rag1 = round(min1,3)
                rag2 = round(max1,3)
                        
                step = round((max1-min1)/20.0,3)
                        
                value = st.slider(color2, min1, max1, (rag1,rag2), step)
                        
                        
                name2.append(color2)
                test2.append(value)
                        
                        
                        
            count = 0
                    
            para = []
            para2 = []
            para4 = []
                    
                    

                    
            if st.button('Run Prediction',key = count):
                        
                para = []
                para0 = []
                para1 = []
                para2 = []
                para4 = []
                                                                        
                
                New_x = pd.DataFrame([Target_v],columns=list(Target_n))
 
                for col in New_x.columns:
                    para0 = itertools.repeat(float(New_x[col].values),100)
                    para1.append(para0)

 
                para1 = pd.DataFrame(para1)
                para1 = para1.T
                para1 = para1.dropna().reset_index()
                para1.drop(['index'],axis=1,inplace=True)
                
                para1.columns = ['Std_L','Std_a','Std_b','Ini_DL','Ini_Da','Ini_Db']
                

               
                
                for para in test2:
                    if para[0] == para[1]:
                        para = itertools.repeat(para[0],100)
                    else:
                        para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
                            #st.write(para)
                    para2.append(para)
                            
                        
                    
                    
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                para2 = para2.dropna().reset_index()
                
                para2.drop(['index'],axis=1,inplace=True)
                
                para2.columns = list(name2)
                        
                Iter2 = para2.shape[1]
                    
                Iter = 100
                    
                para2 = pd.concat([para1,para2], axis=1)
                
                

                New_x2 = pd.DataFrame(X_train.iloc[0,:])
                New_x2 = New_x2.T
                
                para3 = []
                                    
                for i in range(Iter):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                        for col2 in para2.columns:
                            if col == col2:
                                New_x2[col] = random.sample(list(para2[col2]),1)
                        
                        para5.append(float(New_x2[col].values))
                   
                    para3.append(para5)
                        

                para3 = pd.DataFrame(para3) 

                
                para4 = para3
                        

                
                para4 = pd.DataFrame(para4)
                    
                    
                para4.columns=list(Selected_X)
                    
                st.write('**Selected Process Condtions:**')
                
                para4 = para4.dropna()
                
                
                
                
                datafile = para4
        
                
                model.fit(X_train, y_train)
                
                predictions2 = model.predict(datafile)
        
                predictions2 = pd.DataFrame(predictions2, columns=['Final_DL','Final_Da','Final_Db'])
                
                predictions2['Final_De'] = (predictions2['Final_DL']**2+predictions2['Final_Da']**2+predictions2['Final_Db']**2)**0.5
        
                #st.write(predictions2)
                
                ini_color = ['Std_L','Std_a','Std_b','Ini_DL','Ini_Da','Ini_Db']
                
                #Ini_para = para4[ini_color] 
                
                para4 = pd.concat([para4[ini_color],para4[colors]], axis=1)
                
                para4 = para4.dropna()
                
                st.write(para4)
                      
                para4 = pd.concat([para4,predictions2], axis=1)
                    

                para4.sort_values(by='Final_De', ascending=True, inplace =True)
                    
                para4 = para4.reset_index()
                para4.drop(['index'],axis=1,inplace=True)
                    
                    

                st.write('')
                st.write('')
                st.write('')
                st.markdown("<h6 style='text-align: left; color: darkblue;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)

                df_min = para4[para4['Final_De']==para4['Final_De'].min()]

                st.write('**Optimize Process Conditions:**')
                st.write(df_min)

                        #st.info(list(Selected_X2))
                        
                st.write('')
                st.write('**Total results:**')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='red')
                st.pyplot()
                st.write(para4)
                
                             
                
            
    
    
                st.markdown('**Download Predicted Results for Multi Conditions**')
        
                st_pandas_to_csv_download_link(para4, file_name = "Predicted_Results.csv")
                st.write('*Save directory setting : right mouse button -> save link as') 


                
        if selected3 =='PCM 도료':
                    
            Targets = ['Target_L','Target_a','Target_b']
                    
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Targets[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_L')
                
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Targets[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_a')
                
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Targets[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_b')
                
                Target_v.append(value3)
                        
            Initial = ['Ini_DL','Ini_Da','Ini_Db']
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Initial[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[0])
                
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Initial[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[1])
                
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Initial[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[2])
                
                Target_v.append(value3)
                    
                
                
                
            color_list = ['YE2204Y', 'YE2205V', 'YE2206B', 'YE2209R' ,'YE2210G', 'YE2210K', 'YE2231W' ,'YE2263Y'
                          ,'YF3401K', 'YF3401V' ,'YF3406B', 'YF3410K', 'YF3414Y', 'YF3416Y', 'YF3417G' ,'YF3429W'
                          ,'YF3430R' ]
            
            colors = st.multiselect('조색제 선택',color_list)
                    
                    
            name2=[]
            test2=[]
                    
            for color2 in colors:
                        
                max1 = round(float(df[color2].max()),3)
                min1 = round(float(df[color2].min()),3)
                        
                rag1 = round(min1,3)
                rag2 = round(max1,3)
                        
                step = round((max1-min1)/20.0,3)
                        
                value = st.slider(color2, min1, max1, (rag1,rag2), step)
                        
                        
                name2.append(color2)
                test2.append(value)
                        
                        
                        
            count = 0
                    
            para = []
            para2 = []
            para4 = []
                    
                    
   
                    
            if st.button('Run Prediction',key = count):
                        
                para = []
                para0 = []
                para1 = []
                para2 = []
                para4 = []
                                                                        
                
                New_x = pd.DataFrame([Target_v],columns=list(Target_n))
 
                for col in New_x.columns:
                    para0 = itertools.repeat(float(New_x[col].values),100)
                    para1.append(para0)

 
                para1 = pd.DataFrame(para1)
                para1 = para1.T
                para1 = para1.dropna().reset_index()
                para1.drop(['index'],axis=1,inplace=True)
                
                para1.columns = ['Std_L','Std_a','Std_b','Ini_DL','Ini_Da','Ini_Db']
                

               
                
                for para in test2:
                    if para[0] == para[1]:
                        para = itertools.repeat(para[0],100)
                    else:
                        para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
                            #st.write(para)
                    para2.append(para)
                            
                        
                    
                    
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                para2 = para2.dropna().reset_index()
                
                para2.drop(['index'],axis=1,inplace=True)
                
                para2.columns = list(name2)
                        
                Iter2 = para2.shape[1]
                    
                Iter = 100
                    
                para2 = pd.concat([para1,para2], axis=1)
                
                

                New_x2 = pd.DataFrame(X_train.iloc[0,:])
                New_x2 = New_x2.T
                
                para3 = []
                                    
                for i in range(Iter):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                        for col2 in para2.columns:
                            if col == col2:
                                New_x2[col] = random.sample(list(para2[col2]),1)
                        
                        para5.append(float(New_x2[col].values))
                   
                    para3.append(para5)
                        

                para3 = pd.DataFrame(para3) 

                
                para4 = para3
                        

                
                para4 = pd.DataFrame(para4)
                    
                    
                para4.columns=list(Selected_X)
                    
                st.write('**Selected Process Condtions:**')
                
                para4 = para4.dropna()
                
                
                
                
                datafile = para4
        
                
                model.fit(X_train, y_train)
                
                predictions2 = model.predict(datafile)
        
                predictions2 = pd.DataFrame(predictions2, columns=['Final_DL','Final_Da','Final_Db'])
                
                predictions2['Final_De'] = (predictions2['Final_DL']**2+predictions2['Final_Da']**2+predictions2['Final_Db']**2)**0.5
        
                #st.write(predictions2)
                
                ini_color = ['Std_L','Std_a','Std_b','Ini_DL','Ini_Da','Ini_Db']
                
                #Ini_para = para4[ini_color] 
                
                para4 = pd.concat([para4[ini_color],para4[colors]], axis=1)
                para4 = para4.dropna()
                
                st.write(para4)
                      
                para4 = pd.concat([para4,predictions2], axis=1)
                    

                para4.sort_values(by='Final_De', ascending=True, inplace =True)
                    
                para4 = para4.reset_index()
                para4.drop(['index'],axis=1,inplace=True)
                    
                    

                st.write('')
                st.write('')
                st.write('')
                st.markdown("<h6 style='text-align: left; color: darkblue;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)

                df_min = para4[para4['Final_De']==para4['Final_De'].min()]

                st.write('**Optimize Process Conditions:**')
                st.write(df_min)

                        #st.info(list(Selected_X2))
                        
                st.write('')
                st.write('**Total results:**')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='red')
                st.pyplot()
                st.write(para4)
                
                             
                
            
    
    
                st.markdown('**Download Predicted Results for Multi Conditions**')
        
                st_pandas_to_csv_download_link(para4, file_name = "Predicted_Results.csv")
                st.write('*Save directory setting : right mouse button -> save link as') 
                
                
                
        if selected3 =='일반공업 도료':
                    
            Targets = ['Target_L','Target_a','Target_b']
                    
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Targets[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_L')
                
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Targets[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_a')
                
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Targets[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append('Std_b')
                
                Target_v.append(value3)
                        
            Initial = ['Ini_DL','Ini_Da','Ini_Db']
                    
            col1,col2,col3 = st.columns([1,1,1])
            with col1:
                value1 = st.number_input(Initial[0], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[0])
                
                Target_v.append(value1)
            with col2:
                value2 = st.number_input(Initial[1], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[1])
                
                Target_v.append(value2)
            with col3:
                value3 = st.number_input(Initial[2], -1000.00, 1000.00, 0.0,format="%.2f")
                Target_n.append(Initial[2])
                
                Target_v.append(value3)
                    
                
                
                
            color_list = ['YE2326G', 'YE2339K', 'YE2347Y', 'YW2698K', 'YX1205V' ,'YX1210O', 'YX1222W', 'YX1228Y' ,'YX1229Y'
                              ,'YY1103G', 'YY1111R' ,'YY1112B', 'YY1116Y', 'YY1120K', 'YY1137W']
            
            colors = st.multiselect('조색제 선택',color_list)
                    
                    
            name2=[]
            test2=[]
                    
            for color2 in colors:
                        
                max1 = round(float(df[color2].max()),3)
                min1 = round(float(df[color2].min()),3)
                        
                rag1 = round(min1,3)
                rag2 = round(max1,3)
                        
                step = round((max1-min1)/20.0,3)
                        
                value = st.slider(color2, min1, max1, (rag1,rag2), step)
                        
                        
                name2.append(color2)
                test2.append(value)
                        
                        
                        
            count = 0
                    
            para = []
            para2 = []
            para4 = []
                    
                    

                    
            if st.button('Run Prediction',key = count):
                        
                para = []
                para0 = []
                para1 = []
                para2 = []
                para4 = []
                                                                        
                
                New_x = pd.DataFrame([Target_v],columns=list(Target_n))
 
                for col in New_x.columns:
                    para0 = itertools.repeat(float(New_x[col].values),100)
                    para1.append(para0)

 
                para1 = pd.DataFrame(para1)
                para1 = para1.T
                para1 = para1.dropna().reset_index()
                para1.drop(['index'],axis=1,inplace=True)
                
                para1.columns = ['Std_L','Std_a','Std_b','Ini_DL','Ini_Da','Ini_Db']
                

               
                
                for para in test2:
                    if para[0] == para[1]:
                        para = itertools.repeat(para[0],100)
                    else:
                        para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
                            #st.write(para)
                    para2.append(para)
                            
                        
                    
                    
                para2 = pd.DataFrame(para2)
                para2 = para2.T
                para2 = para2.dropna().reset_index()
                
                para2.drop(['index'],axis=1,inplace=True)
                
                para2.columns = list(name2)
                        
                Iter2 = para2.shape[1]
                    
                Iter = 100
                    
                para2 = pd.concat([para1,para2], axis=1)
                
                

                New_x2 = pd.DataFrame(X_train.iloc[0,:])
                New_x2 = New_x2.T
                
                para3 = []
                                    
                for i in range(Iter):
                    para5 = []
                    for col in New_x2.columns:
                        New_x2[col] = 0.0
                        for col2 in para2.columns:
                            if col == col2:
                                New_x2[col] = random.sample(list(para2[col2]),1)
                        
                        para5.append(float(New_x2[col].values))
                   
                    para3.append(para5)
                        

                para3 = pd.DataFrame(para3) 

                
                para4 = para3
                        

                
                para4 = pd.DataFrame(para4)
                    
                    
                para4.columns=list(Selected_X)
                    
                st.write('**Selected Process Condtions:**')
                
                para4 = para4.dropna()
                
                
                
                
                datafile = para4
        
                
                model.fit(X_train, y_train)
                
                predictions2 = model.predict(datafile)
        
                predictions2 = pd.DataFrame(predictions2, columns=['Final_DL','Final_Da','Final_Db'])
                
                predictions2['Final_De'] = (predictions2['Final_DL']**2+predictions2['Final_Da']**2+predictions2['Final_Db']**2)**0.5
        
                #st.write(predictions2)
                
                ini_color = ['Std_L','Std_a','Std_b','Ini_DL','Ini_Da','Ini_Db']
                
                #Ini_para = para4[ini_color] 
                
                para4 = pd.concat([para4[ini_color],para4[colors]], axis=1)
                
                st.write(para4)
                      
                para4 = pd.concat([para4,predictions2], axis=1)
                    
                
                para4 = para4.dropna()
                
                para4.sort_values(by='Final_De', ascending=True, inplace =True)
                
                para4 = para4.reset_index()
                
                para4.drop(['index'],axis=1,inplace=True)
                    
                    

                st.write('')
                st.write('')
                st.write('')
                st.markdown("<h6 style='text-align: left; color: darkblue;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)

                df_min = para4[para4['Final_De']==para4['Final_De'].min()]

                st.write('**Optimize Process Conditions:**')
                st.write(df_min)

                        #st.info(list(Selected_X2))
                        
                st.write('')
                st.write('**Total results:**')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='red')
                st.pyplot()
                st.write(para4)
                
                             
                
            
    
    
                st.markdown('**Download Predicted Results for Multi Conditions**')
        
                st_pandas_to_csv_download_link(para4, file_name = "Predicted_Results.csv")
                st.write('*Save directory setting : right mouse button -> save link as') 
            #st.write(mi)
        #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
  
